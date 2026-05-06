"""
Automated Dataset Diagnostics and Repair Framework
Main Flask Application — hardened for production.
"""

import os
import sys
import json
import uuid
import logging
import threading
import traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.diagnostics import DatasetDiagnostics
from modules.repair import DatasetRepair
from modules.model_evaluator import ModelEvaluator
from modules.visualizer import generate_all_charts, plot_model_comparison

# ----- Logging ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("adrf")

# ----- Flask app --------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", 50)) * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "adrf-fyp-dev-only-change-in-prod")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
VERSIONS_FOLDER = os.path.join(BASE_DIR, "versions")
for d in (UPLOAD_FOLDER, REPORTS_FOLDER, VERSIONS_FOLDER):
    os.makedirs(d, exist_ok=True)

# ----- Session store with TTL -------------------------------------------
SESSION_TTL = timedelta(hours=2)
MAX_SESSIONS = 50  # hard cap to prevent unbounded growth
_session_lock = threading.Lock()
sessions = {}


def _prune_sessions():
    """Evict sessions older than TTL and enforce MAX_SESSIONS limit."""
    now = datetime.now()
    with _session_lock:
        expired = [sid for sid, s in sessions.items()
                   if now - s.get("_last_access", now) > SESSION_TTL]
        for sid in expired:
            sessions.pop(sid, None)
            log.info("Evicted expired session %s", sid[:8])
        # If still over cap, evict least-recently-used
        if len(sessions) > MAX_SESSIONS:
            sorted_sids = sorted(sessions.items(), key=lambda kv: kv[1].get("_last_access", now))
            to_remove = len(sessions) - MAX_SESSIONS
            for sid, _ in sorted_sids[:to_remove]:
                sessions.pop(sid, None)


def _get_session(sid):
    _prune_sessions()
    with _session_lock:
        if sid and sid in sessions:
            sessions[sid]["_last_access"] = datetime.now()
            return sessions[sid]
        return None


def _put_session(sid, data):
    with _session_lock:
        data["_last_access"] = datetime.now()
        sessions[sid] = data


# ----- Routes -----------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/logo.png")
def logo():
    logo_path = os.path.join(BASE_DIR, "logo.png")
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype="image/png")
    return "", 404


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "sessions": len(sessions)})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        session_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}.csv")
        file.save(filepath)

        # Defensive read: handle bad encoding / bad separators gracefully
        try:
            df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding="latin-1")
        except pd.errors.ParserError as e:
            return jsonify({"error": f"CSV parse error: {e}"}), 400

        if df.empty:
            return jsonify({"error": "CSV file is empty"}), 400

        # Cap absurdly wide/long datasets for Render free tier
        max_rows = int(os.environ.get("MAX_ROWS", 100_000))
        if len(df) > max_rows:
            log.warning("Dataset has %d rows; sampling down to %d for stability", len(df), max_rows)
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

        _put_session(session_id, {
            "filepath": filepath,
            "filename": file.filename,
            "df": df,
            "uploaded_at": datetime.now().isoformat(),
        })

        return jsonify({
            "session_id": session_id,
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": df.columns.tolist(),
            "preview": df.head(5).fillna("").to_dict("records"),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        })
    except Exception as e:
        log.exception("Upload failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Invalid or expired session"}), 400

    try:
        df = sess["df"]
        diagnostics = DatasetDiagnostics(df)
        results = diagnostics.run_full_diagnostics()
        charts = generate_all_charts(results)
        sess["diagnostics"] = results
        sess["charts"] = charts

        return jsonify({
            "success": True,
            "results": _serialize(results),
            "charts": charts,
        })
    except Exception as e:
        log.exception("Diagnose failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/repair", methods=["POST"])
def repair():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Invalid session"}), 400
    if "diagnostics" not in sess:
        return jsonify({"error": "Run diagnostics first"}), 400

    try:
        df = sess["df"]
        diag_results = sess["diagnostics"]

        performance_guarded = bool(data.get("performance_guarded", True))
        preserve_columns = bool(data.get("preserve_columns", True))
        column_specific = data.get("column_specific_processing", {})

        if data.get("drop_columns"):
            return jsonify({"error": "Column dropping is disabled"}), 400

        engine = DatasetRepair(df, diag_results)
        repaired_df, repair_log, versions = engine.run_auto_repair(
            preserve_columns=preserve_columns,
            column_specific_processing=column_specific,
            performance_guarded=performance_guarded,
        )
        summary = engine.get_repair_summary()
        sess["repaired_df"] = repaired_df
        sess["repair_log"] = repair_log
        sess["repair_summary"] = summary
        sess["explanations"] = engine.explanations

        repaired_path = os.path.join(VERSIONS_FOLDER, f"{sid}_repaired.csv")
        repaired_df.to_csv(repaired_path, index=False)
        sess["repaired_path"] = repaired_path

        return jsonify({
            "success": True,
            "repair_summary": _serialize(summary),
            "repair_log": _serialize(repair_log),
            "explanations": _serialize(engine.explanations),
            "repaired_shape": {"rows": int(repaired_df.shape[0]),
                               "columns": int(repaired_df.shape[1])},
            "original_shape": {"rows": int(df.shape[0]),
                               "columns": int(df.shape[1])},
            "columns_preserved": True,
            "performance_guarded": performance_guarded,
        })
    except Exception as e:
        log.exception("Repair failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"error": "Invalid session"}), 400

    try:
        ev = ModelEvaluator()
        before = ev.train_and_evaluate(sess["df"], label="Before Repair")

        after, comparison, chart = None, None, None
        if "repaired_df" in sess:
            after = ModelEvaluator().train_and_evaluate(sess["repaired_df"], label="After Repair")
            comparison = ev.compare_results(before, after)
            try:
                chart = plot_model_comparison(comparison)
            except Exception:
                log.exception("Comparison chart failed")

        sess["evaluation"] = {"before": before, "after": after, "comparison": comparison}

        return jsonify({
            "success": True,
            "before": _serialize(before),
            "after": _serialize(after),
            "comparison": _serialize(comparison),
            "comparison_chart": chart,
        })
    except Exception as e:
        log.exception("Evaluate failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/download_repaired", methods=["GET"])
def download_repaired():
    sid = request.args.get("session_id")
    sess = _get_session(sid)
    if not sess or "repaired_path" not in sess:
        return jsonify({"error": "No repaired dataset available"}), 400
    return send_file(sess["repaired_path"], as_attachment=True,
                     download_name="repaired_dataset.csv")


@app.route("/api/session_status", methods=["GET"])
def session_status():
    sid = request.args.get("session_id")
    sess = _get_session(sid)
    if not sess:
        return jsonify({"exists": False})
    return jsonify({
        "exists": True,
        "filename": sess.get("filename"),
        "has_diagnostics": "diagnostics" in sess,
        "has_repair": "repaired_df" in sess,
        "has_evaluation": "evaluation" in sess,
    })


# ----- Serialization ----------------------------------------------------

def _serialize(obj, _depth=0):
    """Make objects JSON-serializable with cycle and depth protection."""
    if _depth > 12:
        return "<truncated>"
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _serialize(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(i, _depth + 1) for i in obj]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item") and not isinstance(obj, str):
        try:
            return obj.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(obj, float) and obj != obj:  # NaN
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print("  Automated Dataset Diagnostics & Repair Framework")
    print(f"  Running on http://localhost:{port}  (debug={is_debug})")
    print("=" * 60 + "\n")
    app.run(debug=is_debug, host="0.0.0.0", port=port)
