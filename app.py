"""
Automated Dataset Diagnostics and Repair Framework
Main Flask Application
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.diagnostics import DatasetDiagnostics
from modules.repair import DatasetRepair
from modules.model_evaluator import ModelEvaluator
from modules.visualizer import generate_all_charts, plot_model_comparison

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['SECRET_KEY'] = 'adrf-fyp-2024-secret'

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
REPORTS_FOLDER = os.path.join(os.path.dirname(__file__), 'reports')
VERSIONS_FOLDER = os.path.join(os.path.dirname(__file__), 'versions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(VERSIONS_FOLDER, exist_ok=True)

# In-memory session store (for demo; production would use Redis/DB)
sessions = {}


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logo.png')
def logo():
    logo_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    if os.path.exists(logo_path):
        return send_file(logo_path, mimetype='image/png')
    else:
        # Return a simple placeholder if logo doesn't exist
        return '', 404


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only CSV files are supported'}), 400

    try:
        session_id = str(uuid.uuid4())
        filepath = os.path.join(UPLOAD_FOLDER, f'{session_id}.csv')
        file.save(filepath)

        df = pd.read_csv(filepath)
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400

        sessions[session_id] = {
            'filepath': filepath,
            'filename': file.filename,
            'df': df,
            'uploaded_at': datetime.now().isoformat()
        }

        return jsonify({
            'session_id': session_id,
            'filename': file.filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'preview': df.head(5).fillna('').to_dict('records'),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid or expired session'}), 400

    try:
        df = sessions[session_id]['df']
        diagnostics = DatasetDiagnostics(df)
        results = diagnostics.run_full_diagnostics()
        charts = generate_all_charts(results)
        sessions[session_id]['diagnostics'] = results
        sessions[session_id]['charts'] = charts

        return jsonify({
            'success': True,
            'results': _serialize(results),
            'charts': charts
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/repair', methods=['POST'])
def repair():
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    if 'diagnostics' not in sessions[session_id]:
        return jsonify({'error': 'Run diagnostics first'}), 400

    try:
        df = sessions[session_id]['df']
        diag_results = sessions[session_id]['diagnostics']
        
        # Get column consistency parameters from frontend
        preserve_columns = data.get('preserve_columns', True)
        drop_columns = data.get('drop_columns', False)
        column_specific_processing = data.get('column_specific_processing', {
            'enforce_column_consistency': True,
            'maintain_same_format': True,
            'detect_mixed_data_types': True,
            'standardize_column_formats': True
        })
        
        # Ensure no column dropping
        if drop_columns:
            return jsonify({'error': 'Column dropping is disabled to preserve data integrity'}), 400
        
        repair_engine = DatasetRepair(df, diag_results)
        repaired_df, repair_log, versions = repair_engine.run_auto_repair(
            preserve_columns=preserve_columns,
            column_specific_processing=column_specific_processing
        )
        repair_summary = repair_engine.get_repair_summary()
        sessions[session_id]['repaired_df'] = repaired_df
        sessions[session_id]['repair_log'] = repair_log
        sessions[session_id]['repair_summary'] = repair_summary

        # Save repaired CSV
        repaired_path = os.path.join(VERSIONS_FOLDER, f'{session_id}_repaired.csv')
        repaired_df.to_csv(repaired_path, index=False)
        sessions[session_id]['repaired_path'] = repaired_path

        return jsonify({
            'success': True,
            'repair_summary': repair_summary,
            'repair_log': repair_log,
            'repaired_shape': {'rows': int(repaired_df.shape[0]), 'columns': int(repaired_df.shape[1])},
            'original_shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
            'columns_preserved': True,
            'column_consistency_applied': True
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    data = request.json
    session_id = data.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400

    try:
        evaluator = ModelEvaluator()
        df_original = sessions[session_id]['df']
        before_results = evaluator.train_and_evaluate(df_original, label='Before Repair')

        after_results = None
        comparison = None
        if 'repaired_df' in sessions[session_id]:
            evaluator2 = ModelEvaluator()
            df_repaired = sessions[session_id]['repaired_df']
            after_results = evaluator2.train_and_evaluate(df_repaired, label='After Repair')
            comparison = evaluator.compare_results(before_results, after_results)

        comp_chart = None
        if comparison:
            try:
                comp_chart = plot_model_comparison(comparison)
            except Exception:
                pass

        sessions[session_id]['evaluation'] = {
            'before': before_results,
            'after': after_results,
            'comparison': comparison
        }

        return jsonify({
            'success': True,
            'before': _serialize(before_results),
            'after': _serialize(after_results) if after_results else None,
            'comparison': _serialize(comparison) if comparison else None,
            'comparison_chart': comp_chart
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/download_repaired', methods=['GET'])
def download_repaired():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'error': 'Invalid session'}), 400
    if 'repaired_path' not in sessions[session_id]:
        return jsonify({'error': 'No repaired dataset available'}), 400
    return send_file(sessions[session_id]['repaired_path'],
                     as_attachment=True, download_name='repaired_dataset.csv')


@app.route('/api/session_status', methods=['GET'])
def session_status():
    session_id = request.args.get('session_id')
    if not session_id or session_id not in sessions:
        return jsonify({'exists': False})
    s = sessions[session_id]
    return jsonify({
        'exists': True,
        'filename': s.get('filename'),
        'has_diagnostics': 'diagnostics' in s,
        'has_repair': 'repaired_df' in s,
        'has_evaluation': 'evaluation' in s
    })


def _serialize(obj):
    """Make objects JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Automated Dataset Diagnostics & Repair Framework")
    print("  FYP Project | Running on http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
