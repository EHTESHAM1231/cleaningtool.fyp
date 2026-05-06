"""End-to-end HTTP smoke test against a running server."""
import sys
import time
import json
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:5000"


def _post_json(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post_file(path, filepath):
    import mimetypes
    boundary = "----ADRF" + str(int(time.time()))
    with open(filepath, "rb") as f:
        body = f.read()
    name = filepath.split("/")[-1]
    mime = mimetypes.guess_type(name)[0] or "text/csv"
    data = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode() + body + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def run():
    print("→ /healthz")
    with urllib.request.urlopen(BASE + "/healthz", timeout=10) as r:
        assert r.status == 200, r.status
        print("   ok")

    print("→ /api/upload")
    status, body = _post_file("/api/upload", "sample_dataset.csv")
    assert status == 200, body
    sid = body["session_id"]
    print(f"   session={sid[:8]}  rows={body['rows']} cols={body['columns']}")

    print("→ /api/diagnose")
    status, body = _post_json("/api/diagnose", {"session_id": sid})
    assert status == 200 and body.get("success"), body
    print(f"   health={body['results']['health_score']['score']}  grade={body['results']['health_score']['grade']}")

    print("→ /api/repair")
    status, body = _post_json("/api/repair", {"session_id": sid, "performance_guarded": True})
    assert status == 200 and body.get("success"), body
    print(f"   actions={len(body['repair_log'])}  explanations={len(body['explanations'])}")

    print("→ /api/evaluate")
    status, body = _post_json("/api/evaluate", {"session_id": sid})
    assert status == 200 and body.get("success"), body
    before_acc = body["before"]["metrics"]["accuracy"]
    after_acc = body["after"]["metrics"]["accuracy"]
    print(f"   before_acc={before_acc}  after_acc={after_acc}  delta={after_acc - before_acc:+.4f}")
    sig = body["comparison"].get("statistical_significance")
    if sig:
        print(f"   stat_sig: p={sig['p_value']}  significant={sig['significant_0.05']}")

    print("→ /api/session_status")
    with urllib.request.urlopen(BASE + f"/api/session_status?session_id={sid}", timeout=10) as r:
        body = json.loads(r.read())
        assert body["has_evaluation"], body
        print("   ok")

    print("\n✔ All endpoints responded successfully")
    return 0


if __name__ == "__main__":
    sys.exit(run())
