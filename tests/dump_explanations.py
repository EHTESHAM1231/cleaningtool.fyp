"""Print the explainability and interpretation output from a live run."""
import json
import urllib.request
import mimetypes
import time
import sys

BASE = "http://127.0.0.1:5000"


def _post_json(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_file(path, filepath):
    boundary = "----DUMP" + str(int(time.time()))
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
        return json.loads(r.read().decode("utf-8"))


up = _post_file("/api/upload", "sample_dataset.csv")
sid = up["session_id"]
_post_json("/api/diagnose", {"session_id": sid})
rep = _post_json("/api/repair", {"session_id": sid, "performance_guarded": True})
ev = _post_json("/api/evaluate", {"session_id": sid})

print("\n" + "=" * 70)
print("  REPAIR EXPLANATIONS (Phase 4 — explainability layer)")
print("=" * 70)
for i, exp in enumerate(rep.get("explanations", []), 1):
    print(f"\n[{i}] step: {exp.get('step')}  (accepted={exp.get('accepted')})")
    print(f"    WHY:    {exp.get('why', '—')}")
    print(f"    WHAT:   {exp.get('what', '—')}")
    print(f"    IMPACT: {exp.get('impact', '—')}")
    if exp.get("cv_before") is not None:
        print(f"    CV:     before={exp['cv_before']:.4f}  after={exp['cv_after']:.4f}  Δ={exp.get('cv_delta', 0):+.4f}")

print("\n" + "=" * 70)
print("  METRIC INTERPRETATION (Phase 5)")
print("=" * 70)
interp = ev.get("comparison", {}).get("interpretation", [])
for i, line in enumerate(interp, 1):
    print(f"\n  {i}. {line}")

print("\n" + "=" * 70)
print("  BEFORE vs AFTER")
print("=" * 70)
print(f"  before: {json.dumps(ev['before']['metrics'], indent=2)}")
print(f"  after:  {json.dumps(ev['after']['metrics'], indent=2)}")
