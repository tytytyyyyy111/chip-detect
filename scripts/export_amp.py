import csv, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = Path(r"%CHIP_ROOT%")
logs = base / "chip-agent" / "logs"
out = base / "competition"

acc = [json.loads(l) for l in (logs / "amp_access_leader.jsonl").read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
with (out / "AMP访问明细.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "aic", "caller_aic", "callee_aic", "method", "url", "status", "duration_ms", "correlation_id"])
    for r in acc:
        b = r.get("body") or {}
        req = b.get("request") or {}
        res = b.get("response") or {}
        w.writerow([
            r.get("timestamp", ""), r.get("aic", ""),
            (b.get("caller") or {}).get("aic", ""),
            (b.get("callee") or {}).get("aic", ""),
            req.get("method", ""), req.get("url", ""),
            res.get("statusCode", ""), b.get("durationMs", ""),
            r.get("correlation_id", ""),
        ])
print("access_csv", len(acc), flush=True)

msgs = []
for p in sorted(logs.glob("amp_message_*.jsonl")):
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            msgs.append(json.loads(line))
with (out / "AMP消息明细.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "aic", "event_type", "message_id", "correlation_id", "destination"])
    for r in msgs:
        b = r.get("body") or {}
        w.writerow([
            r.get("timestamp", ""), r.get("aic", ""),
            b.get("eventType", ""), b.get("messageId", ""),
            r.get("correlation_id", ""), (b.get("destination") or {}).get("name", ""),
        ])
print("message_csv", len(msgs), flush=True)
