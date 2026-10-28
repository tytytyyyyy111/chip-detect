import json
import subprocess
from pathlib import Path

CLI = str(Path(".venv/Scripts/acps-cli.exe").resolve())
r = subprocess.run(
    [CLI, "--config", "acps-cli.wutong.toml", "agent", "list", "--json"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
if r.returncode != 0:
    print("ERROR", r.returncode, r.stderr[:500])
    raise SystemExit(1)
data = json.loads(r.stdout)
print("total:", data.get("total"))
for a in data.get("items", []):
    acs = a.get("acs") or {}
    eps = acs.get("endPoints") or []
    endpoint = eps[0].get("url") if eps else "-"
    print("%-22s | %-9s | aic=%-52s | %s" % (
        a.get("name"), a.get("approval_status"), a.get("aic"), endpoint
    ))
