import json
from pathlib import Path

from acps_sdk.acs import AgentCapabilitySpec

DIR = Path(__file__).resolve().parent / "acs"

for path in sorted(DIR.glob("acs_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        spec = AgentCapabilitySpec.model_validate(data)
        dumped = spec.model_dump(by_alias=True)
        print(
            "OK   %-24s name=%s skills=%d endpoints=%d"
            % (
                path.name,
                dumped.get("name"),
                len(dumped.get("skills") or []),
                len(dumped.get("endPoints") or []),
            )
        )
    except Exception as exc:  # noqa: BLE001
        print("FAIL %-24s %s: %s" % (path.name, type(exc).__name__, exc))
