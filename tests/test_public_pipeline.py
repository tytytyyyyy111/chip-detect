from pathlib import Path

import requests

tok = Path(r"%CHIP_ROOT%\frp\config\url_token.txt").read_text(encoding="utf-8").strip()
base = "http://YOUR_PUBLIC_HOST:8080"
sample = Path(r"%CHIP_ROOT%\chip-samples\real\sample_1.jpg")

s = requests.Session()
r = s.get(base + "/", params={"key": tok}, timeout=30)
print("auth", r.status_code)

with sample.open("rb") as fh:
    response = s.post(
        base + "/api/v1/pipeline",
        files={"image": (sample.name, fh, "image/jpeg")},
        data={
            "auto_crop": "true",
            "target_size": "640",
            "imgsz": "640",
            "conf": "0.25",
            "include_report": "true",
        },
        timeout=300,
    )

print("status", response.status_code)
data = response.json()
print("detection_count", data.get("detection_count"))
report = data.get("report") or {}
print("verdict", report.get("verdict"), report.get("verdict_label"))
print("defect_total", report.get("defect_total"))
print("has_annotated", bool(data.get("annotated_image_base64")))
print("model", (data.get("detection") or {}).get("model") or data.get("model"))