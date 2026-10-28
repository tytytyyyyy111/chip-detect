from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc
    return YOLO


def predict_images(
    weights: str | Path,
    image_paths: list[str | Path],
    output_dir: str | Path,
    imgsz: int = 640,
    conf: float = 0.25,
    device: str | None = None,
) -> list[dict[str, Any]]:
    weight_path = Path(weights).expanduser()
    if not weight_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weight_path}")
    paths = [Path(path).expanduser() for path in image_paths]
    if not paths:
        raise ValueError("Provide at least one image")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image(s) not found: " + ", ".join(missing))
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    YOLO = load_yolo()
    model = YOLO(str(weight_path))
    results = model.predict(
        source=[str(path) for path in paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    records: list[dict[str, Any]] = []
    for image_path, result in zip(paths, results):
        annotated_name = f"{image_path.stem}_detected.jpg"
        annotated_path = output / annotated_name
        Image.fromarray(result.plot()[..., ::-1].copy()).save(annotated_path, quality=95)

        boxes = getattr(result, "boxes", None)
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            coordinates = boxes.xyxy.cpu().tolist()
            confidences = boxes.conf.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            for coords, confidence, class_id in zip(coordinates, confidences, class_ids):
                class_name = str(model.names.get(class_id, class_id))
                detections.append({
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "confidence": round(float(confidence), 6),
                    "xyxy": [round(float(value), 2) for value in coords],
                })

        records.append({
            "image": str(image_path.resolve()),
            "annotated_image": str(annotated_path.resolve()),
            "image_width": int(result.orig_shape[1]),
            "image_height": int(result.orig_shape[0]),
            "detection_count": len(detections),
            "detections": detections,
        })

    (output / "predictions.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["image", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writeheader()
        for record in records:
            for detection in record["detections"]:
                x1, y1, x2, y2 = detection["xyxy"]
                writer.writerow({
                    "image": record["image"],
                    "class_id": detection["class_id"],
                    "class_name": detection["class_name"],
                    "confidence": detection["confidence"],
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })
    return records
