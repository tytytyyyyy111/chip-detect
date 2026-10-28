"""Retrain the chip-defect model from accumulated feedback samples.

Pipeline:
  manifest -> YOLO dataset (train/val) -> fine-tune -> evaluate -> promote if better

Usage:
  python retrain.py --dry-run --force      # build dataset only
  python retrain.py --force --epochs 1     # smoke test
  python retrain.py                        # normal run (needs >= min-samples)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(r"%CHIP_ROOT%\chip-samples\feedback")
MANIFEST = ROOT / "manifest.jsonl"
DATASET = ROOT / "dataset"
RUNS = ROOT / "runs"
ARCHIVE = ROOT / "weights_archive"
METRICS = ROOT / "metrics.json"
REPORT = ROOT / "retrain_report.json"

CURRENT_WEIGHTS = Path(
    r"%CHIP_ROOT%\chip-inspection\runs\detect"
    r"\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"
)
CLASS_NAMES = {0: "ZangWu", 1: "PoSun", 2: "BengLie", 3: "DuMoTuoLuo"}


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def build_dataset(rows: list[dict]) -> dict:
    if DATASET.exists():
        shutil.rmtree(DATASET)
    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    usable = []
    for row in rows:
        image = Path(row.get("image", ""))
        label = Path(row.get("label", ""))
        if image.is_file() and label.is_file():
            usable.append(row)
    usable.sort(key=lambda r: r.get("sha", ""))

    if not usable:
        raise SystemExit("no usable feedback samples")

    val_count = max(1, round(len(usable) * 0.15)) if len(usable) >= 2 else 0
    for index, row in enumerate(usable):
        split = "val" if index < val_count else "train"
        stem = row.get("sha") or "sample"
        shutil.copy2(row["image"], DATASET / "images" / split / ("%s.jpg" % stem))
        shutil.copy2(row["label"], DATASET / "labels" / split / ("%s.txt" % stem))

    yaml_path = DATASET / "data.yaml"
    names = "\n".join("  %d: %s" % (k, v) for k, v in CLASS_NAMES.items())
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\nnames:\n%s\n"
        % (DATASET.as_posix(), names),
        encoding="utf-8",
    )
    return {
        "total": len(usable),
        "train": len(usable) - val_count,
        "val": val_count,
        "yaml": str(yaml_path),
    }


def load_metrics() -> dict:
    if METRICS.exists():
        try:
            return json.loads(METRICS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    rows = load_manifest()
    if len(rows) < args.min_samples and not args.force:
        raise SystemExit(
            "not enough samples: %d < %d (use --force to override)"
            % (len(rows), args.min_samples)
        )

    info = build_dataset(rows)
    print("dataset:", json.dumps(info, ensure_ascii=False))

    if args.dry_run:
        print("dry-run: dataset built, training skipped")
        return

    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = "retrain-%s" % timestamp
    model = YOLO(str(CURRENT_WEIGHTS))
    model.train(
        data=info["yaml"],
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(RUNS),
        name=run_name,
        exist_ok=True,
        verbose=False,
        amp=False,
        plots=False,
        workers=0,
    )

    best = RUNS / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit("training finished but best.pt not found")

    metrics = model.val(data=info["yaml"], split="val", device=args.device, verbose=False)
    map50_95 = float(getattr(metrics.box, "map", 0.0))
    map50 = float(getattr(metrics.box, "map50", 0.0))

    baseline = load_metrics()
    baseline_map = float(baseline.get("best_map50_95", 0.0))
    improved = map50_95 > baseline_map

    report = {
        "timestamp": timestamp,
        "samples": info["total"],
        "epochs": args.epochs,
        "map50_95": round(map50_95, 6),
        "map50": round(map50, 6),
        "baseline_map50_95": baseline_map,
        "improved": improved,
        "promoted": False,
        "candidate": str(best),
        "current": str(CURRENT_WEIGHTS),
    }

    if improved and not args.no_promote:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        backup = ARCHIVE / ("best_%s.pt" % timestamp)
        if CURRENT_WEIGHTS.is_file():
            shutil.copy2(CURRENT_WEIGHTS, backup)
        shutil.copy2(best, CURRENT_WEIGHTS)
        METRICS.write_text(
            json.dumps(
                {
                    "best_map50_95": map50_95,
                    "best_map50": map50,
                    "model": str(CURRENT_WEIGHTS),
                    "updated_at": timestamp,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        report["promoted"] = True
        report["backup"] = str(backup)
    elif args.no_promote:
        report["promote_skipped"] = True

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
