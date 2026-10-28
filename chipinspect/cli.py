from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import validate_dataset
from .engine import load_yolo, predict_images


def _add_common_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--imgsz", type=int, default=640, help="Inference/training image size")
    parser.add_argument("--device", default=None, help="Device, e.g. 0, cpu, or mps")


def main() -> None:
    parser = argparse.ArgumentParser(prog="chipinspect", description="Chip-package defect YOLO baseline")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check-data", help="Validate dataset YAML, images and YOLO box labels")
    check.add_argument("--data", required=True)
    check.add_argument("--skip-label-check", action="store_true")

    train = commands.add_parser("train", help="Fine-tune a pretrained YOLO detection model")
    train.add_argument("--data", required=True)
    train.add_argument(
        "--model",
        default="runs/pretrained/yolo11n.pt",
        help="Pretrained checkpoint or model YAML",
    )
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--project", default="runs/detect")
    train.add_argument("--name", default="chip-yolo-baseline")
    _add_common_runtime_options(train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a trained checkpoint")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--batch", type=int, default=8)
    _add_common_runtime_options(evaluate)

    predict = commands.add_parser("predict", help="Detect defects and export annotated images plus JSON/CSV")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--source", required=True, nargs="+", help="One or more image paths")
    predict.add_argument("--output", default="runs/predict/chip-inspection")
    predict.add_argument("--conf", type=float, default=0.25)
    _add_common_runtime_options(predict)

    args = parser.parse_args()
    if args.command == "check-data":
        summary = validate_dataset(args.data, inspect_labels=not args.skip_label_check)
        print(json.dumps({
            "dataset_yaml": str(summary.yaml_path),
            "dataset_root": str(summary.root),
            "classes": list(summary.class_names),
            "splits": summary.splits,
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "train":
        summary = validate_dataset(args.data)
        print(f"Classes: {', '.join(summary.class_names)}")
        YOLO = load_yolo()
        model = YOLO(args.model)
        result = model.train(
            data=str(summary.yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            pretrained=True,
            seed=42,
            deterministic=True,
            plots=True,
        )
        best = Path(result.save_dir) / "weights" / "best.pt"
        print(f"Training finished. Best checkpoint: {best.resolve()}")
        return

    if args.command == "evaluate":
        summary = validate_dataset(args.data)
        YOLO = load_yolo()
        model = YOLO(args.weights)
        metrics = model.val(
            data=str(summary.yaml_path),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=True,
        )
        report = {
            "weights": str(Path(args.weights).resolve()),
            "split": args.split,
            "metrics": getattr(metrics, "results_dict", {}),
            "per_class_map50_95": getattr(metrics, "maps", []).tolist()
            if hasattr(getattr(metrics, "maps", None), "tolist") else [],
            "speed_ms_per_image": getattr(metrics, "speed", {}),
        }
        destination = Path(getattr(metrics, "save_dir", "runs/val")) / "evaluation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"Evaluation report: {destination.resolve()}")
        print(json.dumps(report, ensure_ascii=False, indent=2, default=float))
        return

    records = predict_images(
        weights=args.weights,
        image_paths=args.source,
        output_dir=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Results saved to: {Path(args.output).resolve()}")
