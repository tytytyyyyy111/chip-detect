from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageOps

from .model_runner import ModelRuntimeError, YoloDetector

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAP_IOUS = tuple(0.50 + 0.05 * index for index in range(10))
DEFAULT_IMAGE_SIZE = 640
DEFAULT_PREDICTION_CONFIDENCE = 0.001
DISPLAY_CONFIDENCE = 0.25
WARMUP_IMAGES = 3


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    weights: Path
    backend: str


@dataclass(frozen=True)
class BenchmarkConfig:
    baseline: ModelSpec
    candidate: ModelSpec
    data_yaml: Path
    device: str
    imgsz: int


@dataclass(frozen=True)
class BenchmarkDataset:
    yaml_path: Path
    names: tuple[str, ...]
    image_paths: tuple[Path, ...]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_benchmark_config() -> BenchmarkConfig:
    root = project_root()
    return BenchmarkConfig(
        baseline=ModelSpec(
            key="baseline",
            name=os.getenv(
                "CHIP_COMPARE_BASELINE_NAME",
                "YOLO11n 基线（SGD+余弦，100轮，640）",
            ),
            weights=Path(
                os.getenv(
                    "CHIP_COMPARE_BASELINE_PATH",
                    str(root / "runs" / "detect" / "front-3000-noamp-sgd-cosine-100e-640" / "weights" / "best.pt"),
                )
            ).expanduser().resolve(),
            backend=os.getenv("CHIP_COMPARE_BASELINE_BACKEND", "ultralytics").strip().lower(),
        ),
        candidate=ModelSpec(
            key="candidate",
            name=os.getenv(
                "CHIP_COMPARE_CANDIDATE_NAME",
                "YOLO11n + RFS 候选（700轮训练，验证集优选，640）",
            ),
            weights=Path(
                os.getenv(
                    "CHIP_COMPARE_CANDIDATE_PATH",
                    str(
                        root
                        / "runs"
                        / "detect"
                        / "front-3000-rfs-noamp-sgd-cosine-700e-20261017"
                        / "weights"
                        / "best.pt"
                    ),
                )
            ).expanduser().resolve(),
            backend=os.getenv("CHIP_COMPARE_CANDIDATE_BACKEND", "ultralytics").strip().lower(),
        ),
        data_yaml=Path(
            os.getenv(
                "CHIP_COMPARE_DATA_YAML",
                str(
                    root
                    / "yolov7"
                    / "runs"
                    / "paper-reproduction"
                    / "configs"
                    / "chip-paper-fourclass-50e-640-paperchecked"
                    / "data.yaml"
                ),
            )
        ).expanduser().resolve(),
        device=os.getenv("CHIP_MODEL_DEVICE", "0").strip() or "auto",
        imgsz=int(os.getenv("CHIP_COMPARE_IMGSZ", str(DEFAULT_IMAGE_SIZE))),
    )


def benchmark_status() -> dict[str, Any]:
    config = default_benchmark_config()
    missing = [
        label
        for label, path in (
            ("基线权重", config.baseline.weights),
            ("训练后权重", config.candidate.weights),
            ("测试集配置", config.data_yaml),
        )
        if not path.is_file()
    ]
    latest_path = project_root() / "runs" / "benchmark" / "latest.json"
    latest = None
    if latest_path.is_file():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            latest = None
    return {
        "configured": not missing,
        "missing": missing,
        "models": [
            {
                "key": config.baseline.key,
                "name": config.baseline.name,
                "weights_file": config.baseline.weights.name,
                "backend": config.baseline.backend,
            },
            {
                "key": config.candidate.key,
                "name": config.candidate.name,
                "weights_file": config.candidate.weights.name,
                "backend": config.candidate.backend,
            },
        ],
        "data_file": config.data_yaml.name,
        "imgsz": config.imgsz,
        "device": config.device,
        "latest": latest,
    }


def _dataset_names(raw_names: Any, nc: int | None) -> tuple[str, ...]:
    if isinstance(raw_names, dict):
        names = tuple(str(raw_names.get(index, raw_names.get(str(index), index))) for index in range(len(raw_names)))
    elif isinstance(raw_names, (list, tuple)):
        names = tuple(str(name) for name in raw_names)
    else:
        names = ()
    if not names and nc:
        names = tuple(str(index) for index in range(int(nc)))
    if not names:
        raise ValueError("测试集 YAML 没有定义 names/nc，无法核对模型类别。")
    return names


def _resolve_split_path(raw_path: str | Path, data_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def _collect_images(raw_split: Any, data_root: Path) -> list[Path]:
    raw_items = raw_split if isinstance(raw_split, (list, tuple)) else [raw_split]
    images: list[Path] = []
    for item in raw_items:
        split_path = _resolve_split_path(str(item), data_root)
        if split_path.is_dir():
            images.extend(
                path for path in split_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif split_path.is_file() and split_path.suffix.lower() == ".txt":
            for line in split_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    image_path = _resolve_split_path(line, data_root)
                    if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                        images.append(image_path)
        else:
            raise FileNotFoundError(f"测试集图片路径不存在或格式不支持：{split_path}")
    unique = sorted(set(images), key=lambda path: str(path).casefold())
    if not unique:
        raise ValueError("测试集没有找到图片。")
    return unique


def load_test_dataset(data_yaml: str | Path) -> BenchmarkDataset:
    yaml_path = Path(data_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"测试集 YAML 不存在：{yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if "test" not in data:
        raise ValueError("测试集 YAML 没有 test split；对比必须使用独立 test 数据。")

    raw_root = Path(str(data.get("path", "."))).expanduser()
    data_root = raw_root.resolve() if raw_root.is_absolute() else (yaml_path.parent / raw_root).resolve()
    names = _dataset_names(data.get("names"), data.get("nc"))
    image_paths = _collect_images(data["test"], data_root)
    return BenchmarkDataset(yaml_path, names, tuple(image_paths))


def _label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    image_dir_indexes = [index for index, part in enumerate(parts[:-1]) if part.casefold() == "images"]
    if not image_dir_indexes:
        raise ValueError(f"无法从图片路径推导 YOLO 标签路径：{image_path}")
    parts[image_dir_indexes[-1]] = "labels"
    parts[-1] = f"{Path(parts[-1]).stem}.txt"
    return Path(*parts)


def _read_ground_truth(image_path: Path, image_size: tuple[int, int], class_count: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    label_path = _label_path_for(image_path)
    if not label_path.is_file():
        return []
    width, height = image_size
    ground_truths: list[tuple[int, tuple[float, float, float, float]]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number}: 需要 YOLO 检测框格式的 5 个字段。")
        class_id = int(float(values[0]))
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number}: 类别 ID 超出测试集 names 范围。")
        cx, cy, box_w, box_h = (float(value) for value in values[1:])
        x1 = (cx - box_w / 2) * width
        y1 = (cy - box_h / 2) * height
        x2 = (cx + box_w / 2) * width
        y2 = (cy + box_h / 2) * height
        ground_truths.append((class_id, (x1, y1, x2, y2)))
    return ground_truths


def _iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    # Match YOLOv7's default AP endpoint so a perfect curve remains AP=1.
    end_recall = float(recall[-1] + 0.01) if recall.size else 0.01
    padded_recall = np.concatenate(([0.0], recall, [end_recall]))
    padded_precision = np.concatenate(([1.0], precision, [0.0]))
    padded_precision = np.flip(np.maximum.accumulate(np.flip(padded_precision)))
    sample_points = np.linspace(0.0, 1.0, 101)
    return float(np.interp(sample_points, padded_recall, padded_precision).mean())


def _model_names(detector: YoloDetector) -> list[str]:
    return detector.class_names()


def _synchronize(device: str) -> None:
    if device.strip().lower() == "cpu":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _match_image_predictions(
    predictions: list[Any],
    ground_truths: list[tuple[int, tuple[float, float, float, float]]],
    class_count: int,
) -> list[tuple[int, float, np.ndarray]]:
    truths_by_class: list[list[tuple[float, float, float, float]]] = [[] for _ in range(class_count)]
    for class_id, box in ground_truths:
        truths_by_class[class_id].append(box)

    claimed: list[set[int]] = [set() for _ in range(class_count)]
    scored: list[tuple[int, float, np.ndarray]] = []
    ordered_predictions = sorted(predictions, key=lambda item: item.confidence, reverse=True)
    for prediction in ordered_predictions:
        class_id = int(prediction.class_id)
        if not 0 <= class_id < class_count:
            continue
        candidate_truths = truths_by_class[class_id]
        best_index = -1
        best_iou = 0.0
        for index, truth_box in enumerate(candidate_truths):
            if index in claimed[class_id]:
                continue
            overlap = _iou(tuple(prediction.xyxy), truth_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        correct = np.zeros(len(MAP_IOUS), dtype=bool)
        if best_index >= 0 and best_iou >= MAP_IOUS[0]:
            claimed[class_id].add(best_index)
            correct = np.asarray([best_iou >= threshold for threshold in MAP_IOUS], dtype=bool)
        scored.append((class_id, float(prediction.confidence), correct))
    return scored


def _summarize_metrics(
    class_names: tuple[str, ...],
    class_scores: list[list[float]],
    class_correct: list[list[np.ndarray]],
    target_counts: list[int],
    image_count: int,
    latency_ms: list[float],
) -> dict[str, Any]:
    per_class: list[dict[str, Any]] = []
    ap_values: list[list[float]] = []
    total_tp_at_display_conf = 0
    total_predictions_at_display_conf = 0
    total_targets = sum(target_counts)

    for class_id, class_name in enumerate(class_names):
        scores = np.asarray(class_scores[class_id], dtype=np.float64)
        correct = (
            np.asarray(class_correct[class_id], dtype=bool)
            if class_correct[class_id]
            else np.zeros((0, len(MAP_IOUS)), dtype=bool)
        )
        if scores.size:
            order = np.argsort(-scores, kind="stable")
            scores = scores[order]
            correct = correct[order]
        num_targets = target_counts[class_id]

        aps: list[float] = []
        if num_targets:
            if scores.size:
                for iou_index in range(len(MAP_IOUS)):
                    tp = correct[:, iou_index].astype(np.float64)
                    fp = 1.0 - tp
                    cumulative_tp = np.cumsum(tp)
                    cumulative_fp = np.cumsum(fp)
                    recall = cumulative_tp / max(num_targets, 1)
                    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
                    aps.append(_average_precision(recall, precision))
            else:
                aps = [0.0] * len(MAP_IOUS)
            ap_values.append(aps)

        selected = scores >= DISPLAY_CONFIDENCE
        selected_correct = correct[selected, 0] if scores.size else np.zeros(0, dtype=bool)
        tp_at_threshold = int(selected_correct.sum())
        predictions_at_threshold = int(selected.sum())
        total_tp_at_display_conf += tp_at_threshold
        total_predictions_at_display_conf += predictions_at_threshold
        per_class.append(
            {
                "name": class_name,
                "targets": num_targets,
                "predictions_at_conf": predictions_at_threshold,
                "precision_at_conf": (
                    tp_at_threshold / predictions_at_threshold if predictions_at_threshold else 0.0
                ),
                "recall_at_conf": tp_at_threshold / num_targets if num_targets else None,
                "ap50": aps[0] if aps else None,
                "ap50_95": float(np.mean(aps)) if aps else None,
            }
        )

    if not ap_values:
        raise ValueError("测试集没有有效标注对象，无法计算 mAP。")
    latency_values = np.asarray(latency_ms, dtype=np.float64)
    mean_latency = float(latency_values.mean()) if latency_values.size else 0.0
    return {
        "images": image_count,
        "targets": total_targets,
        "precision_at_conf": (
            total_tp_at_display_conf / total_predictions_at_display_conf
            if total_predictions_at_display_conf
            else 0.0
        ),
        "recall_at_conf": total_tp_at_display_conf / total_targets if total_targets else 0.0,
        "map50": float(np.mean([values[0] for values in ap_values])),
        "map50_95": float(np.mean(ap_values)),
        "mean_latency_ms": mean_latency,
        "median_latency_ms": float(np.median(latency_values)) if latency_values.size else 0.0,
        "p95_latency_ms": float(np.percentile(latency_values, 95)) if latency_values.size else 0.0,
        "fps": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        "per_class": per_class,
    }


def _evaluate_model(
    spec: ModelSpec,
    dataset: BenchmarkDataset,
    imgsz: int,
    device: str,
) -> dict[str, Any]:
    if spec.backend not in {"ultralytics", "yolov7"}:
        raise ValueError(f"{spec.name} 的 backend 必须是 ultralytics 或 yolov7。")
    if not spec.weights.is_file():
        raise FileNotFoundError(f"{spec.name} 权重不存在：{spec.weights}")

    detector = YoloDetector(weights=spec.weights, device=device, backend=spec.backend)
    names = _model_names(detector)
    if names != list(dataset.names):
        raise ValueError(
            f"{spec.name} 类别顺序与测试集不一致：模型={names}，测试集={list(dataset.names)}。"
        )

    first_image = dataset.image_paths[0]
    with Image.open(first_image) as source:
        warmup_image = ImageOps.exif_transpose(source).convert("RGB")
    for _ in range(min(WARMUP_IMAGES, len(dataset.image_paths))):
        detector.predict(warmup_image, imgsz=imgsz, conf=DEFAULT_PREDICTION_CONFIDENCE)
    _synchronize(device)

    class_scores: list[list[float]] = [[] for _ in dataset.names]
    class_correct: list[list[np.ndarray]] = [[] for _ in dataset.names]
    target_counts = [0 for _ in dataset.names]
    latencies: list[float] = []

    try:
        for image_path in dataset.image_paths:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            image_truths = _read_ground_truth(image_path, image.size, len(dataset.names))
            for class_id, _ in image_truths:
                target_counts[class_id] += 1

            _synchronize(device)
            started = time.perf_counter()
            result = detector.predict(
                image,
                imgsz=imgsz,
                conf=DEFAULT_PREDICTION_CONFIDENCE,
            )
            _synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)

            for class_id, score, correct in _match_image_predictions(
                list(result.detections), image_truths, len(dataset.names)
            ):
                class_scores[class_id].append(score)
                class_correct[class_id].append(correct)

        metrics = _summarize_metrics(
            dataset.names,
            class_scores,
            class_correct,
            target_counts,
            len(dataset.image_paths),
            latencies,
        )
        return {
            "key": spec.key,
            "name": spec.name,
            "weights_file": spec.weights.name,
            "backend": spec.backend,
            "metrics": metrics,
        }
    finally:
        del warmup_image
        del detector
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_benchmark() -> dict[str, Any]:
    config = default_benchmark_config()
    if not 320 <= config.imgsz <= 2048 or config.imgsz % 32:
        raise ValueError("对比输入尺寸必须在 320–2048 之间，且为 32 的倍数。")
    dataset = load_test_dataset(config.data_yaml)
    baseline = _evaluate_model(config.baseline, dataset, config.imgsz, config.device)
    candidate = _evaluate_model(config.candidate, dataset, config.imgsz, config.device)

    baseline_latency = baseline["metrics"]["mean_latency_ms"]
    candidate_latency = candidate["metrics"]["mean_latency_ms"]
    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "name": config.data_yaml.stem,
            "split": "test",
            "images": len(dataset.image_paths),
            "classes": list(dataset.names),
            "targets": baseline["metrics"]["targets"],
        },
        "conditions": {
            "imgsz": config.imgsz,
            "device": config.device,
            "device_name": _device_name(config.device),
            "batch_size": 1,
            "confidence_for_map": DEFAULT_PREDICTION_CONFIDENCE,
            "confidence_for_precision_recall": DISPLAY_CONFIDENCE,
            "latency_includes": "image preprocessing, inference and postprocessing; image file read excluded",
        },
        "models": [baseline, candidate],
        "delta": {
            "map50_points": (
                candidate["metrics"]["map50"] - baseline["metrics"]["map50"]
            ) * 100.0,
            "map50_95_points": (
                candidate["metrics"]["map50_95"] - baseline["metrics"]["map50_95"]
            ) * 100.0,
            "precision_points": (
                candidate["metrics"]["precision_at_conf"] - baseline["metrics"]["precision_at_conf"]
            ) * 100.0,
            "recall_points": (
                candidate["metrics"]["recall_at_conf"] - baseline["metrics"]["recall_at_conf"]
            ) * 100.0,
            "latency_percent": (
                (candidate_latency / baseline_latency - 1.0) * 100.0
                if baseline_latency > 0
                else None
            ),
            "fps_percent": (
                (candidate["metrics"]["fps"] / baseline["metrics"]["fps"] - 1.0) * 100.0
                if baseline["metrics"]["fps"] > 0
                else None
            ),
        },
        "caveat": (
            "本次为同测试集上的系统级模型对比。模型架构或训练设置不完全相同时，"
            "结果不能单独证明某一个训练改进造成了性能变化。"
        ),
    }
    output_path = project_root() / "runs" / "benchmark" / "latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


def _device_name(device: str) -> str:
    if device.strip().lower() == "cpu":
        return "CPU"
    try:
        import torch

        if torch.cuda.is_available():
            index = int(device) if device.isdigit() else torch.cuda.current_device()
            return str(torch.cuda.get_device_name(index))
    except (ImportError, RuntimeError, ValueError):
        pass
    return device
