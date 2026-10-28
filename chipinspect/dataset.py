from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class DatasetSummary:
    yaml_path: Path
    root: Path
    class_names: tuple[str, ...]
    splits: dict[str, dict[str, int]]


def _ordered_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        names = tuple(str(item) for item in value)
    elif isinstance(value, dict):
        indexed: dict[int, str] = {}
        for key, name in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Class ID must be an integer, got {key!r}") from exc
            indexed[index] = str(name)
        expected = list(range(len(indexed)))
        if sorted(indexed) != expected:
            raise ValueError(f"Class IDs must be consecutive from 0; got {sorted(indexed)}")
        names = tuple(indexed[index] for index in expected)
    else:
        raise ValueError("Dataset YAML `names` must be a list or integer-keyed mapping")

    if not names or any(not name.strip() for name in names):
        raise ValueError("Dataset YAML must define at least one non-empty class name")
    return names


def load_dataset_config(yaml_path: str | Path) -> tuple[Path, dict[str, Any], tuple[str, ...]]:
    config_path = Path(yaml_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Dataset YAML must contain a mapping")

    root_value = config.get("path", ".")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = (config_path.parent / root).resolve()
    names = _ordered_names(config.get("names"))
    for split in ("train", "val"):
        if split not in config:
            raise ValueError(f"Dataset YAML is missing required split `{split}`")
    return root, config, names


def validate_dataset(yaml_path: str | Path, inspect_labels: bool = True) -> DatasetSummary:
    config_path = Path(yaml_path).expanduser().resolve()
    root, config, names = load_dataset_config(config_path)
    splits: dict[str, dict[str, int]] = {}

    for split in ("train", "val", "test"):
        if split not in config:
            continue
        split_path = Path(config[split]).expanduser()
        if not split_path.is_absolute():
            split_path = (root / split_path).resolve()
        if not split_path.is_dir():
            raise FileNotFoundError(f"`{split}` image directory not found: {split_path}")

        images = sorted(
            item for item in split_path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(f"No supported images found for `{split}` at {split_path}")

        missing_labels = 0
        malformed_labels: list[str] = []
        object_count = 0
        if inspect_labels:
            for image in images:
                relative = image.relative_to(split_path)
                label = (root / "labels" / split / relative).with_suffix(".txt")
                if not label.is_file():
                    missing_labels += 1
                    continue
                for line_number, raw_line in enumerate(
                    label.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    line = raw_line.strip()
                    if not line:
                        continue
                    fields = line.split()
                    try:
                        if len(fields) != 5:
                            raise ValueError("expected class_id x_center y_center width height")
                        class_id = int(fields[0])
                        coords = [float(value) for value in fields[1:]]
                        if class_id < 0 or class_id >= len(names):
                            raise ValueError(f"class_id {class_id} outside [0, {len(names) - 1}]")
                        if any(value < 0 or value > 1 for value in coords):
                            raise ValueError("normalized box coordinates must be between 0 and 1")
                        if coords[2] <= 0 or coords[3] <= 0:
                            raise ValueError("box width and height must be positive")
                        object_count += 1
                    except ValueError as exc:
                        malformed_labels.append(f"{label}:{line_number}: {exc}")
                        if len(malformed_labels) >= 10:
                            break
                if len(malformed_labels) >= 10:
                    break

            if missing_labels:
                raise ValueError(
                    f"{missing_labels}/{len(images)} `{split}` images have no matching label file"
                )
            if malformed_labels:
                raise ValueError("Invalid YOLO labels:\n  - " + "\n  - ".join(malformed_labels))

        splits[split] = {
            "images": len(images),
            "objects": object_count if inspect_labels else -1,
        }

    return DatasetSummary(config_path, root, names, splits)
