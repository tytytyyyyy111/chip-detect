from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class ModelNotReadyError(RuntimeError):
    """Raised when no chip-defect model has been configured."""


class ModelRuntimeError(RuntimeError):
    """Raised when the configured model cannot complete inference."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    inference_ms: float
    model_name: str
    device: str


class Detector(Protocol):
    def status(self) -> dict[str, Any]: ...

    def predict(self, image: Image.Image, imgsz: int, conf: float) -> InferenceResult: ...


class YoloDetector:
    """Lazy-loaded inference adapter for Ultralytics and the local paper YOLOv7 model."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str | None = None,
        backend: str | None = None,
    ) -> None:
        configured = weights or os.getenv("CHIP_MODEL_PATH")
        self.weights = Path(configured).expanduser().resolve() if configured else None
        self.device = device if device is not None else os.getenv("CHIP_MODEL_DEVICE", "0")
        self.backend = (backend or os.getenv("CHIP_MODEL_BACKEND", "ultralytics")).strip().lower()
        self._model = None
        self._load_error: str | None = None
        self._lock = threading.Lock()
        self._yolov7_runtime: dict[str, Any] | None = None

    def _configure_local_cache(self) -> None:
        # Ultralytics writes settings on import. Keep all runtime configuration
        # inside the project so inference does not depend on a writable profile.
        project_root = Path(__file__).resolve().parents[2]
        local_config = project_root / "Ultralytics"
        local_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(project_root))
        os.environ.setdefault("MPLCONFIGDIR", str(project_root / "runs" / "matplotlib"))
        os.environ.setdefault("YOLO_OFFLINE", "true")
        os.environ.setdefault("COMET_START_ONLINE", "0")
        os.environ.setdefault("WANDB_MODE", "offline")
        os.environ.setdefault("MLFLOW_TRACKING_URI", str(project_root / "runs" / "mlflow"))

    def status(self) -> dict[str, Any]:
        configured = bool(self.weights and self.weights.is_file())
        return {
            "configured": configured,
            "loaded": self._model is not None,
            "weights": str(self.weights) if configured else None,
            "device": self.device or "auto",
            "backend": self.backend,
            "error": self._load_error,
        }

    def class_names(self) -> list[str]:
        """Return model class names in class-id order after loading the checkpoint."""
        model = self._get_model()
        names = getattr(model, "names", None)
        if names is None and hasattr(model, "model"):
            names = getattr(model.model, "names", None)
        if isinstance(names, dict):
            return [str(names.get(index, names.get(str(index), index))) for index in range(len(names))]
        if isinstance(names, (tuple, list)):
            return [str(name) for name in names]
        raise ModelRuntimeError("模型权重中没有可读取的类别名称，无法进行公平对比。")

    def _get_model(self):
        if not self.weights or not self.weights.is_file():
            raise ModelNotReadyError(
                "芯片缺陷模型尚未配置。请设置 CHIP_MODEL_PATH 指向微调完成的 .pt 权重。"
            )
        if self._model is not None:
            return self._model
        if self._load_error:
            raise ModelRuntimeError(f"模型加载失败：{self._load_error}")

        with self._lock:
            if self._model is not None:
                return self._model
            try:
                self._configure_local_cache()
                if self.backend == "yolov7":
                    self._load_yolov7()
                elif self.backend == "ultralytics":
                    from ultralytics import YOLO

                    self._model = YOLO(str(self.weights))
                else:
                    raise ValueError(
                        f"Unsupported CHIP_MODEL_BACKEND={self.backend!r}; use `yolov7` or `ultralytics`."
                    )
            except Exception as exc:  # runtime may be installed separately from the web API
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise ModelRuntimeError(
                    "无法加载模型；请检查推理依赖、权重格式和 CUDA 环境。"
                ) from exc
        return self._model

    def _load_yolov7(self) -> None:
        import torch

        project_root = Path(__file__).resolve().parents[2]
        yolov7_root = project_root / "yolov7"
        if not (yolov7_root / "models" / "experimental.py").is_file():
            raise FileNotFoundError(f"本地 YOLOv7 推理代码不存在：{yolov7_root}")
        root_text = str(yolov7_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        requested_device = (self.device or "auto").strip().lower()
        if requested_device in {"", "auto"}:
            runtime_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif requested_device == "cpu":
            runtime_device = torch.device("cpu")
        elif requested_device.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError("CHIP_MODEL_DEVICE requests CUDA, but CUDA is unavailable.")
            runtime_device = torch.device(f"cuda:{requested_device}")
        else:
            runtime_device = torch.device(requested_device)

        from models.experimental import attempt_load
        from utils.cp_cluster import cp_cluster_non_max_suppression
        from utils.datasets import letterbox
        from utils.general import scale_coords

        model = attempt_load(str(self.weights), map_location=runtime_device)
        model.to(runtime_device).eval()
        use_half = runtime_device.type == "cuda"
        if use_half:
            model.half()
        self._yolov7_runtime = {
            "torch": torch,
            "device": runtime_device,
            "half": use_half,
            "stride": int(model.stride.max()),
            "letterbox": letterbox,
            "scale_coords": scale_coords,
            "postprocess": cp_cluster_non_max_suppression,
        }
        self._model = model

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, names.get(str(class_id), class_id)))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _predict_yolov7(self, image: Image.Image, imgsz: int, conf: float) -> InferenceResult:
        import numpy as np

        model = self._get_model()
        runtime = self._yolov7_runtime
        if runtime is None:
            raise ModelRuntimeError("YOLOv7 runtime did not initialize.")
        torch = runtime["torch"]
        device = runtime["device"]
        image_array = np.asarray(image.convert("RGB"))
        padded = runtime["letterbox"](
            image_array, new_shape=imgsz, stride=runtime["stride"], auto=False
        )[0]
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(device)
        tensor = tensor.half() if runtime["half"] else tensor.float()
        tensor /= 255.0

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            predictions = model(tensor)[0]
            detections = runtime["postprocess"](
                predictions,
                conf_thres=conf,
                iou_thres=float(os.getenv("CHIP_MODEL_IOU", "0.65")),
                multi_label=True,
            )[0]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000

        output: list[Detection] = []
        if len(detections):
            detections[:, :4] = runtime["scale_coords"](
                tensor.shape[2:], detections[:, :4], image_array.shape
            ).round()
            names = getattr(model, "names", {})
            for row in detections.detach().float().cpu().tolist():
                x1, y1, x2, y2, confidence, class_id = row
                class_id = int(class_id)
                output.append(
                    Detection(
                        class_id=class_id,
                        class_name=self._class_name(names, class_id),
                        confidence=float(confidence),
                        xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return InferenceResult(
            detections=tuple(output),
            inference_ms=elapsed_ms,
            model_name=self.weights.name,
            device=str(device),
        )

    def predict(self, image: Image.Image, imgsz: int, conf: float) -> InferenceResult:
        if self.backend == "yolov7":
            try:
                return self._predict_yolov7(image, imgsz, conf)
            except ModelRuntimeError:
                raise
            except Exception as exc:
                raise ModelRuntimeError(
                    f"YOLOv7 推理失败：{type(exc).__name__}: {exc}"
                ) from exc
        model = self._get_model()
        started = time.perf_counter()
        try:
            # Pass PIL directly so Ultralytics performs its expected RGB/BGR conversion.
            result = model.predict(
                source=image.convert("RGB"),
                imgsz=imgsz,
                conf=conf,
                device=self.device or None,
                verbose=False,
            )[0]
        except Exception as exc:
            raise ModelRuntimeError(f"模型推理失败：{type(exc).__name__}: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            coordinates = boxes.xyxy.detach().cpu().tolist()
            confidences = boxes.conf.detach().cpu().tolist()
            class_ids = [int(value) for value in boxes.cls.detach().cpu().tolist()]
            names = getattr(model, "names", {})
            for coords, confidence, class_id in zip(coordinates, confidences, class_ids):
                class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
                detections.append(
                    Detection(
                        class_id=int(class_id),
                        class_name=str(class_name),
                        confidence=float(confidence),
                        xyxy=tuple(float(value) for value in coords),
                    )
                )

        return InferenceResult(
            detections=tuple(detections),
            inference_ms=elapsed_ms,
            model_name=self.weights.name,
            device=self.device or "auto",
        )
