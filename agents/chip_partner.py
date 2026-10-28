from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image

from acps_sdk.aip import (
    FileDataItem,
    Product,
    StructuredDataItem,
    TaskCommand,
    TaskResult,
    TaskState,
    TextDataItem,
)
from acps_sdk.aip.aip_rpc_server import (
    CommandHandlers,
    DefaultHandlers,
    TaskManager,
    add_aip_rpc_router,
)
from acps_sdk.aip.aip_peer_cert import AipPeerCertificateMiddleware

PARTNER_AIC = "chip-detection-agent"

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(
    r"%CHIP_ROOT%\chip-inspection\runs\detect"
    r"\front-3000-rfs-noamp-sgd-cosine-700e-20261017\weights\best.pt"
)
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE))
os.environ.setdefault("YOLO_OFFLINE", "true")
os.environ.setdefault("COMET_START_ONLINE", "0")
os.environ.setdefault("WANDB_MODE", "offline")

MODEL_ROOTS = [
    Path(r"%CHIP_ROOT%\chip-inspection\models\selectable"),
]
ADMIN_TOKEN_FILE = Path(r"%CHIP_ROOT%\frp\config\admin_token.txt")
ADMIN_TOKEN = (
    ADMIN_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if ADMIN_TOKEN_FILE.exists()
    else ""
)


def _require_admin(request: Request) -> None:
    if ADMIN_TOKEN and request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")


def _is_allowed_model(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.suffix.lower() != ".pt" or not resolved.is_file():
        return False
    current = Path(os.getenv("CHIP_MODEL_PATH", str(DEFAULT_MODEL))).resolve()
    if resolved == current:
        return True
    for root in MODEL_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False

app = FastAPI(title="Chip Defect Detection Partner")


class Detector:
    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self._override: Path | None = None

    @property
    def weights(self) -> Path:
        if self._override is not None:
            return self._override
        return Path(os.getenv("CHIP_MODEL_PATH", str(DEFAULT_MODEL)))

    def load_weights(self, path: Path):
        import gc

        from ultralytics import YOLO

        resolved = Path(path).resolve()
        with self._lock:
            if self._model is not None:
                try:
                    if self.weights.resolve() == resolved:
                        return self._model
                except OSError:
                    pass
        model = YOLO(str(resolved))
        with self._lock:
            previous = self._model
            self._override = resolved
            self._model = model
            self._load_error = None
        del previous
        gc.collect()
        return model

    def model(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise RuntimeError(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from ultralytics import YOLO

                self._model = YOLO(str(self.weights))
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise
        return self._model

    def predict(self, image: Image.Image, imgsz: int, conf: float) -> dict:
        model = self.model()
        device = os.getenv("CHIP_MODEL_DEVICE", "0")
        started = time.perf_counter()
        try:
            result = model.predict(
                source=image.convert("RGB"),
                imgsz=imgsz,
                conf=conf,
                device=device,
                verbose=False,
            )[0]
        except Exception:
            # CUDA fallback to CPU
            result = model.predict(
                source=image.convert("RGB"),
                imgsz=imgsz,
                conf=conf,
                device="cpu",
                verbose=False,
            )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000

        names = getattr(model, "names", {}) or {}
        detections = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confs = boxes.conf.detach().cpu().tolist()
            clss = [int(v) for v in boxes.cls.detach().cpu().tolist()]
            for coords, score, cid in zip(xyxy, confs, clss):
                if isinstance(names, dict):
                    cname = names.get(cid, names.get(str(cid), str(cid)))
                elif 0 <= cid < len(names):
                    cname = names[cid]
                else:
                    cname = str(cid)
                detections.append(
                    {
                        "class_id": int(cid),
                        "class_name": str(cname),
                        "confidence": round(float(score), 4),
                        "xyxy": [round(float(v), 2) for v in coords],
                    }
                )

        return {
            "model": self.weights.name,
            "device": str(device),
            "image_size": [image.width, image.height],
            "inference_ms": round(elapsed_ms, 1),
            "count": len(detections),
            "detections": detections,
        }


detector = Detector()


def _decode_b64(text: str) -> bytes:
    payload = text.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload, validate=False)


def _fetch_url(url: str, max_bytes: int = 20 * 1024 * 1024) -> bytes:
    """从 http(s) 链接取图像字节。"""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "chip-agent/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read(max_bytes + 1)
    if not data:
        raise ValueError("empty image body")
    if len(data) > max_bytes:
        raise ValueError("image too large")
    return data


def _find_url(text: str) -> str | None:
    import re

    text = text.strip()
    if text.startswith(("http://", "https://")):
        return text
    m = re.search(r"https?://[^\s\"'<>）)，。]+", text)
    return m.group(0).rstrip(".,;") if m else None


def _extract_image(command: TaskCommand) -> Image.Image:
    """Accept an image from FileDataItem(bytes/uri) or TextDataItem(base64/path)."""
    for item in command.dataItems or []:
        if isinstance(item, FileDataItem):
            try:
                if item.bytes:
                    return Image.open(io.BytesIO(_decode_b64(item.bytes)))
                if item.uri:
                    uri = item.uri
                    if uri.startswith("file://"):
                        uri = uri[7:]
                    if Path(uri).is_file():
                        return Image.open(uri)
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(item, TextDataItem) and item.text:
            raw = item.text.strip()
            url = _find_url(raw)
            if url:
                return Image.open(io.BytesIO(_fetch_url(url)))
            try:
                return Image.open(io.BytesIO(_decode_b64(raw)))
            except Exception:  # noqa: BLE001
                if Path(raw).is_file():
                    return Image.open(raw)
    raise ValueError("no image found in task command")


async def on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    if task:
        return task

    try:
        image = _extract_image(command)
    except Exception:  # noqa: BLE001
        pending = TaskManager.create_task(
            command,
            initial_state=TaskState.AwaitingInput,
            data_items=[
                TextDataItem(
                    text="请提供芯片封装图像：FileDataItem.bytes(base64)、"
                    "TextDataItem(base64 或本地绝对路径)。"
                )
            ],
        )
        pending.senderId = LOCAL_AIC
        return pending

    try:
        payload = detector.predict(image, imgsz=640, conf=0.25)
    except Exception as exc:  # noqa: BLE001
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[TextDataItem(text=f"推理失败：{type(exc).__name__}: {exc}")],
        )
        failed.senderId = LOCAL_AIC
        return failed

    done = TaskManager.create_task(command, initial_state=TaskState.AwaitingCompletion)
    TaskManager.set_products(
        done.taskId,
        [
            Product(
                id=f"product-{done.taskId}",
                name="chip-defect-detection",
                dataItems=[
                    StructuredDataItem(data=payload),
                    TextDataItem(
                        text=json.dumps(payload, ensure_ascii=False)
                        if "json" in globals()
                        else str(payload)
                    ),
                ],
            )
        ],
    )
    result = TaskManager.get_task(done.taskId) or done
    result.senderId = LOCAL_AIC
    return result



_amp_emitter = None


async def _amp_receive(command, task):
    global _amp_emitter
    try:
        from amp_message import make_emitter, message_body
        import os as _os
        aic = _os.getenv("CHIP_AGENT_AIC", PARTNER_AIC)
        if _amp_emitter is None:
            _amp_emitter = make_emitter(_os.path.basename(__file__).split("_")[0], aic)
        tid = getattr(command, "taskId", "") or ""
        await _amp_emitter.emit(message_body("receive", tid, aic), correlation_id=tid)
    except Exception:
        pass


_original_on_start = on_start


async def on_start(command, task):
    result = await _original_on_start(command, task)
    await _amp_receive(command, task)
    return result

handlers = CommandHandlers(
    on_start=on_start,
    on_get=DefaultHandlers.get,
    on_cancel=DefaultHandlers.cancel,
    on_complete=DefaultHandlers.complete,
    on_continue=DefaultHandlers.continue_,
)

LOCAL_AIC = os.getenv("CHIP_AGENT_AIC", PARTNER_AIC)
IDENTITY_BINDING = os.getenv("CHIP_IDENTITY_BINDING", "0") == "1"
if IDENTITY_BINDING:
    app.add_middleware(AipPeerCertificateMiddleware)
add_aip_rpc_router(
    app,
    "/rpc",
    handlers,
    local_aic=LOCAL_AIC,
    identity_binding_enabled=IDENTITY_BINDING,
)


@app.on_event("startup")
async def _warmup() -> None:
    # Load YOLO weights in the background so the first AIP request is fast.
    import threading

    threading.Thread(target=lambda: _safe_load(), daemon=True).start()


def _safe_load() -> None:
    try:
        detector.model()
    except Exception:  # noqa: BLE001
        pass


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "aic": LOCAL_AIC,
        "model": str(detector.weights),
        "model_loaded": detector._model is not None,  # noqa: SLF001
    }


@app.get("/models")
async def list_models(request: Request):
    _require_admin(request)
    items = []
    for root in MODEL_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.pt"):
            if not _is_allowed_model(path):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            items.append(
                {
                    "name": path.relative_to(root.parent).as_posix(),
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return {"models": items, "current": str(detector.weights)}


@app.post("/model/select")
async def select_model(request: Request):
    _require_admin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")
    path_value = str(payload.get("path", "")).strip()
    if not path_value:
        raise HTTPException(status_code=400, detail="path required")
    path = Path(path_value)
    if not _is_allowed_model(path):
        raise HTTPException(status_code=400, detail="model path not allowed")
    try:
        detector.load_weights(path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail="load failed: %s: %s" % (type(exc).__name__, exc),
        )
    return {"status": "ok", "weights": str(path)}
