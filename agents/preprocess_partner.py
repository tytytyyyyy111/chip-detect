from __future__ import annotations

import base64
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI

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

# Share the preprocessing implementation with the ChipInspect system.
CHIP_SRC = Path(
    r"%CHIP_ROOT%\chip-inspection\src"
)
if str(CHIP_SRC) not in sys.path:
    sys.path.insert(0, str(CHIP_SRC))

from chipinspect.preprocess import (  # noqa: E402
    PreprocessError,
    encode_image,
    preprocess_image,
)

PARTNER_AIC = "chip-preprocess-agent"
DEFAULT_TARGET = 640
DEFAULT_TOLERANCE = 12

app = FastAPI(title="Chip Image Preprocess Partner")


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



class RequestSpec:
    def __init__(self) -> None:
        self.raw: bytes | None = None
        self.filename: str = "upload"
        self.auto_crop: bool = True
        self.target_size: int | None = DEFAULT_TARGET
        self.fit_mode: str = "letterbox"
        self.tolerance: int = DEFAULT_TOLERANCE
        self.pad: int = 0
        self.output_format: str = "PNG"


def _parse_spec(command: TaskCommand) -> RequestSpec:
    spec = RequestSpec()

    # Optional JSON options object carried in a TextDataItem.
    for item in command.dataItems or []:
        if isinstance(item, TextDataItem) and item.text:
            text = item.text.strip()
            if text.startswith("{"):
                try:
                    opts = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(opts, dict):
                    if opts.get("auto_crop") is not None:
                        spec.auto_crop = bool(opts["auto_crop"])
                    if opts.get("target_size") is not None:
                        value = int(opts["target_size"])
                        spec.target_size = value if value > 0 else None
                    if opts.get("fit_mode"):
                        spec.fit_mode = str(opts["fit_mode"])
                    if opts.get("tolerance") is not None:
                        spec.tolerance = int(opts["tolerance"])
                    if opts.get("pad") is not None:
                        spec.pad = int(opts["pad"])
                    if opts.get("output_format"):
                        spec.output_format = str(opts["output_format"])
                    if opts.get("filename"):
                        spec.filename = str(opts["filename"])
                    if opts.get("image_path"):
                        p = Path(str(opts["image_path"]))
                        if p.is_file():
                            spec.raw = p.read_bytes()
                            spec.filename = p.name
                    if opts.get("image_base64"):
                        spec.raw = _decode_b64(str(opts["image_base64"]))
                    if spec.raw is not None:
                        return spec

    # Plain payload: FileDataItem(bytes/uri) or TextDataItem(base64/path).
    for item in command.dataItems or []:
        if isinstance(item, FileDataItem):
            try:
                if item.bytes:
                    spec.raw = _decode_b64(item.bytes)
                    spec.filename = item.name or "upload"
                    return spec
                if item.uri:
                    uri = item.uri
                    if uri.startswith("file://"):
                        uri = uri[7:]
                    p = Path(uri)
                    if p.is_file():
                        spec.raw = p.read_bytes()
                        spec.filename = p.name
                        return spec
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(item, TextDataItem) and item.text:
            text = item.text.strip()
            url = _find_url(text)
            if url:
                spec.raw = _fetch_url(url)
                spec.filename = "url-image"
                return spec
            try:
                spec.raw = _decode_b64(text)
                return spec
            except Exception:  # noqa: BLE001
                p = Path(text)
                if p.is_file():
                    spec.raw = p.read_bytes()
                    spec.filename = p.name
                    return spec
    return spec


def _with_sender(task: TaskResult) -> TaskResult:
    task.senderId = LOCAL_AIC
    return task


async def on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    if task:
        return _with_sender(task)

    spec = _parse_spec(command)
    if spec.raw is None:
        pending = TaskManager.create_task(
            command,
            initial_state=TaskState.AwaitingInput,
            data_items=[
                TextDataItem(
                    text="请提供图像：FileDataItem.bytes(base64)、"
                    "TextDataItem(base64 或本地路径)，或 JSON 选项对象。"
                )
            ],
        )
        return _with_sender(pending)

    try:
        result = preprocess_image(
            spec.raw,
            auto_crop=spec.auto_crop,
            target_size=spec.target_size,
            fit_mode=spec.fit_mode,
            tolerance=spec.tolerance,
            pad=spec.pad,
        )
        data, mime = encode_image(result.image, spec.output_format)
    except PreprocessError as exc:
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[TextDataItem(text="预处理失败：%s" % exc)],
        )
        return _with_sender(failed)
    except Exception as exc:  # noqa: BLE001
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[
                TextDataItem(text="预处理异常：%s: %s" % (type(exc).__name__, exc))
            ],
        )
        return _with_sender(failed)

    done = TaskManager.create_task(command, initial_state=TaskState.AwaitingCompletion)
    meta = {
        "filename": spec.filename,
        "source_format": result.source_format,
        "original_size": list(result.original_size),
        "crop_box": list(result.crop_box) if result.crop_box else None,
        "output_size": list(result.output_size),
        "steps": result.steps,
        "auto_crop": spec.auto_crop,
        "target_size": spec.target_size,
        "fit_mode": spec.fit_mode,
        "tolerance": spec.tolerance,
        "mime": mime,
        "bytes": len(data),
    }
    TaskManager.set_products(
        done.taskId,
        [
            Product(
                id="product-%s" % done.taskId,
                name="chip-image-preprocess",
                dataItems=[
                    StructuredDataItem(data=meta),
                    FileDataItem(
                        name="preprocessed.png",
                        mimeType=mime,
                        bytes=base64.b64encode(data).decode("ascii"),
                    ),
                ],
            )
        ],
    )
    result_task = TaskManager.get_task(done.taskId) or done
    return _with_sender(result_task)



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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "aic": LOCAL_AIC}
