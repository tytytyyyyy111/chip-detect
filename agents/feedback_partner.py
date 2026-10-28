from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from PIL import Image

from acps_sdk.aip import (
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

PARTNER_AIC = "chip-feedback-agent"

ROOT = Path(
    r"%CHIP_ROOT%\chip-samples\feedback"
)
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"
MANIFEST = ROOT / "manifest.jsonl"
TRIGGER = ROOT / "retrain_request.json"
RETRAIN_THRESHOLD = int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "50"))

CLASS_ORDER = ["ZangWu", "PoSun", "BengLie", "DuMoTuoLuo"]

app = FastAPI(title="Chip Feedback Partner")


def _ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)


_UNWRAP_KEYS = ("input", "payload", "data", "params", "arguments", "argumentsJson")
_TEXT_ACTIONS = (
    ("clear_trigger", ("clear_trigger", "clear", "清除")),
    ("trigger", ("trigger", "触发", "重训")),
    ("submit", ("submit", "提交样本", "提交回流", "上传样本")),
    ("stats", ("stats", "统计", "状态", "回流")),
)


def _unwrap(value: Any) -> dict[str, Any] | None:
    """接受多层包裹的入参：{"input": {...}} / {"payload": "{...}"} 等。"""
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("{"):
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    for key in _UNWRAP_KEYS:
        if key in value:
            inner = _unwrap(value[key])
            if inner is not None:
                merged = {k: v for k, v in value.items() if k not in _UNWRAP_KEYS}
                merged.update(inner)
                return merged
    return value


def _extract_payload(command: TaskCommand) -> dict[str, Any] | None:
    texts = []
    for item in command.dataItems or []:
        if isinstance(item, StructuredDataItem):
            got = _unwrap(item.data)
            if got is not None:
                return got
        if isinstance(item, TextDataItem) and item.text:
            texts.append(item.text.strip())
    for text in texts:
        got = _unwrap(text)
        if got is not None:
            return got
    joined = " ".join(texts).lower()
    for action, words in _TEXT_ACTIONS:
        if any(w.lower() in joined for w in words):
            return {"action": action}
    return None


def _read_manifest() -> list[dict]:
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


def _yolo_labels(detections: list[dict], width: float, height: float) -> str:
    lines = []
    for det in detections or []:
        xyxy = det.get("xyxy") or []
        if len(xyxy) != 4 or width <= 0 or height <= 0:
            continue
        class_id = int(det.get("class_id", 0))
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        x1, x2 = sorted((max(0.0, x1), min(width, x2)))
        y1, y2 = sorted((max(0.0, y1), min(height, y2)))
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        nw = w / width
        nh = h / height
        lines.append("%d %.6f %.6f %.6f %.6f" % (class_id, cx, cy, nw, nh))
    return "\n".join(lines)


def _stats() -> dict[str, Any]:
    rows = _read_manifest()
    per_class: dict[str, int] = {name: 0 for name in CLASS_ORDER}
    defects = 0
    clean = 0
    for row in rows:
        counts = row.get("class_counts") or {}
        for name, value in counts.items():
            per_class[name] = per_class.get(name, 0) + int(value)
            defects += int(value)
        if not counts:
            clean += 1
    return {
        "samples": len(rows),
        "defects": defects,
        "clean_samples": clean,
        "per_class": per_class,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "retrain_ready": len(rows) >= RETRAIN_THRESHOLD,
        "retrain_requested": TRIGGER.exists(),
    }


def _submit(payload: dict[str, Any]) -> dict[str, Any]:
    image_b64 = payload.get("image_base64")
    if not image_b64:
        raise ValueError("image_base64 is required")
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    with Image.open(io.BytesIO(raw)) as probe:
        width, height = probe.size
        probe.verify()

    sha = hashlib.sha256(raw).hexdigest()[:16]
    _ensure_dirs()
    image_path = IMAGES_DIR / ("%s.jpg" % sha)
    if not image_path.exists():
        with Image.open(io.BytesIO(raw)) as img:
            img.convert("RGB").save(image_path, "JPEG", quality=92)

    human = payload.get("human_correction") or []
    label_text = _yolo_labels(human, float(width), float(height))
    label_path = LABELS_DIR / ("%s.txt" % sha)
    label_path.write_text(label_text, encoding="utf-8")

    class_counts: dict[str, int] = {}
    for det in human:
        name = str(det.get("class_name", "unknown"))
        class_counts[name] = class_counts.get(name, 0) + 1

    record = {
        "id": "%s-%s" % (sha, uuid.uuid4().hex[:6]),
        "sha": sha,
        "filename": payload.get("filename") or image_path.name,
        "image": str(image_path),
        "label": str(label_path),
        "width": width,
        "height": height,
        "reviewer": payload.get("reviewer") or "",
        "note": payload.get("note") or "",
        "model_prediction_count": len(payload.get("model_prediction") or []),
        "human_count": len(human),
        "class_counts": class_counts,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = _stats()
    if stats["retrain_ready"] and not TRIGGER.exists():
        TRIGGER.write_text(
            json.dumps(
                {
                    "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "samples": stats["samples"],
                    "threshold": RETRAIN_THRESHOLD,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        stats = _stats()

    return {"saved": record, "stats": stats}


def _action(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "submit").lower()
    if action == "stats":
        return {"action": "stats", "stats": _stats()}
    if action == "trigger":
        _ensure_dirs()
        stats = _stats()
        TRIGGER.write_text(
            json.dumps(
                {
                    "requested_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "samples": stats["samples"],
                    "threshold": RETRAIN_THRESHOLD,
                    "manual": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"action": "trigger", "stats": _stats()}
    if action == "clear_trigger":
        if TRIGGER.exists():
            TRIGGER.unlink()
        return {"action": "clear_trigger", "stats": _stats()}
    if action == "submit":
        if not (payload.get("image_base64") or payload.get("imageBase64")):
            return {
                "action": "submit",
                "ok": False,
                "reason": "提交回流需要提供 image_base64（以及 labels 或 detections）；"
                          "当前未收到图像，已返回统计供参考。",
                "usage": {
                    "action": "submit",
                    "image_base64": "<base64 编码的图片>",
                    "labels": [{"class": "ZangWu", "x": 0.1, "y": 0.2, "w": 0.05, "h": 0.04}],
                },
                "stats": _stats(),
            }
        return _action_submit(payload)
    raise ValueError("unsupported action: %s" % action)


def _action_submit(payload: dict[str, Any]) -> dict[str, Any]:
    result = _submit(payload)
    return {"action": "submit", **result}


async def on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    if task:
        return task

    payload = _extract_payload(command)
    if payload is None:
        # 平台侧调用常不带结构化入参：默认返回只读统计，避免阻塞为「等待输入」
        payload = {"action": "stats"}

    try:
        result = _action(payload)
    except Exception as exc:  # noqa: BLE001
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[TextDataItem(text="回流处理失败：%s: %s" % (type(exc).__name__, exc))],
        )
        failed.senderId = LOCAL_AIC
        return failed

    done = TaskManager.create_task(command, initial_state=TaskState.AwaitingCompletion)
    TaskManager.set_products(
        done.taskId,
        [
            Product(
                id="product-%s" % done.taskId,
                name="chip-feedback",
                dataItems=[StructuredDataItem(data=result)],
            )
        ],
    )
    task_result = TaskManager.get_task(done.taskId) or done
    task_result.senderId = LOCAL_AIC
    return task_result



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
    return {"status": "ok", "aic": LOCAL_AIC, "stats": _stats()}
