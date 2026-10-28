from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import datetime
from typing import Any

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

PARTNER_AIC = "chip-report-agent"

DEFAULTS = {
    "conf_threshold": 0.5,
    "max_defects_per_image": 5,
    "max_defects_total": 20,
}

CLASS_ORDER = ["ZangWu", "PoSun", "BengLie", "DuMoTuoLuo"]
CLASS_LABELS = {
    "ZangWu": "脏污",
    "PoSun": "破损",
    "BengLie": "崩裂",
    "DuMoTuoLuo": "镀膜脱落",
}

VERDICT_LABELS = {
    "OK": "合格",
    "NG": "不合格",
    "REVIEW": "待复核",
}

app = FastAPI(title="Chip Quality Report Partner")


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


_UNWRAP_KEYS = ("input", "payload", "data", "params", "arguments")


def _as_dict(value: Any) -> dict[str, Any] | None:
    """把 dict / JSON 文本 / 内嵌 JSON 的说明文字统一转成 dict。"""
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start_i = text.find("{")
            end_i = text.rfind("}")
            if start_i < 0 or end_i <= start_i:
                return None
            try:
                value = json.loads(text[start_i:end_i + 1])
            except json.JSONDecodeError:
                return None
    if not isinstance(value, dict):
        return None
    for key in _UNWRAP_KEYS:
        if key in value:
            inner = _as_dict(value[key])
            if inner is not None:
                merged = {k: v for k, v in value.items() if k not in _UNWRAP_KEYS}
                merged.update(inner)
                return merged
    return value


def _extract_payload(command: TaskCommand) -> dict[str, Any] | None:
    for item in command.dataItems or []:
        if isinstance(item, StructuredDataItem):
            got = _as_dict(item.data)
            if got is not None:
                return got
        if isinstance(item, FileDataItem):
            raw = getattr(item, "bytes", None)
            if raw:
                got = _as_dict(raw)
                if got is not None:
                    return got
        if isinstance(item, TextDataItem) and item.text:
            got = _as_dict(item.text)
            if got is not None:
                return got
    return None


def _normalise(payload: dict[str, Any]) -> tuple[list[dict], dict]:
    images = payload.get("images")
    if images is None and payload.get("detections") is not None:
        images = [
            {
                "filename": payload.get("filename", "image"),
                "image_width": payload.get("image_width"),
                "image_height": payload.get("image_height"),
                "detections": payload.get("detections", []),
            }
        ]
    if not isinstance(images, list) or not images:
        raise ValueError("payload.images must be a non-empty list")
    options = dict(DEFAULTS)
    if isinstance(payload.get("options"), dict):
        options.update({k: v for k, v in payload["options"].items() if k in DEFAULTS})
    return images, options


def _build_report(payload: dict[str, Any]) -> dict[str, Any]:
    images, options = _normalise(payload)
    conf_threshold = float(options["conf_threshold"])
    max_per_image = int(options["max_defects_per_image"])
    max_total = int(options["max_defects_total"])

    total = 0
    counts: Counter[str] = Counter()
    max_conf = 0.0
    image_rows = []
    reasons: list[str] = []

    for image in images:
        detections = image.get("detections") or []
        if not isinstance(detections, list):
            detections = []
        clean = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            conf = float(det.get("confidence", 0) or 0)
            clean.append(
                {
                    "class_id": det.get("class_id"),
                    "class_name": str(det.get("class_name", "unknown")),
                    "confidence": round(conf, 4),
                    "xyxy": det.get("xyxy"),
                }
            )
            counts[str(det.get("class_name", "unknown"))] += 1
            max_conf = max(max_conf, conf)
        clean.sort(key=lambda d: d["confidence"], reverse=True)
        total += len(clean)
        image_rows.append(
            {
                "filename": image.get("filename", "image"),
                "width": image.get("image_width"),
                "height": image.get("image_height"),
                "detection_count": len(clean),
                "high_confidence_count": sum(
                    1 for d in clean if d["confidence"] >= conf_threshold
                ),
                "max_confidence": max(d["confidence"] for d in clean) if clean else 0,
                "detections": clean,
            }
        )
        if len(clean) > max_per_image:
            reasons.append(
                "%s 检出 %d 处，超过单图阈值 %d 处"
                % (image_rows[-1]["filename"], len(clean), max_per_image)
            )
        if any(d["confidence"] >= conf_threshold for d in clean):
            reasons.append(
                "%s 存在置信度 ≥ %.2f 的缺陷"
                % (image_rows[-1]["filename"], conf_threshold)
            )

    if total > max_total:
        reasons.append("总检出 %d 处，超过批次阈值 %d 处" % (total, max_total))

    if total == 0:
        verdict = "OK"
        reasons = ["未检出缺陷"]
    elif reasons:
        verdict = "NG"
    else:
        verdict = "REVIEW"
        reasons = ["检出缺陷但未超过判定阈值，建议人工复核"]

    summary = {name: int(counts.get(name, 0)) for name in CLASS_ORDER}
    for name, value in counts.items():
        summary.setdefault(name, int(value))

    markdown = _to_markdown(
        payload,
        verdict,
        summary,
        total,
        max_conf,
        image_rows,
        reasons,
        options,
    )

    return {
        "batch_id": payload.get("batch_id") or "batch-%s" % uuid.uuid4().hex[:8],
        "generated_at": _now(),
        "verdict": verdict,
        "verdict_label": VERDICT_LABELS[verdict],
        "image_count": len(image_rows),
        "defect_total": total,
        "max_confidence": round(max_conf, 4),
        "summary": summary,
        "summary_labels": {k: CLASS_LABELS.get(k, k) for k in summary},
        "reasons": list(dict.fromkeys(reasons)),
        "options": options,
        "images": image_rows,
        "report_markdown": markdown,
    }


def _to_markdown(payload, verdict, summary, total, max_conf, image_rows, reasons, options) -> str:
    lines = [
        "# 芯片封装质检报告",
        "",
        "- 批次号：%s" % (payload.get("batch_id") or "自动生成"),
        "- 生成时间：%s" % _now(),
        "- 结论：**%s（%s）**" % (VERDICT_LABELS[verdict], verdict),
        "",
        "## 汇总",
        "",
        "| 缺陷类别 | 数量 |",
        "| --- | --- |",
    ]
    for name in CLASS_ORDER:
        if name in summary:
            lines.append("| %s（%s） | %d |" % (CLASS_LABELS.get(name, name), name, summary[name]))
    for name, value in summary.items():
        if name not in CLASS_ORDER:
            lines.append("| %s | %d |" % (name, value))
    lines += [
        "",
        "- 检出总数：%d" % total,
        "- 最高置信度：%.3f" % max_conf,
        "- 涉及图片：%d 张" % len(image_rows),
        "- 判定阈值：置信度 ≥ %.2f · 单图 ≤ %d · 批次 ≤ %d"
        % (
            options["conf_threshold"],
            options["max_defects_per_image"],
            options["max_defects_total"],
        ),
        "",
        "## 判定依据",
        "",
    ]
    for reason in dict.fromkeys(reasons):
        lines.append("- %s" % reason)
    lines += ["", "## 图片明细", ""]
    for row in image_rows:
        lines.append(
            "### %s（%s×%s）" % (row["filename"], row["width"], row["height"])
        )
        lines.append(
            "- 检出 %d 处 · ≥阈值 %d 处 · 最高 %.3f"
            % (row["detection_count"], row["high_confidence_count"], row["max_confidence"])
        )
        for index, det in enumerate(row["detections"][:20], start=1):
            lines.append(
                "  %d. %s %.1f%% %s"
                % (
                    index,
                    det["class_name"],
                    det["confidence"] * 100,
                    det["xyxy"] if det.get("xyxy") else "",
                )
            )
        if len(row["detections"]) > 20:
            lines.append("  … 其余 %d 处见结构化数据" % (len(row["detections"]) - 20))
        lines.append("")
    lines.append("> 本报告为辅助质检建议，最终判定需人工复核。")
    return "\n".join(lines)


async def on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    if task:
        return task

    payload = _extract_payload(command)
    if payload is None:
        pending = TaskManager.create_task(
            command,
            initial_state=TaskState.AwaitingInput,
            data_items=[
                TextDataItem(
                    text='请提供 JSON：{"images":[{"filename":"a.jpg","detections":[...]}],"options":{...}}'
                )
            ],
        )
        pending.senderId = LOCAL_AIC
        return pending

    try:
        report = _build_report(payload)
    except Exception as exc:  # noqa: BLE001
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[TextDataItem(text="报告生成失败：%s: %s" % (type(exc).__name__, exc))],
        )
        failed.senderId = LOCAL_AIC
        return failed

    done = TaskManager.create_task(command, initial_state=TaskState.AwaitingCompletion)
    TaskManager.set_products(
        done.taskId,
        [
            Product(
                id="product-%s" % done.taskId,
                name="chip-quality-report",
                dataItems=[
                    StructuredDataItem(data=report),
                    TextDataItem(text=report["report_markdown"]),
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


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "aic": LOCAL_AIC}
