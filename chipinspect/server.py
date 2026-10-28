from __future__ import annotations

import base64
from urllib.parse import urlparse
import httpx
import binascii
import json
import logging
import math
import os
import re
import secrets
import ssl
import time
import traceback
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import mtls_config as mtlscfg

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from .benchmark import benchmark_status, run_benchmark
from .model_runner import (
    Detector,
    ModelNotReadyError,
    ModelRuntimeError,
    YoloDetector,
)
from .preprocess import (
    NATIVE_FORMATS,
    PreprocessError,
    encode_image,
    map_detections_to_original,
    prepare_image,
    preprocess_image,
)
from acps_sdk.aip import (
    AipRpcClient,
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
from acps_sdk.amp.access_emitter import AccessEmitter
import amp_message as ampmsg

MAX_UPLOAD_BYTES = int(os.getenv("CHIP_MAX_UPLOAD_BYTES", 20 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.getenv("CHIP_MAX_IMAGE_PIXELS", 40_000_000))
MAX_BATCH_FILES = int(os.getenv("CHIP_MAX_BATCH_FILES", 12))
ALLOWED_FORMATS = set(NATIVE_FORMATS) | {"PDF"}
WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
LOGGER = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(
    title="ChipInspect API",
    version="0.1.0",
    description="Versioned API for chip-package surface-defect inspection.",
)
app.state.detector = YoloDetector()


def _decode_image(raw: bytes) -> Image.Image:
    if not raw:
        raise HTTPException(status_code=400, detail="上传的图片为空。")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"图片超过大小限制（{MAX_UPLOAD_BYTES // (1024 * 1024)} MB）。",
        )
    try:
        with Image.open(BytesIO(raw)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=415, detail="暂不支持该图片格式。")
            probe.verify()
        with Image.open(BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="图片分辨率超过安全限制。")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="无法读取图片，请更换有效的图像文件。") from exc
    return image


def _parse_region(region_json: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not region_json:
        return None
    try:
        raw_region = json.loads(region_json)
        values = [float(raw_region[key]) for key in ("x1", "y1", "x2", "y2")]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="裁剪区域格式无效。") from exc
    if not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=400, detail="裁剪区域坐标无效。")
    x1, y1, x2, y2 = values
    left = max(0, min(width, math.floor(x1)))
    top = max(0, min(height, math.floor(y1)))
    right = max(0, min(width, math.ceil(x2)))
    bottom = max(0, min(height, math.ceil(y2)))
    if right - left < 2 or bottom - top < 2:
        raise HTTPException(status_code=400, detail="裁剪区域太小或超出图片范围。")
    return left, top, right, bottom


_BOX_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 140, 255),
    (255, 200, 60),
    (200, 120, 255),
    (80, 220, 220),
]


def _load_font(size: int):
    for name in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _annotate_image(image: Image.Image, detections: list[dict[str, Any]]) -> bytes:
    """Draw numbered boxes onto a copy of the original image."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    base = min(canvas.size)
    line_width = max(3, int(base / 300))
    font_size = max(14, int(base / 45))
    font = _load_font(font_size)

    for index, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["xyxy"]
        color = _BOX_COLORS[det["class_id"] % len(_BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        label = "%d. %s %.1f%%" % (
            index,
            det["class_name"],
            det["confidence"] * 100,
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        pad = max(4, int(font_size * 0.25))
        top = y1 - text_h - pad * 2
        if top < 0:
            top = y1
        draw.rectangle(
            [x1, top, x1 + text_w + pad * 2, top + text_h + pad * 2],
            fill=color,
        )
        draw.text((x1 + pad, top + pad - bbox[1]), label, fill=(255, 255, 255), font=font)

    buf = BytesIO()
    canvas.save(buf, "JPEG", quality=90)
    return buf.getvalue()


MTLS_ENABLED = os.getenv("CHIP_MTLS_ENABLED", "0") == "1"
_SCHEME = "https" if MTLS_ENABLED else "http"
PREPROCESS_RPC = os.getenv("CHIP_PREPROCESS_RPC", _SCHEME + "://127.0.0.1:8013/rpc")
DETECT_RPC = os.getenv("CHIP_DETECT_RPC", _SCHEME + "://127.0.0.1:8012/rpc")
REPORT_RPC = os.getenv("CHIP_REPORT_RPC", _SCHEME + "://127.0.0.1:8015/rpc")
FEEDBACK_RPC = os.getenv("CHIP_FEEDBACK_RPC", _SCHEME + "://127.0.0.1:8016/rpc")

DISCOVERY_URL = os.getenv("CHIP_DISCOVERY_URL", "https://wt.ioa.pub/discovery").rstrip("/")
DISCOVERY_TTL_SECONDS = float(os.getenv("CHIP_DISCOVERY_TTL", "300"))
_DISCOVERY_CACHE: dict[str, tuple[float, str]] = {}


_LOCAL_PARTNER_PORTS = {"8443": 8013, "8444": 8012, "8445": 8015, "8446": 8016}


def _prefer_local(url: str) -> str:
    if os.getenv("CHIP_PREFER_LOCAL_PARTNERS", "1") != "1":
        return url
    try:
        parsed = urlparse(url)
    except Exception:  # noqa: BLE001
        return url
    local_port = _LOCAL_PARTNER_PORTS.get(str(parsed.port or ""))
    if not local_port:
        return url
    path = parsed.path or "/rpc"
    return "%s://127.0.0.1:%d%s" % (parsed.scheme, local_port, path)


async def _resolve_partner(skill_id: str, fallback_url: str) -> str:
    now = time.time()
    cached = _DISCOVERY_CACHE.get(skill_id)
    if cached and cached[0] > now:
        return cached[1]
    query_map = {
        "chip-image.preprocess": "芯片图像预处理",
        "chip-inspection.defect-detection": "芯片缺陷检测",
        "chip-quality.report": "质检报告",
        "chip-quality.feedback": "质检数据回流",
    }
    # 平台发现服务按技能 ID 建索引，中文名检索命中率低；优先用技能 ID 查询
    payload = {"type": "explicit", "query": skill_id, "limit": 20}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.post(DISCOVERY_URL + "/acps-adp-v2/discover", json=payload)
            resp.raise_for_status()
            data = resp.json()
        acs_map = ((data.get("result") or {}).get("acsMap") or {})
        for aic, acs in acs_map.items():
            if not isinstance(acs, dict) or not acs.get("active", True):
                continue
            skills = [str(s.get("id") or "") for s in (acs.get("skills") or [])]
            if skill_id not in skills:
                continue
            for endpoint in acs.get("endPoints") or []:
                url = str(endpoint.get("url") or "")
                if url.startswith("https://"):
                    local_url = _prefer_local(url)
                    _DISCOVERY_CACHE[skill_id] = (now + DISCOVERY_TTL_SECONDS, local_url)
                    return local_url
    except Exception:
        pass
    return _prefer_local(fallback_url)

DETECT_ADMIN_URL = os.getenv("CHIP_DETECT_ADMIN_URL", _SCHEME + "://127.0.0.1:8012")
ADMIN_TOKEN_FILE = Path(r"%CHIP_ROOT%\frp\config\admin_token.txt")
ADMIN_TOKEN = (
    ADMIN_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if ADMIN_TOKEN_FILE.exists()
    else ""
)


def _expected_aic_for(partner_url: str) -> str:
    if ":8013" in partner_url or ":8443" in partner_url:
        return mtlscfg.PARTNER_AICS["preprocess"]
    if ":8012" in partner_url or ":8444" in partner_url:
        return mtlscfg.PARTNER_AICS["detect"]
    if ":8015" in partner_url or ":8445" in partner_url:
        return mtlscfg.PARTNER_AICS["report"]
    if ":8016" in partner_url or ":8446" in partner_url:
        return mtlscfg.PARTNER_AICS["feedback"]
    return ""


_TRACE: list = []
_AGENT_NAMES = {":8013": "预处理", ":8443": "预处理", ":8012": "检测", ":8444": "检测", ":8015": "报告", ":8445": "报告", ":8016": "回流", ":8446": "回流"}


def _agent_name(url: str) -> str:
    for suffix, name in _AGENT_NAMES.items():
        if suffix in url:
            return name
    return "unknown"


def _record_agent(url: str, started: float, status: str, error: str = "") -> None:
    _TRACE.append(
        {
            "agent": _agent_name(url),
            "aic": _expected_aic_for(url),
            "endpoint": url,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "status": status,
            "error": error,
        }
    )


def _client_ssl_context():
    if not MTLS_ENABLED:
        return None
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.load_verify_locations(mtlscfg.TRUST_BUNDLE)
    ctx.load_cert_chain(mtlscfg.LEADER_CERT, mtlscfg.LEADER_KEY)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


async def _emit_message(event_type: str, message_id: str, destination: str, payload_size: int = 0) -> None:
    """写一条 AMP message 日志；失败不影响主流程。"""
    global _MESSAGE_EMITTER
    try:
        if _MESSAGE_EMITTER is None:
            _MESSAGE_EMITTER = ampmsg.make_emitter("leader", LEADER_LOCAL_AIC)
        await _MESSAGE_EMITTER.emit(
            ampmsg.message_body(event_type, message_id, destination, payload_size),
            correlation_id=message_id,
        )
    except Exception:
        pass


async def _aip_call(partner_url: str, payload: Any, expected_aic_override: str | None = None) -> list:
    text = (
        json.dumps(payload, ensure_ascii=False)
        if isinstance(payload, (dict, list))
        else str(payload)
    )
    started = time.perf_counter()
    expected_aic = expected_aic_override or _expected_aic_for(partner_url)
    client = AipRpcClient(
        partner_url=partner_url,
        leader_id=mtlscfg.LEADER_AIC,
        ssl_context=_client_ssl_context(),
        access_emitter=_ACCESS_EMITTER,
        expected_partner_aic=expected_aic or None,
        callee_aic=expected_aic or "",
        caller_service="chipinspect-leader",
        callee_service=_agent_name(partner_url),
        identity_binding_enabled=MTLS_ENABLED,
    )
    task_id = "task-%s" % uuid.uuid4()
    session_id = task_id
    try:
        task = await client.start_task(
            session_id=session_id, task_id=task_id, user_input=text
        )
        await _emit_message("send", task_id, expected_aic or "", len(text.encode("utf-8")))
        if task.status.state != TaskState.AwaitingCompletion:
            raise HTTPException(
                status_code=502,
                detail="partner %s returned %s" % (partner_url, task.status.state),
            )
        items = []
        for product in task.products or []:
            for item in product.dataItems or []:
                items.append(item)
        await client.complete_task(task_id=task_id, session_id=session_id)
        await _emit_message("ack", task_id, expected_aic or "", 0)
        _record_agent(partner_url, started, "ok")
        return items
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        _record_agent(partner_url, started, "error", "%s: %s" % (type(exc).__name__, exc))
        raise HTTPException(
            status_code=502,
            detail="partner %s failed: %s: %s"
            % (partner_url, type(exc).__name__, exc),
        ) from exc
    finally:
        await client.close()


def _get_detector(request: Request) -> Detector:
    return request.app.state.detector


def _run_detection(
    detector: Detector,
    image: Image.Image,
    request_id: str,
    crop_region: tuple[int, int, int, int] | None,
    imgsz: int,
    conf: float,
    filename: str,
    auto_crop: bool = False,
    target_size: int = 0,
    fit_mode: str = "letterbox",
    tolerance: int = 12,
    annotate: bool = True,
) -> dict[str, Any]:
    image_width, image_height = image.size
    base_x, base_y = 0, 0
    if crop_region:
        work = image.crop(crop_region)
        base_x, base_y = crop_region[0], crop_region[1]
    else:
        work = image

    transform = None
    applied_crop = None
    if auto_crop or (target_size and target_size > 0):
        try:
            work, transform, applied_crop = prepare_image(
                work,
                auto_crop=auto_crop,
                target_size=target_size or None,
                fit_mode=fit_mode,
                tolerance=tolerance,
            )
        except PreprocessError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if applied_crop:
            base_x += applied_crop[0]
            base_y += applied_crop[1]

    started = time.perf_counter()
    try:
        result = detector.predict(work, imgsz=imgsz, conf=conf)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelRuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    total_ms = (time.perf_counter() - started) * 1000

    def _to_original(xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if transform is not None:
            x1, y1, x2, y2 = transform.to_input(xyxy)
        else:
            x1, y1, x2, y2 = xyxy
        return (x1 + base_x, y1 + base_y, x2 + base_x, y2 + base_y)

    detections = []
    for item in result.detections:
        x1, y1, x2, y2 = _to_original(item.xyxy)
        detections.append(
            {
                "class_id": item.class_id,
                "class_name": item.class_name,
                "confidence": round(item.confidence, 6),
                "xyxy": [
                    round(max(0.0, min(float(image_width), x1)), 2),
                    round(max(0.0, min(float(image_height), y1)), 2),
                    round(max(0.0, min(float(image_width), x2)), 2),
                    round(max(0.0, min(float(image_height), y2)), 2),
                ],
            }
        )

    preprocess_meta = None
    if transform is not None:
        preprocess_meta = {
            "auto_crop": auto_crop,
            "target_size": target_size or None,
            "fit_mode": fit_mode,
            "tolerance": tolerance,
            "crop_box": list(transform.crop_box) if transform.crop_box else None,
        }

    return {
        "request_id": request_id,
        "image": {
            "filename": Path(filename).name[:180],
            "width": image_width,
            "height": image_height,
        },
        "crop_region": (
            {"x1": crop_region[0], "y1": crop_region[1], "x2": crop_region[2], "y2": crop_region[3]}
            if crop_region
            else None
        ),
        "preprocess": preprocess_meta,
        "detection_count": len(detections),
        "detections": detections,
        "timing_ms": {
            "inference": round(result.inference_ms, 2),
            "server_total": round(total_ms, 2),
        },
        "model": {
            "name": result.model_name,
            "device": result.device,
        },
        "annotated_image_base64": (
            base64.b64encode(_annotate_image(image, detections)).decode("ascii")
            if annotate
            else None
        ),
        "annotated_mime": "image/jpeg" if annotate else None,
        "requires_human_review": any(item["confidence"] < 0.5 for item in detections)
        or not detections,
    }


AUTH_USER = os.getenv("CHIP_AUTH_USER", "")
AUTH_PASSWORD = os.getenv("CHIP_AUTH_PASSWORD", "")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not AUTH_USER or not AUTH_PASSWORD:
        return await call_next(request)
    if request.url.path == "/health":
        return await call_next(request)
    header = request.headers.get("authorization", "")
    if header.startswith("Basic "):
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8")
            username, _, password = decoded.partition(":")
            if secrets.compare_digest(username, AUTH_USER) and secrets.compare_digest(
                password, AUTH_PASSWORD
            ):
                return await call_next(request)
        except Exception:  # noqa: BLE001
            pass
    return Response(
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="ChipInspect"'},
    )


@app.middleware("http")
async def url_token_gate(request: Request, call_next):
    if not mtlscfg.URL_TOKEN:
        return await call_next(request)
    if request.url.path == "/health":
        return await call_next(request)
    # AIP RPC over TLS 由 mTLS 客户端证书做身份校验，不应再要求 URL token；
    # 明文端口（8080 网页）上的 /rpc 仍然需要 token。
    if request.url.path == "/rpc" and request.url.scheme == "https":
        return await call_next(request)
    # 图片分享路径免 token：文件名带随机串，不可枚举；便于在第三方平台直接引用
    if request.url.path.startswith("/s/"):
        return await call_next(request)
    supplied = request.query_params.get("key") or request.cookies.get("chip_token")
    if supplied and secrets.compare_digest(supplied, mtlscfg.URL_TOKEN):
        response = await call_next(request)
        if request.query_params.get("key") and request.cookies.get("chip_token") != mtlscfg.URL_TOKEN:
            response.set_cookie(
                "chip_token",
                mtlscfg.URL_TOKEN,
                max_age=7 * 24 * 3600,
                httponly=True,
                samesite="lax",
            )
        return response
    return Response(status_code=403, content="Forbidden")


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["Cache-Control"] = "no-store"
    return response


PUBLIC_IMAGES = Path(os.getenv("CHIP_PUBLIC_IMAGES", str(Path(__file__).resolve().parents[2] / "public_images")))


@app.get("/s/{name}")
async def public_image(name: str):
    """把上传的图片发布成可分享链接；仍受 URL token 保护。"""
    from fastapi.responses import FileResponse
    from fastapi import HTTPException

    safe = os.path.basename(name)
    if not safe or safe.startswith("."):
        raise HTTPException(status_code=400, detail="invalid name")
    target = (PUBLIC_IMAGES / safe).resolve()
    if not str(target).startswith(str(PUBLIC_IMAGES.resolve())):
        raise HTTPException(status_code=400, detail="invalid path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), headers={"Cache-Control": "no-store"})


@app.get("/health")
async def health(request: Request):
    detector = _get_detector(request)
    return {
        "status": "ok",
        "service": "chipinspect",
        "api_version": "v1",
        "model": detector.status(),
    }


@app.get("/api/v1/model")
async def model_info(request: Request):
    return {"model": _get_detector(request).status()}


@app.get("/api/v1/benchmark")
async def benchmark_info():
    return benchmark_status()


@app.post("/api/v1/benchmark/run")
def compare_models():
    try:
        return run_benchmark()
    except (FileNotFoundError, ValueError, ModelRuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Local model benchmark failed")
        raise HTTPException(
            status_code=503,
            detail=f"对比评估失败：{type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/detect")
async def detect(
    request: Request,
    image: UploadFile = File(...),
    crop_region: str | None = Form(default=None),
    imgsz: int = Form(default=960, ge=320, le=2048),
    conf: float = Form(default=0.25, gt=0, le=1),
    auto_crop: bool = Form(default=True),
    target_size: int = Form(default=0, ge=0, le=2048),
    fit_mode: str = Form(default="letterbox"),
    tolerance: int = Form(default=12, ge=0, le=255),
    annotate: bool = Form(default=True),
):
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    decoded = _decode_image(raw)
    region = _parse_region(crop_region, *decoded.size) if crop_region else None
    return _run_detection(
        _get_detector(request),
        decoded,
        request.state.request_id,
        region,
        imgsz,
        conf,
        image.filename or "upload",
        auto_crop=auto_crop,
        target_size=target_size,
        fit_mode=fit_mode,
        tolerance=tolerance,
        annotate=annotate,
    )


@app.post("/api/v1/batch")
async def batch_detect(
    request: Request,
    images: list[UploadFile] = File(...),
    imgsz: int = Form(default=960, ge=320, le=2048),
    conf: float = Form(default=0.25, gt=0, le=1),
    auto_crop: bool = Form(default=True),
    target_size: int = Form(default=0, ge=0, le=2048),
    fit_mode: str = Form(default="letterbox"),
    tolerance: int = Form(default=12, ge=0, le=255),
    annotate: bool = Form(default=False),
):
    if not images or len(images) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"一次可提交 1 到 {MAX_BATCH_FILES} 张图片。",
        )
    detector = _get_detector(request)
    results = []
    for upload in images:
        raw = await upload.read(MAX_UPLOAD_BYTES + 1)
        decoded = _decode_image(raw)
        results.append(
            _run_detection(
                detector,
                decoded,
                str(uuid.uuid4()),
                None,
                imgsz,
                conf,
                upload.filename or "upload",
                auto_crop=auto_crop,
                target_size=target_size,
                fit_mode=fit_mode,
                tolerance=tolerance,
                annotate=annotate,
            )
        )
    return {
        "request_id": request.state.request_id,
        "image_count": len(results),
        "detection_count": sum(item["detection_count"] for item in results),
        "results": results,
    }


@app.post("/aip/rpc")
async def aip_rpc(request: Request):
    """JSON-RPC adapter; image content is inline base64 for the local MVP."""
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="请求体必须为 JSON。") from exc
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        raise HTTPException(status_code=400, detail="仅接受 JSON-RPC 2.0 请求。")

    call_id = body.get("id")
    params = body.get("params") or {}
    command = params.get("command") if isinstance(params, dict) else None
    if not isinstance(command, dict) or command.get("command") != "detect":
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32601, "message": "Unsupported command; use params.command.command=detect"},
        }

    image_b64 = command.get("imageBase64")
    if not isinstance(image_b64, str) or not image_b64:
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32602, "message": "params.command.imageBase64 is required"},
        }
    if len(image_b64) > ((MAX_UPLOAD_BYTES + 2) // 3) * 4 + 8:
        raise HTTPException(status_code=413, detail="图片超过大小限制。")
    try:
        raw = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32602, "message": "imageBase64 is invalid"},
        }

    decoded = _decode_image(raw)
    try:
        imgsz = int(command.get("imgsz", 960))
        conf = float(command.get("conf", 0.25))
    except (TypeError, ValueError):
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32602, "message": "imgsz/conf must be numeric"},
        }
    if imgsz < 320 or imgsz > 2048 or not math.isfinite(conf) or conf <= 0 or conf > 1:
        return {
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32602, "message": "imgsz/conf is outside supported range"},
        }
    region_value = command.get("cropRegion")
    region_json = json.dumps(region_value) if region_value is not None else None
    region = _parse_region(region_json, *decoded.size) if region_json else None
    try:
        result = _run_detection(
            _get_detector(request),
            decoded,
            str(call_id or request.state.request_id),
            region,
            imgsz,
            conf,
            str(command.get("filename") or "aip-image"),
        )
    except HTTPException as exc:
        if exc.status_code == 503:
            return {
                "jsonrpc": "2.0",
                "id": call_id,
                "error": {"code": -32000, "message": str(exc.detail)},
            }
        raise
    return {"jsonrpc": "2.0", "id": call_id, "result": result}




async def _pipeline_core(
    raw: bytes,
    filename: str,
    imgsz: int,
    conf: float,
    auto_crop: bool,
    target_size: int,
    fit_mode: str,
    tolerance: int,
    include_report: bool,
    request_id: str,
) -> dict[str, Any]:
    _TRACE.clear()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过大小限制。")
    decoded = _decode_image(raw)
    width, height = decoded.size
    filename = Path(filename or "upload").name[:180]

    # 1) preprocess partner: multi-format decode + auto crop + fit
    pre_items = await _aip_call(
        await _resolve_partner("chip-image.preprocess", PREPROCESS_RPC),
        {
            "image_base64": base64.b64encode(raw).decode("ascii"),
            "filename": filename,
            "auto_crop": auto_crop,
            "target_size": target_size,
            "fit_mode": fit_mode,
            "tolerance": tolerance,
            "output_format": "PNG",
        },
    )
    pre_meta = None
    processed_b64 = None
    for item in pre_items:
        if isinstance(item, StructuredDataItem):
            pre_meta = item.data
        elif isinstance(item, FileDataItem) and item.bytes:
            processed_b64 = item.bytes
    if not processed_b64:
        raise HTTPException(status_code=502, detail="preprocess partner returned no image")

    # 2) detect partner on the preprocessed image
    det_items = await _aip_call(await _resolve_partner("chip-inspection.defect-detection", DETECT_RPC), processed_b64)
    det_data = None
    for item in det_items:
        if isinstance(item, StructuredDataItem):
            det_data = item.data
    if det_data is None:
        raise HTTPException(status_code=502, detail="detect partner returned no data")

    detections = map_detections_to_original(
        det_data.get("detections") or [], pre_meta or {}
    )

    # 3) report partner
    report = None
    if include_report:
        rep_items = await _aip_call(
            await _resolve_partner("chip-quality.report", REPORT_RPC),
            {
                "batch_id": "pipe-%s" % uuid.uuid4().hex[:8],
                "images": [
                    {
                        "filename": filename,
                        "image_width": width,
                        "image_height": height,
                        "detections": detections,
                    }
                ],
            },
        )
        for item in rep_items:
            if isinstance(item, StructuredDataItem):
                report = item.data

    annotated = _annotate_image(decoded, detections)
    return {
        "request_id": request_id,
        "image": {"filename": filename, "width": width, "height": height},
        "preprocess": pre_meta,
        "detection_count": len(detections),
        "detections": detections,
        "annotated_image_base64": base64.b64encode(annotated).decode("ascii"),
        "annotated_mime": "image/jpeg",
        "report": report,
        "agents": list(_TRACE),
        "timing_ms": {
            "inference": det_data.get("inference_ms", 0),
            "server_total": 0,
        },
        "model": {
            "name": det_data.get("model"),
            "device": det_data.get("device"),
        },
        "requires_human_review": (report or {}).get("verdict") != "OK",
    }
@app.post("/api/v1/pipeline")
async def pipeline(
    request: Request,
    image: UploadFile = File(...),
    imgsz: int = Form(default=640, ge=320, le=2048),
    conf: float = Form(default=0.25, gt=0, le=1),
    auto_crop: bool = Form(default=True),
    target_size: int = Form(default=640, ge=0, le=2048),
    fit_mode: str = Form(default="letterbox"),
    tolerance: int = Form(default=12, ge=0, le=255),
    include_report: bool = Form(default=True),
):
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    filename = Path(image.filename or "upload").name[:180]
    return await _pipeline_core(
        raw, filename, imgsz, conf, auto_crop, target_size,
        fit_mode, tolerance, include_report, request.state.request_id,
    )

@app.post("/api/v1/preprocess")
async def preprocess_endpoint(
    image: UploadFile = File(...),
    auto_crop: bool = Form(default=True),
    target_size: int = Form(default=640, ge=0, le=2048),
    fit_mode: str = Form(default="letterbox"),
    tolerance: int = Form(default=12, ge=0, le=255),
    pad: int = Form(default=0, ge=0, le=256),
    output_format: str = Form(default="PNG"),
):
    raw = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="文件超过大小限制。")
    try:
        result = preprocess_image(
            raw,
            auto_crop=auto_crop,
            target_size=target_size or None,
            fit_mode=fit_mode,
            tolerance=tolerance,
            pad=pad,
        )
        data, mime = encode_image(result.image, output_format)
    except PreprocessError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "filename": Path(image.filename or "upload").name[:180],
        "source_format": result.source_format,
        "original_size": list(result.original_size),
        "crop_box": list(result.crop_box) if result.crop_box else None,
        "output_size": list(result.output_size),
        "steps": result.steps,
        "mime": mime,
        "image_base64": base64.b64encode(data).decode("ascii"),
    }


@app.get("/api/v1/models")
async def proxy_models():
    headers = {"x-admin-token": ADMIN_TOKEN} if ADMIN_TOKEN else {}
    try:
        async with httpx.AsyncClient(timeout=20, verify=_client_ssl_context() or True) as client:
            response = await client.get(DETECT_ADMIN_URL + "/models", headers=headers)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="detect partner unreachable: %s" % exc)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=response.text[:300])
    return response.json()


@app.post("/api/v1/model/select")
async def proxy_select_model(payload: dict):
    headers = {"x-admin-token": ADMIN_TOKEN} if ADMIN_TOKEN else {}
    try:
        async with httpx.AsyncClient(timeout=180, verify=_client_ssl_context() or True) as client:
            response = await client.post(
                DETECT_ADMIN_URL + "/model/select",
                json=payload,
                headers=headers,
            )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail="detect partner unreachable: %s" % exc)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text[:300])
    return response.json()


@app.get("/api/v1/agents")
async def list_agents():
    specs = [
        ("预处理", PREPROCESS_RPC),
        ("检测", DETECT_RPC),
        ("报告", REPORT_RPC),
        ("回流", FEEDBACK_RPC),
    ]
    out = []
    for name, rpc in specs:
        base = rpc.rsplit("/rpc", 1)[0]
        started = time.perf_counter()
        status = "down"
        detail = ""
        try:
            async with httpx.AsyncClient(
                timeout=5, verify=_client_ssl_context() or True
            ) as client:
                resp = await client.get(base + "/health")
                status = "ok" if resp.status_code == 200 else ("http-%s" % resp.status_code)
                detail = resp.text[:160]
        except Exception as exc:  # noqa: BLE001
            detail = "%s: %s" % (type(exc).__name__, exc)
        out.append(
            {
                "name": name,
                "aic": _expected_aic_for(rpc),
                "endpoint": base,
                "ms": round((time.perf_counter() - started) * 1000, 1),
                "status": status,
                "detail": detail,
            }
        )
    return {"agents": out, "mtls": MTLS_ENABLED}


@app.post("/api/v1/feedback")
async def submit_feedback(payload: dict):
    data = dict(payload)
    data.setdefault("action", "submit")
    _TRACE.clear()
    items = await _aip_call(await _resolve_partner("chip-quality.feedback", FEEDBACK_RPC), data)
    result = None
    for item in items:
        if isinstance(item, StructuredDataItem):
            result = item.data
    return {"result": result, "agents": list(_TRACE)}


LEADER_LOCAL_AIC = os.getenv("CHIP_AGENT_AIC", mtlscfg.LEADER_AIC)

AMP_LOG_DIR = Path(os.getenv("AMP_LOG_DIR", "logs"))
_MESSAGE_EMITTER = None
AMP_LOG_DIR.mkdir(parents=True, exist_ok=True)
_ACCESS_EMITTER = AccessEmitter(
    AMP_LOG_DIR / "amp_access_leader.jsonl",
    aic=LEADER_LOCAL_AIC,
) if os.getenv("CHIP_AMP_ACCESS", "1") == "1" else None
IDENTITY_BINDING = os.getenv("CHIP_IDENTITY_BINDING", "0") == "1"
if IDENTITY_BINDING:
    app.add_middleware(AipPeerCertificateMiddleware)


def _fetch_image_url(url: str, max_bytes: int = 20 * 1024 * 1024) -> bytes:
    """从 http(s) 链接取图像；供平台侧直接粘贴图片链接使用。"""
    import httpx

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
    if not data:
        raise ValueError("链接返回空内容")
    if len(data) > max_bytes:
        raise ValueError("图像超过 20MB")
    return data


def _command_image_bytes(command):
    for item in command.dataItems or []:
        if isinstance(item, FileDataItem) and item.bytes:
            return base64.b64decode(item.bytes), "aip-image"
        if isinstance(item, TextDataItem) and item.text:
            text = item.text.strip()
            if text.startswith(("http://", "https://")):
                return _fetch_image_url(text), "aip-url"
            # 平台可能把整段对话当入参：在全文里找第一个图片链接
            found = re.search(r"https?://[^\s\"'<>）)，。]+", text)
            if found:
                return _fetch_image_url(found.group(0).rstrip(".,;")), "aip-url"
            # 同理，支持文本里内嵌 data URI
            data_uri = re.search(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+", text)
            if data_uri:
                body = data_uri.group(0).split(",", 1)[1]
                return base64.b64decode(re.sub(r"\s+", "", body)), "aip-datauri"
            if text.startswith("data:") and "," in text:
                text = text.split(",", 1)[1]
            try:
                return base64.b64decode(text), "aip-text"
            except Exception:  # noqa: BLE001
                candidate = Path(text)
                if candidate.is_file():
                    return candidate.read_bytes(), candidate.name
    raise ValueError("no image found in task command")


async def _execute_pipeline(command: TaskCommand, raw: bytes, filename: str) -> TaskResult:
    try:
        result = await _pipeline_core(
            raw, filename, 640, 0.25, True, 0, "letterbox", 12, True,
            "aip-%s" % uuid.uuid4().hex[:8],
        )
    except Exception as exc:  # noqa: BLE001
        detail = traceback.format_exc()
        failed = TaskManager.create_task(
            command,
            initial_state=TaskState.Failed,
            data_items=[TextDataItem(text="编排失败：%s: %s\n%s" % (type(exc).__name__, exc, detail[-1200:]))],
        )
        failed.senderId = LEADER_LOCAL_AIC
        return failed
    done = TaskManager.create_task(command, initial_state=TaskState.AwaitingCompletion)
    TaskManager.set_products(
        done.taskId,
        [
            Product(
                id="product-%s" % done.taskId,
                name="chip-inspection",
                dataItems=[StructuredDataItem(data=result)],
            )
        ],
    )
    output = TaskManager.get_task(done.taskId) or done
    output.senderId = LEADER_LOCAL_AIC
    return output


def _awaiting_input(command: TaskCommand) -> TaskResult:
    pending = TaskManager.create_task(
        command,
        initial_state=TaskState.AwaitingInput,
        data_items=[
            TextDataItem(
                text=(
                    "请提供芯片封装图像，支持三种方式：\n"
                    "1) 直接回复图片链接（http:// 或 https://）\n"
                    "2) FileDataItem(bytes=base64)\n"
                    "3) TextDataItem(base64 或本地绝对路径)"
                )
            )
        ],
    )
    pending.senderId = LEADER_LOCAL_AIC
    return pending


async def _aip_on_start(command: TaskCommand, task: TaskResult | None) -> TaskResult:
    if task:
        # 已存在任务：本次若带了图像就继续跑，否则返回当前快照
        try:
            raw, filename = _command_image_bytes(command)
        except Exception:  # noqa: BLE001
            task.senderId = LEADER_LOCAL_AIC
            return task
        return await _execute_pipeline(command, raw, filename)
    try:
        raw, filename = _command_image_bytes(command)
    except Exception:  # noqa: BLE001
        return _awaiting_input(command)
    return await _execute_pipeline(command, raw, filename)


async def _aip_on_continue(command: TaskCommand, task: TaskResult) -> TaskResult:
    """平台在追问后把用户回复作为 continue 发来；这里重新尝试取图并执行。"""
    try:
        TaskManager.add_command_to_history(task.taskId, command)
    except Exception:  # noqa: BLE001
        pass
    try:
        raw, filename = _command_image_bytes(command)
    except Exception:  # noqa: BLE001
        task.senderId = LEADER_LOCAL_AIC
        return task
    return await _execute_pipeline(command, raw, filename)


_aip_handlers = CommandHandlers(
    on_start=_aip_on_start,
    on_get=DefaultHandlers.get,
    on_cancel=DefaultHandlers.cancel,
    on_complete=DefaultHandlers.complete,
    on_continue=_aip_on_continue,
)
add_aip_rpc_router(
    app,
    "/rpc",
    _aip_handlers,
    local_aic=LEADER_LOCAL_AIC,
    identity_binding_enabled=IDENTITY_BINDING,
)


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_ROOT / "index.html", media_type="text/html; charset=utf-8")
