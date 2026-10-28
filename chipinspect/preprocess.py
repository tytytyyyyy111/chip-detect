# Image preprocessing: multi-format import, auto-crop, size fitting.
# Shared by the ChipInspect web system and the ACPs preprocess Partner.
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = 40_000_000
NATIVE_FORMATS = {
    "JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF",
    "ICO", "MPO", "PPM", "PNM", "TGA", "PCX",
}
FIT_MODES = {"letterbox", "stretch", "crop"}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class PreprocessError(ValueError):
    pass


@dataclass
class PreprocessResult:
    image: Image.Image
    source_format: str
    original_size: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None
    output_size: tuple[int, int]
    steps: list[str]


def decode_any(raw: bytes) -> tuple[Image.Image, str]:
    # Decode any supported file into an RGB image (EXIF fixed, first frame).
    if not raw:
        raise PreprocessError("empty file")
    if raw[:5] == b"%PDF-":
        return _decode_pdf(raw)
    try:
        with Image.open(io.BytesIO(raw)) as src:
            fmt = (src.format or "UNKNOWN").upper()
            if src.width * src.height > MAX_IMAGE_PIXELS:
                raise PreprocessError("image resolution exceeds safety limit")
            if getattr(src, "n_frames", 1) > 1:
                src.seek(0)
            img = ImageOps.exif_transpose(src)
            return img.convert("RGB"), fmt
    except PreprocessError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise PreprocessError(
            "unsupported or corrupt image (%s)" % type(exc).__name__
        ) from exc


def _decode_pdf(raw: bytes) -> tuple[Image.Image, str]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        raise PreprocessError(
            "PDF import requires optional dependency: pypdfium2"
        ) from exc
    try:
        pdf = pdfium.PdfDocument(raw)
        page = pdf[0]
        bitmap = page.render(scale=2)
        img = bitmap.to_pil().convert("RGB")
        return img, "PDF"
    except Exception as exc:
        raise PreprocessError("failed to render PDF: %s" % type(exc).__name__) from exc


def _median_corner_color(img: Image.Image) -> tuple[int, int, int]:
    w, h = img.size
    points = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    colors = sorted(img.getpixel(p) for p in points)
    return colors[len(colors) // 2]


def auto_trim(
    img: Image.Image,
    tolerance: int = 12,
    min_keep_ratio: float = 0.05,
    pad: int = 0,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    # Trim a uniform border relative to the median corner colour.
    w, h = img.size
    if w < 4 or h < 4:
        return img, None
    bg = _median_corner_color(img)
    background = Image.new("RGB", (w, h), bg)
    diff = ImageChops.difference(img, background).convert("L")
    mask = diff.point(lambda value: 255 if value > tolerance else 0)
    bbox = mask.getbbox()
    if not bbox or bbox == (0, 0, w, h):
        return img, None
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if bw < w * min_keep_ratio or bh < h * min_keep_ratio:
        return img, None
    if pad:
        bbox = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(w, bbox[2] + pad),
            min(h, bbox[3] + pad),
        )
    return img.crop(bbox), bbox


def fit_size(
    img: Image.Image,
    target: int | None = 640,
    mode: str = "letterbox",
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    # Resize to a square target: letterbox / stretch / center-crop.
    if not target or target <= 0:
        return img
    mode = mode.lower()
    if mode not in FIT_MODES:
        raise PreprocessError("unsupported fit mode: %s" % mode)
    if mode == "stretch":
        return img.resize((target, target), Image.LANCZOS)
    if mode == "crop":
        return ImageOps.fit(img, (target, target), method=Image.LANCZOS)
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (target, target), fill)
    canvas.paste(resized, ((target - nw) // 2, (target - nh) // 2))
    return canvas


def preprocess_image(
    raw: bytes,
    *,
    auto_crop: bool = True,
    target_size: int | None = 640,
    fit_mode: str = "letterbox",
    tolerance: int = 12,
    pad: int = 0,
) -> PreprocessResult:
    img, fmt = decode_any(raw)
    original = img.size
    steps = ["decode:%s" % fmt]
    crop_box = None

    if auto_crop:
        cropped, box = auto_trim(img, tolerance=tolerance, pad=pad)
        if box is not None:
            img = cropped
            crop_box = box
            steps.append("auto_trim")

    if target_size:
        img = fit_size(img, target_size, fit_mode)
        steps.append("fit:%s->%s" % (fit_mode, target_size))

    return PreprocessResult(
        image=img,
        source_format=fmt,
        original_size=original,
        crop_box=crop_box,
        output_size=img.size,
        steps=steps,
    )


def encode_image(img: Image.Image, fmt: str = "PNG", quality: int = 92) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue(), "image/jpeg"
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue(), "image/png"

@dataclass
class Transform:
    # Maps fitted-image coordinates back to the pre-prepare image space.
    crop_box: tuple[int, int, int, int] | None
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    crop_origin_x: float
    crop_origin_y: float

    def to_input(self, xyxy: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        x1 = (x1 - self.pad_x) / self.scale_x + self.crop_origin_x
        x2 = (x2 - self.pad_x) / self.scale_x + self.crop_origin_x
        y1 = (y1 - self.pad_y) / self.scale_y + self.crop_origin_y
        y2 = (y2 - self.pad_y) / self.scale_y + self.crop_origin_y
        return (x1, y1, x2, y2)


def prepare_image(
    img: Image.Image,
    *,
    auto_crop: bool = True,
    target_size: int | None = None,
    fit_mode: str = "letterbox",
    tolerance: int = 12,
    pad: int = 0,
) -> tuple[Image.Image, Transform, tuple[int, int, int, int] | None]:
    # Auto-crop then fit, returning a transform for coordinate mapping.
    crop_box = None
    if auto_crop:
        cropped, box = auto_trim(img, tolerance=tolerance, pad=pad)
        if box is not None:
            img = cropped
            crop_box = box

    sw, sh = img.size
    if not target_size or target_size <= 0:
        return img, Transform(crop_box, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0), crop_box

    mode = fit_mode.lower()
    if mode not in FIT_MODES:
        raise PreprocessError("unsupported fit mode: %s" % mode)

    if mode == "stretch":
        sx, sy = target_size / sw, target_size / sh
        out = img.resize((target_size, target_size), Image.LANCZOS)
        return out, Transform(crop_box, sx, sy, 0.0, 0.0, 0.0, 0.0), crop_box

    if mode == "crop":
        side = min(sw, sh)
        ox = (sw - side) / 2.0
        oy = (sh - side) / 2.0
        scale = target_size / side
        out = ImageOps.fit(img, (target_size, target_size), method=Image.LANCZOS)
        return out, Transform(crop_box, scale, scale, 0.0, 0.0, ox, oy), crop_box

    scale = min(target_size / sw, target_size / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    px, py = (target_size - nw) / 2.0, (target_size - nh) / 2.0
    out = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    out.paste(img.resize((nw, nh), Image.LANCZOS), (round(px), round(py)))
    return out, Transform(crop_box, scale, scale, px, py, 0.0, 0.0), crop_box

def map_detections_to_original(detections, meta):
    # Map detections produced on the preprocessed image back to original pixels.
    original = meta.get("original_size") or [0, 0]
    width, height = float(original[0]), float(original[1])
    crop = meta.get("crop_box")
    if crop:
        src_w = float(crop[2] - crop[0])
        src_h = float(crop[3] - crop[1])
        off_x, off_y = float(crop[0]), float(crop[1])
    else:
        src_w, src_h = width, height
        off_x = off_y = 0.0

    target = meta.get("target_size")
    fit = str(meta.get("fit_mode") or "letterbox").lower()
    if target and src_w > 0 and src_h > 0:
        if fit == "stretch":
            scale_x, scale_y = target / src_w, target / src_h
            pad_x = pad_y = 0.0
            crop_x = crop_y = 0.0
        elif fit == "crop":
            side = min(src_w, src_h)
            scale_x = scale_y = target / side
            pad_x = pad_y = 0.0
            crop_x = (src_w - side) / 2.0
            crop_y = (src_h - side) / 2.0
        else:
            scale = min(target / src_w, target / src_h)
            scale_x = scale_y = scale
            pad_x = (target - src_w * scale) / 2.0
            pad_y = (target - src_h * scale) / 2.0
            crop_x = crop_y = 0.0
    else:
        scale_x = scale_y = 1.0
        pad_x = pad_y = 0.0
        crop_x = crop_y = 0.0

    mapped = []
    for det in detections or []:
        xyxy = det.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy]

        def inv_x(value):
            return (value - pad_x) / scale_x + crop_x + off_x

        def inv_y(value):
            return (value - pad_y) / scale_y + crop_y + off_y

        nx1, nx2 = sorted((inv_x(x1), inv_x(x2)))
        ny1, ny2 = sorted((inv_y(y1), inv_y(y2)))
        item = dict(det)
        item["xyxy"] = [
            round(max(0.0, min(width, nx1)), 2),
            round(max(0.0, min(height, ny1)), 2),
            round(max(0.0, min(width, nx2)), 2),
            round(max(0.0, min(height, ny2)), 2),
        ]
        mapped.append(item)
    return mapped
