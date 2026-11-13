"""Image pipeline and optics model for Samoiloff-effect prints.

A Samoiloff print hides three grayscale images in one full-color picture.
Each target image drives the density of one subtractive ink:

    red-light target   -> cyan ink    (cyan absorbs red)
    green-light target -> magenta ink (magenta absorbs green)
    blue-light target  -> yellow ink  (yellow absorbs blue)

Light a print with a single primary and only the ink that absorbs that
primary casts a shadow, so only its target shows up. All math runs in
linear light so mixing and fades behave like real light does.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DPI = 300
DEFAULT_CANVAS_INCHES = (6.0, 4.0)
DEFAULT_DENSITY = 1.5

RED = (1.0, 0.0, 0.0)
GREEN = (0.0, 1.0, 0.0)
BLUE = (0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0)

GRAY_METHODS = ("luminance", "max", "average", "channel_mix")


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelsParams:
    black_point_pct: float = 0.0  # 0..100
    white_point_pct: float = 100.0  # 0..100
    gamma: float = 1.0  # 0.1..4.0
    auto_normalize: bool = False


@dataclass(frozen=True)
class ContrastBrightnessParams:
    contrast: float = 0.0  # -100..100
    brightness: float = 0.0  # -100..100


@dataclass(frozen=True)
class ClarityParams:
    radius_px: float = 0.0  # >= 0
    amount: float = 0.0  # -100..100


@dataclass(frozen=True)
class SharpenParams:
    amount: float = 0.0  # 0..150 (percent)
    radius: float = 1.0  # 0.1..3.0
    threshold: int = 0  # 0..10


@dataclass(frozen=True)
class GrayMethodParams:
    method: str = "luminance"  # one of GRAY_METHODS
    mix_r: float = 1.0
    mix_g: float = 1.0
    mix_b: float = 1.0


@dataclass(frozen=True)
class TargetSettings:
    """Every per-image adjustment, bundled so it can be cached and passed around."""

    levels: LevelsParams = field(default_factory=LevelsParams)
    contrast_brightness: ContrastBrightnessParams = field(default_factory=ContrastBrightnessParams)
    clarity: ClarityParams = field(default_factory=ClarityParams)
    sharpen: SharpenParams = field(default_factory=SharpenParams)
    gray: GrayMethodParams = field(default_factory=GrayMethodParams)


# -----------------------------------------------------------------------------
# Color transforms
# -----------------------------------------------------------------------------


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb, dtype=np.float32)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        np.power((np.maximum(srgb, 0.04045) + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.asarray(lin, dtype=np.float32)
    return np.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * np.power(np.maximum(lin, 0.0031308), 1 / 2.4) - 0.055,
    ).astype(np.float32)


# -----------------------------------------------------------------------------
# PIL <-> numpy helpers
# -----------------------------------------------------------------------------


def open_image(file_bytes: bytes) -> Image.Image:
    """Open PNG/JPEG bytes as RGB or RGBA, honoring EXIF rotation."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))
    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    return img.convert("RGBA" if has_alpha else "RGB")


def to_numpy_float01(img: Image.Image) -> np.ndarray:
    """PIL image -> HxWxC float32 array in [0, 1] (C is 1, 3 or 4)."""
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[:, :, None] if arr.ndim == 2 else arr


def from_numpy_to_pil(arr01: np.ndarray) -> Image.Image:
    """HxW, HxWx3 or HxWx4 float array in [0, 1] -> 8-bit PIL image."""
    arr8 = (clamp01(arr01) * 255.0 + 0.5).astype(np.uint8)
    if arr8.ndim == 3 and arr8.shape[2] == 1:
        arr8 = arr8[:, :, 0]
    elif arr8.ndim == 3 and arr8.shape[2] not in (3, 4):
        arr8 = arr8[:, :, :3]
    return Image.fromarray(np.ascontiguousarray(arr8))


def linear_to_pil(lin: np.ndarray) -> Image.Image:
    """Linear-light array -> sRGB-encoded PIL image."""
    return from_numpy_to_pil(linear_to_srgb(clamp01(lin)))


def compute_canvas_pixels(width_in: float, height_in: float, dpi: int = DPI) -> tuple[int, int]:
    return (int(round(width_in * dpi)), int(round(height_in * dpi)))


def contain_center_pad(
    src: Image.Image,
    canvas_px: tuple[int, int],
    padding_mode: str = "White",
) -> Image.Image:
    """Scale `src` to fit inside the canvas and center it.

    "White" padding returns an RGB image (alpha flattened onto white).
    "Transparent" padding returns RGBA with a transparent border.
    """
    canvas_w, canvas_h = canvas_px
    src_w, src_h = src.size
    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = src.convert("RGBA").resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)

    if padding_mode == "Transparent":
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
        canvas.paste(resized, offset)
        return canvas

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(resized, offset, mask=resized.getchannel("A"))
    return canvas


# -----------------------------------------------------------------------------
# Per-image adjustments (linear light unless noted)
# -----------------------------------------------------------------------------


def apply_levels_exposure_linear(img_lin: np.ndarray, params: LevelsParams) -> np.ndarray:
    x = img_lin
    if params.auto_normalize:
        min_c = x.min(axis=(0, 1), keepdims=True)
        max_c = x.max(axis=(0, 1), keepdims=True)
        x = (x - min_c) / np.maximum(max_c - min_c, 1e-6)

    bp = float(np.clip(params.black_point_pct / 100.0, 0.0, 1.0))
    wp = max(float(np.clip(params.white_point_pct / 100.0, 0.0, 1.0)), bp + 1e-6)
    x = clamp01((x - bp) / (wp - bp))

    gamma = max(0.1, min(4.0, params.gamma))
    return clamp01(np.power(x, 1.0 / gamma))


def apply_contrast_brightness_linear(img_lin: np.ndarray, params: ContrastBrightnessParams) -> np.ndarray:
    c = np.clip(params.contrast / 100.0, -1.0, 1.0)
    b = np.clip(params.brightness / 100.0, -1.0, 1.0)
    return clamp01((img_lin - 0.5) * (1.0 + c) + 0.5 + b)


def apply_clarity(img_lin: np.ndarray, params: ClarityParams) -> np.ndarray:
    """Local contrast: add back (image - blurred) at a wide radius. Runs in sRGB."""
    if params.amount == 0 or params.radius_px <= 0:
        return img_lin
    srgb = from_numpy_to_pil(linear_to_srgb(clamp01(img_lin)))
    blurred = srgb.filter(ImageFilter.GaussianBlur(radius=float(params.radius_px)))
    base = to_numpy_float01(srgb)[:, :, :3]
    detail = base - to_numpy_float01(blurred)[:, :, :3]
    amount = np.clip(params.amount / 100.0, -1.0, 1.0)
    return srgb_to_linear(clamp01(base + amount * detail))


def apply_sharpen(img_lin: np.ndarray, params: SharpenParams) -> np.ndarray:
    """Unsharp mask via PIL (sRGB), which handles the threshold for us."""
    if params.amount <= 0 or params.radius <= 0:
        return img_lin
    srgb = from_numpy_to_pil(linear_to_srgb(clamp01(img_lin)))
    sharpened = srgb.filter(
        ImageFilter.UnsharpMask(
            radius=float(params.radius),
            percent=int(params.amount),
            threshold=int(params.threshold),
        )
    )
    return srgb_to_linear(to_numpy_float01(sharpened)[:, :, :3])


def to_grayscale(img_lin: np.ndarray, gray_params: GrayMethodParams) -> np.ndarray:
    rgb = clamp01(img_lin[:, :, :3])
    method = gray_params.method
    if method == "max":
        gray = rgb.max(axis=2)
    elif method == "average":
        gray = rgb.mean(axis=2)
    elif method == "channel_mix":
        w = np.array([gray_params.mix_r, gray_params.mix_g, gray_params.mix_b], dtype=np.float32)
        w = np.full(3, 1 / 3, dtype=np.float32) if w.sum() == 0 else w / w.sum()
        gray = rgb @ w
    else:  # Rec.709 luminance in linear light
        gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return clamp01(gray).astype(np.float32)


def process_target(
    file_bytes: bytes,
    canvas_px: tuple[int, int],
    padding_mode: str = "White",
    settings: TargetSettings = TargetSettings(),
) -> tuple[np.ndarray, Image.Image]:
    """Fit, adjust and grayscale one target image.

    Returns (gray_lin, padded): the HxW linear-light grayscale that this
    target should show under its light, and the padded source image (kept
    for its alpha channel when padding is transparent).
    """
    padded = contain_center_pad(open_image(file_bytes), canvas_px, padding_mode)
    lin = srgb_to_linear(to_numpy_float01(padded)[:, :, :3])
    lin = apply_levels_exposure_linear(lin, settings.levels)
    lin = apply_contrast_brightness_linear(lin, settings.contrast_brightness)
    lin = apply_clarity(lin, settings.clarity)
    lin = apply_sharpen(lin, settings.sharpen)
    return to_grayscale(lin, settings.gray), padded


# -----------------------------------------------------------------------------
# Optics: inks, the print, and what it looks like under a given light
# -----------------------------------------------------------------------------


def apply_cmy_density(
    c: np.ndarray, m: np.ndarray, y: np.ndarray, density: float = DEFAULT_DENSITY
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ink response curve: x' = 1 - (1 - x)^k. Larger k means heavier ink."""
    k = float(max(0.1, min(5.0, density)))

    def densify(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(clamp01(1.0 - x), k)

    return densify(c), densify(m), densify(y)


def ink_plates(r_gray: np.ndarray, g_gray: np.ndarray, b_gray: np.ndarray) -> np.ndarray:
    """Target grays -> HxWx3 CMY ink coverage in [0, 1] (1 = full ink)."""
    return clamp01(1.0 - np.stack([r_gray, g_gray, b_gray], axis=2)).astype(np.float32)


def print_transmittance(cmy: np.ndarray, density: float = DEFAULT_DENSITY) -> np.ndarray:
    """CMY coverage -> per-channel transmittance of the print (linear RGB).

    Cyan only absorbs red, magenta only green, yellow only blue, so each
    output channel depends on exactly one ink.
    """
    c, m, y = apply_cmy_density(cmy[:, :, 0], cmy[:, :, 1], cmy[:, :, 2], density)
    return clamp01(1.0 - np.stack([c, m, y], axis=2)).astype(np.float32)


def under_light(transmittance: np.ndarray, light: Sequence[float] = WHITE) -> np.ndarray:
    """Linear RGB seen through (or reflected off) the print under `light`."""
    return clamp01(transmittance * np.asarray(light, dtype=np.float32).reshape(1, 1, 3))


def build_white_light_composite_srgb(
    c_lin: np.ndarray, m_lin: np.ndarray, y_lin: np.ndarray, density: float = DEFAULT_DENSITY
) -> Image.Image:
    """The printable image: what the print looks like under white light."""
    cmy = np.stack([clamp01(c_lin), clamp01(m_lin), clamp01(y_lin)], axis=2)
    return linear_to_pil(print_transmittance(cmy, density))


def apply_preview_gamma_srgb(pil_img: Image.Image, gamma: float) -> Image.Image:
    """Display-only gamma tweak; never touches exported data."""
    gamma = float(max(0.5, min(2.5, gamma)))
    lin = srgb_to_linear(to_numpy_float01(pil_img)[:, :, :3])
    return linear_to_pil(np.power(clamp01(lin), 1.0 / gamma))


def apply_preview_brightness_srgb(pil_img: Image.Image, trim: float) -> Image.Image:
    """Display-only brightness trim (e.g. +0.1 = 10% brighter)."""
    if abs(trim) < 1e-6:
        return pil_img
    return from_numpy_to_pil(to_numpy_float01(pil_img)[:, :, :3] * (1.0 + trim))


def tint_grayscale_to_color(pil_img: Image.Image, color_rgb: Sequence[float]) -> Image.Image:
    """Multiply a grayscale image by a color so white maps to that color."""
    arr = to_numpy_float01(pil_img.convert("RGB"))
    color = np.asarray(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    return from_numpy_to_pil(arr * color)


def plate_image(coverage: np.ndarray, ink_rgb: Sequence[float]) -> Image.Image:
    """Render one ink plate as that ink printed on white paper."""
    ink = np.asarray(ink_rgb, dtype=np.float32).reshape(1, 1, 3)
    return linear_to_pil(1.0 - coverage[:, :, None] * (1.0 - ink))


# -----------------------------------------------------------------------------
# Animation
# -----------------------------------------------------------------------------


def crossfade_gif(
    stills: Iterable[Image.Image],
    hold_s: float = 1.5,
    fade_s: float = 0.8,
    fps: int = 24,
    max_side: int = 900,
) -> tuple[bytes, float]:
    """Loop through `stills`, holding each and fading (in linear light) to the next.

    Frames are downscaled so the long side is at most `max_side` pixels; a
    300 DPI canvas makes for a huge, slow GIF otherwise.
    Returns (gif_bytes, total_seconds).
    """

    def shrink(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        scale = max_side / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        return img

    lin = [srgb_to_linear(to_numpy_float01(shrink(s))) for s in stills]
    if not lin:
        raise ValueError("crossfade_gif needs at least one still")
    fps = max(1, int(fps))
    hold_n = max(1, int(round(hold_s * fps)))
    fade_n = max(1, int(round(fade_s * fps))) if len(lin) > 1 else 0

    frames: list[Image.Image] = []
    for i, cur in enumerate(lin):
        held = linear_to_pil(cur)
        frames.extend([held] * hold_n)
        nxt = lin[(i + 1) % len(lin)]
        for step in range(1, fade_n + 1):
            t = step / fade_n
            frames.append(linear_to_pil((1.0 - t) * cur + t * nxt))

    frame_ms = int(round(1000 / fps))
    bio = io.BytesIO()
    frames[0].save(
        bio,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    return bio.getvalue(), len(frames) * frame_ms / 1000.0


def generate_colored_cycle_gif(
    transmittance: np.ndarray,
    hold_s: float = 1.5,
    fade_s: float = 0.8,
    fps: int = 24,
) -> tuple[bytes, float]:
    """Simulated light show: the print under red, then green, then blue light.

    Light adds linearly, so fading between the three stills in linear light
    is exactly what dimming one lamp while raising the next would show.
    """
    stills = [linear_to_pil(under_light(transmittance, light)) for light in (RED, GREEN, BLUE)]
    return crossfade_gif(stills, hold_s, fade_s, fps)


def generate_cycle_gif(
    r_gray: np.ndarray,
    g_gray: np.ndarray,
    b_gray: np.ndarray,
    hold_s: float = 1.5,
    fade_s: float = 0.8,
    fps: int = 24,
) -> tuple[bytes, float]:
    """Grayscale cycle through the three processed targets."""
    stills = [linear_to_pil(g) for g in (r_gray, g_gray, b_gray)]
    return crossfade_gif(stills, hold_s, fade_s, fps)


# -----------------------------------------------------------------------------
# One-shot composition
# -----------------------------------------------------------------------------


@dataclass
class Composition:
    """Everything derived from three processed targets."""

    grays: tuple[np.ndarray, np.ndarray, np.ndarray]
    cmy: np.ndarray
    transmittance: np.ndarray
    alpha: np.ndarray | None = None

    def white_light(self) -> Image.Image:
        img = linear_to_pil(self.transmittance)
        if self.alpha is not None:
            img.putalpha(Image.fromarray(self.alpha))
        return img

    def lit_by(self, light: Sequence[float]) -> Image.Image:
        return linear_to_pil(under_light(self.transmittance, light))

    def plates(self) -> dict[str, Image.Image]:
        inks = {"cyan": (0.0, 1.0, 1.0), "magenta": (1.0, 0.0, 1.0), "yellow": (1.0, 1.0, 0.0)}
        return {name: plate_image(self.cmy[:, :, i], ink) for i, (name, ink) in enumerate(inks.items())}


def compose(
    r_gray: np.ndarray,
    g_gray: np.ndarray,
    b_gray: np.ndarray,
    density: float = DEFAULT_DENSITY,
    padded: Sequence[Image.Image] = (),
) -> Composition:
    """Combine three target grays into a print. Shapes must match."""
    if not (r_gray.shape == g_gray.shape == b_gray.shape):
        raise ValueError(f"target shapes differ: {r_gray.shape}, {g_gray.shape}, {b_gray.shape}")
    cmy = ink_plates(r_gray, g_gray, b_gray)
    alpha = None
    for img in padded:
        if img.mode == "RGBA":
            a = np.asarray(img.getchannel("A"))
            alpha = a if alpha is None else np.maximum(alpha, a)
    return Composition((r_gray, g_gray, b_gray), cmy, print_transmittance(cmy, density), alpha)


def export_bundle(
    comp: Composition,
    hold_s: float = 1.5,
    fade_s: float = 0.8,
    fps: int = 24,
    include_gifs: bool = True,
) -> dict[str, bytes]:
    """Render every export file as {filename: bytes}."""

    def png(img: Image.Image) -> bytes:
        bio = io.BytesIO()
        img.save(bio, format="PNG", optimize=True)
        return bio.getvalue()

    files = {"print_white_light.png": png(comp.white_light())}
    for name, plate in comp.plates().items():
        files[f"plate_{name}.png"] = png(plate)
    for name, light in (("red", RED), ("green", GREEN), ("blue", BLUE)):
        files[f"under_{name}_light.png"] = png(comp.lit_by(light))
    if include_gifs:
        files["rgb_cycle_colored.gif"] = generate_colored_cycle_gif(comp.transmittance, hold_s, fade_s, fps)[0]
        files["rgb_cycle.gif"] = generate_cycle_gif(*comp.grays, hold_s, fade_s, fps)[0]
    return files
