import io
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
import streamlit as st


# -----------------------------
# Constants and Types
# -----------------------------
DPI = 300
DEFAULT_CANVAS_INCHES = (6.0, 4.0)


@dataclass(frozen=True)
class LevelsParams:
    black_point_pct: float  # 0..100
    white_point_pct: float  # 0..100
    gamma: float            # 0.1..4.0
    auto_normalize: bool


@dataclass(frozen=True)
class ContrastBrightnessParams:
    contrast: float   # -100..100
    brightness: float # -100..100


@dataclass(frozen=True)
class ClarityParams:
    radius_px: float  # >=0
    amount: float     # -100..100


@dataclass(frozen=True)
class SharpenParams:
    amount: float   # 0..150
    radius: float   # 0.1..3.0
    threshold: int  # 0..10


@dataclass(frozen=True)
class GrayMethodParams:
    method: str  # luminance|max|average|channel_mix
    mix_r: float
    mix_g: float
    mix_b: float


# -----------------------------
# Utilities: Color transforms
# -----------------------------

def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    # srgb: float array in [0,1]
    threshold = 0.04045
    below = srgb <= threshold
    above = ~below
    out = np.empty_like(srgb, dtype=np.float32)
    out[below] = srgb[below] / 12.92
    out[above] = ((srgb[above] + 0.055) / 1.055) ** 2.4
    return out


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    threshold = 0.0031308
    below = lin <= threshold
    above = ~below
    out = np.empty_like(lin, dtype=np.float32)
    out[below] = lin[below] * 12.92
    out[above] = 1.055 * (np.power(lin[above], 1/2.4)) - 0.055
    return out


# -----------------------------
# Utilities: Imaging helpers
# -----------------------------

def pil_open_rgb(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    # Convert to RGB (assume sRGB)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return img


def to_numpy_float01(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] == 4:
        rgb = arr[:, :, :3] / 255.0
        a = arr[:, :, 3:4] / 255.0
        return np.concatenate([rgb, a], axis=2)
    return arr / 255.0


def from_numpy_to_pil(arr01: np.ndarray, mode_hint: Optional[str] = None) -> Image.Image:
    arr01 = clamp01(arr01)
    if arr01.shape[2] == 4:
        arr8 = (arr01 * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr8, mode="RGBA")
    arr8 = (arr01[:, :, :3] * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr8, mode=mode_hint or "RGB")


def compute_canvas_pixels(width_in: float, height_in: float, dpi: int = DPI) -> Tuple[int, int]:
    return (int(round(width_in * dpi)), int(round(height_in * dpi)))


def contain_center_pad(
    src: Image.Image,
    canvas_px: Tuple[int, int],
    padding_mode: str = "White",
) -> Image.Image:
    canvas_w, canvas_h = canvas_px
    src_w, src_h = src.size
    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = src.resize((new_w, new_h), Image.LANCZOS)

    if padding_mode == "Transparent" and src.mode == "RGBA":
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
    elif padding_mode == "Transparent":
        # Ensure alpha channel exists if transparency requested
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))
        if resized.mode != "RGBA":
            resized = resized.convert("RGBA")
    else:
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
        if resized.mode == "RGBA":
            # Flatten against white
            bg = Image.new("RGB", resized.size, (255, 255, 255))
            bg.paste(resized, mask=resized.split()[-1])
            resized = bg

    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2
    if canvas.mode == "RGBA" and resized.mode != "RGBA":
        resized = resized.convert("RGBA")
    canvas.paste(resized, (offset_x, offset_y), mask=resized.split()[-1] if resized.mode == "RGBA" else None)
    return canvas


# -----------------------------
# Image adjustments (mostly in linear domain)
# -----------------------------

def apply_levels_exposure_linear(img_lin: np.ndarray, params: LevelsParams) -> np.ndarray:
    # img_lin: HxWx3 in [0,1]
    x = img_lin
    if params.auto_normalize:
        # Per-channel auto normalize to [0,1]
        min_c = x.min(axis=(0, 1), keepdims=True)
        max_c = x.max(axis=(0, 1), keepdims=True)
        span = np.maximum(max_c - min_c, 1e-6)
        x = (x - min_c) / span

    bp = np.clip(params.black_point_pct / 100.0, 0.0, 1.0)
    wp = np.clip(params.white_point_pct / 100.0, 0.0, 1.0)
    wp = max(wp, bp + 1e-6)

    # Normalize using black/white points
    x = (x - bp) / (wp - bp)
    x = clamp01(x)

    # Gamma (mid-tones): apply 1/gamma power in linear domain
    gamma = max(0.1, min(4.0, params.gamma))
    x = np.power(x, 1.0 / gamma)
    return clamp01(x)


def apply_contrast_brightness_linear(img_lin: np.ndarray, params: ContrastBrightnessParams) -> np.ndarray:
    # Map sliders to [-1,1]
    c = np.clip(params.contrast / 100.0, -1.0, 1.0)
    b = np.clip(params.brightness / 100.0, -1.0, 1.0)
    # Adjust around mid-gray 0.5 in linear light
    x = img_lin
    x = (x - 0.5) * (1.0 + c) + 0.5
    x = x + b
    return clamp01(x)


def apply_clarity(img_lin: np.ndarray, params: ClarityParams) -> np.ndarray:
    if params.amount == 0 or params.radius_px <= 0:
        return img_lin
    # Approximate local contrast via unsharp-like operation in sRGB for practicality
    # Convert to sRGB 8-bit, blur, add detail scaled by amount
    srgb = linear_to_srgb(clamp01(img_lin))
    pil = from_numpy_to_pil(srgb)
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=float(params.radius_px)))
    pil_arr = to_numpy_float01(pil)
    blurred_arr = to_numpy_float01(blurred)
    amount = np.clip(params.amount / 100.0, -1.0, 1.0)
    enhanced = pil_arr[:, :, :3] + amount * (pil_arr[:, :, :3] - blurred_arr[:, :, :3])
    enhanced = clamp01(enhanced)
    # Back to linear
    return srgb_to_linear(enhanced)


def apply_sharpen(img_lin: np.ndarray, params: SharpenParams) -> np.ndarray:
    if params.amount <= 0 or params.radius <= 0:
        return img_lin
    # Use PIL UnsharpMask in sRGB for speed and threshold handling
    srgb = linear_to_srgb(clamp01(img_lin))
    pil = from_numpy_to_pil(srgb)
    # PIL expects percent 0..500; our amount 0..150 → use directly
    sharpened = pil.filter(ImageFilter.UnsharpMask(radius=float(params.radius), percent=int(params.amount), threshold=int(params.threshold)))
    arr = to_numpy_float01(sharpened)[:, :, :3]
    return srgb_to_linear(arr)


# -----------------------------
# Grayscale conversions
# -----------------------------

def to_grayscale(img_lin: np.ndarray, gray_params: GrayMethodParams) -> np.ndarray:
    rgb = clamp01(img_lin[:, :, :3])
    method = gray_params.method
    if method == "luminance":
        # Rec.709 luminance in linear light
        gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    elif method == "max":
        gray = np.max(rgb, axis=2)
    elif method == "average":
        gray = np.mean(rgb, axis=2)
    elif method == "channel_mix":
        w = np.array([gray_params.mix_r, gray_params.mix_g, gray_params.mix_b], dtype=np.float32)
        if w.sum() == 0:
            w = np.array([1/3, 1/3, 1/3], dtype=np.float32)
        else:
            w = w / np.sum(w)
        gray = rgb @ w
    else:
        gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return clamp01(gray)


# -----------------------------
# Processing pipeline for one target image
# -----------------------------

def process_target(
    file_bytes: bytes,
    canvas_px: Tuple[int, int],
    padding_mode: str,
    levels: LevelsParams,
    cont_bright: ContrastBrightnessParams,
    clarity: ClarityParams,
    sharpen: SharpenParams,
    gray_params: GrayMethodParams,
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    # Returns: (channel_gray_lin[H,W], processed_rgb_preview_srgb PIL, padded_original PIL)
    src = pil_open_rgb(file_bytes)
    padded = contain_center_pad(src, canvas_px, padding_mode)

    # Extract RGB (drop alpha for processing)
    padded_arr = to_numpy_float01(padded)
    if padded_arr.shape[2] == 4:
        rgb01 = padded_arr[:, :, :3]
    else:
        rgb01 = padded_arr[:, :, :3]

    # Linearize
    lin = srgb_to_linear(rgb01)

    # Per-image controls
    lin = apply_levels_exposure_linear(lin, levels)
    lin = apply_contrast_brightness_linear(lin, cont_bright)
    lin = apply_clarity(lin, clarity)
    lin = apply_sharpen(lin, sharpen)

    # → Grayscale
    gray_lin = to_grayscale(lin, gray_params)  # HxW in [0,1]

    # Build preview RGB of 1 - channel (for R/G/B pane preview as grayscale)
    inv_lin = 1.0 - gray_lin
    inv_rgb_lin = np.stack([inv_lin, inv_lin, inv_lin], axis=2)

    preview_srgb = linear_to_srgb(inv_rgb_lin)
    preview_pil = from_numpy_to_pil(preview_srgb)

    return gray_lin, preview_pil, padded


@st.cache_data(show_spinner=False)
def process_target_cached(
    file_bytes: bytes,
    canvas_px: Tuple[int, int],
    padding_mode: str,
    levels_tuple: Tuple[float, float, float, bool],
    cb_tuple: Tuple[float, float],
    clarity_tuple: Tuple[float, float],
    sharpen_tuple: Tuple[float, float, int],
    gray_tuple: Tuple[str, float, float, float],
) -> Tuple[np.ndarray, Image.Image, Image.Image]:
    levels = LevelsParams(*levels_tuple)
    cb = ContrastBrightnessParams(*cb_tuple)
    clarity = ClarityParams(*clarity_tuple)
    sharpen = SharpenParams(*sharpen_tuple)
    gray_params = GrayMethodParams(*gray_tuple)
    return process_target(
        file_bytes,
        canvas_px,
        padding_mode,
        levels,
        cb,
        clarity,
        sharpen,
        gray_params,
    )


# -----------------------------
# CMY mapping and composites
# -----------------------------

def build_white_light_composite_srgb(c_lin: np.ndarray, m_lin: np.ndarray, y_lin: np.ndarray, density: float = 1.5) -> Image.Image:
    # Continuous subtractive mixing under white light with density curve
    c_d, m_d, y_d = apply_cmy_density(clamp01(c_lin), clamp01(m_lin), clamp01(y_lin), density)
    R = clamp01(1.0 - c_d)
    G = clamp01(1.0 - m_d)
    B = clamp01(1.0 - y_d)
    rgb_lin = np.stack([R, G, B], axis=2)
    return from_numpy_to_pil(linear_to_srgb(rgb_lin), mode_hint="RGB")


def apply_preview_brightness_srgb(pil_img: Image.Image, trim: float) -> Image.Image:
    # trim: -0.2..0.2 (±20%) applied in sRGB
    if abs(trim) < 1e-6:
        return pil_img
    arr = to_numpy_float01(pil_img)[:, :, :3]
    arr = arr * (1.0 + trim)
    return from_numpy_to_pil(arr, mode_hint="RGB")


def apply_preview_gamma_srgb(pil_img: Image.Image, gamma: float) -> Image.Image:
    # Adjust display gamma without affecting export data. Operate via linear domain.
    gamma = float(max(0.8, min(2.2, gamma)))
    lin = srgb_to_linear(to_numpy_float01(pil_img)[:, :, :3])
    adj = np.power(clamp01(lin), 1.0 / gamma)
    return from_numpy_to_pil(linear_to_srgb(adj), mode_hint="RGB")


def tint_grayscale_to_color(pil_img: Image.Image, color_rgb: Tuple[float, float, float]) -> Image.Image:
    # Colorize a grayscale RGB image so that white maps to the given color
    arr = to_numpy_float01(pil_img)[:, :, :3]
    color = np.array(color_rgb, dtype=np.float32).reshape((1, 1, 3))
    tinted = clamp01(arr * color)
    return from_numpy_to_pil(tinted, mode_hint="RGB")


# -----------------------------
# CMY density helper
# -----------------------------

def apply_cmy_density(c: np.ndarray, m: np.ndarray, y: np.ndarray, density: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # c', m', y' = 1 - (1 - x)^k in linear light
    k = float(max(0.1, min(5.0, density)))
    def densify(x: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(clamp01(1.0 - x), k)
    return densify(c), densify(m), densify(y)


# -----------------------------
# GIF generation
# -----------------------------

def generate_cycle_gif(
    r_preview: Image.Image,
    g_preview: Image.Image,
    b_preview: Image.Image,
    hold_s: float,
    fade_s: float,
    fps: int,
) -> Tuple[bytes, float]:
    import imageio

    def pil_to_linear_rgb(img: Image.Image) -> np.ndarray:
        arr = to_numpy_float01(img)[:, :, :3]
        return srgb_to_linear(arr)

    frames = []
    durations = []

    def add_hold(img: Image.Image, seconds: float):
        num = max(1, int(round(seconds * fps)))
        for _ in range(num):
            frames.append(np.asarray(img.convert("P")))  # palette will be recalculated later
            durations.append(1.0 / fps)

    def add_fade(img_from: Image.Image, img_to: Image.Image, seconds: float):
        steps = max(1, int(round(seconds * fps)))
        lf = pil_to_linear_rgb(img_from)
        lt = pil_to_linear_rgb(img_to)
        for s in range(1, steps + 1):
            t = s / steps
            lin = (1.0 - t) * lf + t * lt
            srgb = linear_to_srgb(lin)
            frame = from_numpy_to_pil(srgb)
            frames.append(np.asarray(frame.convert("P")))
            durations.append(1.0 / fps)

    add_hold(r_preview, hold_s)
    add_fade(r_preview, g_preview, fade_s)
    add_hold(g_preview, hold_s)
    add_fade(g_preview, b_preview, fade_s)
    add_hold(b_preview, hold_s)
    add_fade(b_preview, r_preview, fade_s)

    # Save to bytes
    bio = io.BytesIO()
    imageio.mimsave(bio, frames, format="GIF", duration=durations, loop=0)
    bio.seek(0)
    return bio.getvalue(), sum(durations)


def generate_colored_cycle_gif(
    r_preview: Image.Image,
    g_preview: Image.Image,
    b_preview: Image.Image,
    hold_s: float,
    fade_s: float,
    fps: int,
) -> Tuple[bytes, float]:
    import imageio

    # Our previews are inverted (1 − gray). We need pass-through under R/G/B:
    # passR = R_gray, passG = G_gray, passB = B_gray.
    # So compute pass-through by inverting the preview again in linear.
    passR_lin = 1.0 - srgb_to_linear(to_numpy_float01(r_preview)[:, :, 0])
    passG_lin = 1.0 - srgb_to_linear(to_numpy_float01(g_preview)[:, :, 0])
    passB_lin = 1.0 - srgb_to_linear(to_numpy_float01(b_preview)[:, :, 0])

    # Apply same density used in white-light composite so the GIF matches
    density = 1.5
    C_lin = clamp01(1.0 - passR_lin)
    M_lin = clamp01(1.0 - passG_lin)
    Y_lin = clamp01(1.0 - passB_lin)
    C_d, M_d, Y_d = apply_cmy_density(C_lin, M_lin, Y_lin, density)
    invC_d = clamp01(1.0 - C_d)
    invM_d = clamp01(1.0 - M_d)
    invY_d = clamp01(1.0 - Y_d)

    h, w = invC_d.shape

    def frame_from_backlight(rw: float, gw: float, bw: float) -> Image.Image:
        rgb_lin = np.zeros((h, w, 3), dtype=np.float32)
        rgb_lin[:, :, 0] = clamp01(invC_d * float(rw))
        rgb_lin[:, :, 1] = clamp01(invM_d * float(gw))
        rgb_lin[:, :, 2] = clamp01(invY_d * float(bw))
        return from_numpy_to_pil(linear_to_srgb(rgb_lin), mode_hint="RGB")

    frames = []
    durations = []

    def add_hold(backlight: Tuple[float, float, float], seconds: float):
        num = max(1, int(round(seconds * fps)))
        img = frame_from_backlight(*backlight)
        for _ in range(num):
            frames.append(np.asarray(img.convert("P")))
            durations.append(1.0 / fps)

    def add_fade(bl_from: Tuple[float, float, float], bl_to: Tuple[float, float, float], seconds: float):
        steps = max(1, int(round(seconds * fps)))
        r0, g0, b0 = bl_from
        r1, g1, b1 = bl_to
        for s in range(1, steps + 1):
            t = s / steps
            rw = (1.0 - t) * r0 + t * r1
            gw = (1.0 - t) * g0 + t * g1
            bw = (1.0 - t) * b0 + t * b1
            img = frame_from_backlight(rw, gw, bw)
            frames.append(np.asarray(img.convert("P")))
            durations.append(1.0 / fps)

    # Cycle: R hold → fade to G → G hold → fade to B → B hold → fade to R
    add_hold((1.0, 0.0, 0.0), hold_s)
    add_fade((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), fade_s)
    add_hold((0.0, 1.0, 0.0), hold_s)
    add_fade((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), fade_s)
    add_hold((0.0, 0.0, 1.0), hold_s)
    add_fade((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), fade_s)

    bio = io.BytesIO()
    imageio.mimsave(bio, frames, format="GIF", duration=durations, loop=0)
    bio.seek(0)
    return bio.getvalue(), sum(durations)


@st.cache_data(show_spinner=False)
def generate_cycle_gif_cached(
    r_png_bytes: bytes,
    g_png_bytes: bytes,
    b_png_bytes: bytes,
    hold_s: float,
    fade_s: float,
    fps: int,
) -> Tuple[bytes, float]:
    r_img = Image.open(io.BytesIO(r_png_bytes)).convert("RGB")
    g_img = Image.open(io.BytesIO(g_png_bytes)).convert("RGB")
    b_img = Image.open(io.BytesIO(b_png_bytes)).convert("RGB")
    return generate_cycle_gif(r_img, g_img, b_img, hold_s, fade_s, fps)


# -----------------------------
# UI Components
# -----------------------------

def sidebar_controls() -> Dict:
    st.sidebar.title("SAMOILOFF Effect Composer")
    st.sidebar.caption("Canvas at 300 DPI (fixed)")

    colw, colh = st.sidebar.columns(2)
    width_in = colw.number_input("Canvas Width (in)", min_value=0.5, max_value=40.0, value=DEFAULT_CANVAS_INCHES[0], step=0.25)
    height_in = colh.number_input("Canvas Height (in)", min_value=0.5, max_value=40.0, value=DEFAULT_CANVAS_INCHES[1], step=0.25)

    canvas_px = compute_canvas_pixels(width_in, height_in, DPI)
    st.sidebar.write(f"Pixels: {canvas_px[0]} × {canvas_px[1]} @ {DPI} DPI")

    padding_mode = st.sidebar.selectbox("Padding Mode (PNG only)", ["White", "Transparent"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Output Settings")
    output_dir = st.sidebar.text_input("Output Directory", value=".", help="Directory to save exported files")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Uploads")
    up_r = st.sidebar.file_uploader("R-target (PNG/JPEG)", type=["png", "jpg", "jpeg"], key="upload_r")
    up_g = st.sidebar.file_uploader("G-target (PNG/JPEG)", type=["png", "jpg", "jpeg"], key="upload_g")
    up_b = st.sidebar.file_uploader("B-target (PNG/JPEG)", type=["png", "jpg", "jpeg"], key="upload_b")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Controls")
    gamma_preview = st.sidebar.slider("White-light Gamma", min_value=0.8, max_value=2.2, value=2.2, step=0.05)
    preview_brightness_pct = st.sidebar.slider("On-screen Preview Brightness (\u00B120%)", min_value=-20, max_value=20, value=0, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("GIF Settings")
    hold_s = st.sidebar.number_input("Hold (s)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
    fade_s = st.sidebar.number_input("Fade (s)", min_value=0.1, max_value=10.0, value=0.8, step=0.1)
    fps = st.sidebar.number_input("FPS", min_value=1, max_value=60, value=24, step=1)

    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    reset_clicked = col1.button("Reset All")
    export_clicked = col2.button("Export All")

    return {
        "canvas_px": canvas_px,
        "padding_mode": padding_mode,
        "output_dir": output_dir,
        "upload_r": up_r,
        "upload_g": up_g,
        "upload_b": up_b,
        "gamma_preview": gamma_preview,
        "preview_brightness": preview_brightness_pct / 100.0,
        "gif_hold": hold_s,
        "gif_fade": fade_s,
        "gif_fps": int(fps),
        "reset_clicked": reset_clicked,
        "export_clicked": export_clicked,
    }


def per_image_controls(prefix: str) -> Tuple[LevelsParams, ContrastBrightnessParams, ClarityParams, SharpenParams, GrayMethodParams]:
    st.subheader(f"{prefix} Controls")

    with st.expander("Levels / Exposure", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        bp = c1.slider("Black %", 0, 100, 0, 1, key=f"{prefix}_bp")
        wp = c2.slider("White %", 0, 100, 100, 1, key=f"{prefix}_wp")
        gm = c3.slider("Gamma", 0.10, 4.0, 1.0, 0.01, key=f"{prefix}_gamma")
        auto = c4.checkbox("Auto-normalize", value=False, key=f"{prefix}_auto")
        levels = LevelsParams(bp, wp, gm, auto)

    with st.expander("Contrast & Brightness", expanded=True):
        c1, c2 = st.columns(2)
        ct = c1.slider("Contrast", -100, 100, 0, 1, key=f"{prefix}_contrast")
        br = c2.slider("Brightness", -100, 100, 0, 1, key=f"{prefix}_brightness")
        cb = ContrastBrightnessParams(ct, br)

    with st.expander("Clarity (Local Contrast)"):
        c1, c2 = st.columns(2)
        cr = c1.number_input("Radius (px)", min_value=0.0, max_value=200.0, value=0.0, step=0.5, key=f"{prefix}_clarity_radius")
        ca = c2.slider("Amount", -100, 100, 0, 1, key=f"{prefix}_clarity_amt")
        clarity = ClarityParams(cr, ca)

    with st.expander("Sharpen"):
        c1, c2, c3 = st.columns(3)
        sa = c1.slider("Amount", 0, 150, 0, 1, key=f"{prefix}_sharpen_amt")
        sr = c2.number_input("Radius (px)", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key=f"{prefix}_sharpen_radius")
        sthr = c3.slider("Threshold", 0, 10, 0, 1, key=f"{prefix}_sharpen_thresh")
        sharpen = SharpenParams(sa, sr, int(sthr))

    with st.expander("Color → Gray Method", expanded=True):
        method = st.selectbox("Method", ["luminance", "max", "average", "channel_mix"], index=0, key=f"{prefix}_gray_method")
        if method == "channel_mix":
            c1, c2, c3 = st.columns(3)
            mr = c1.slider("Mix R", 0.0, 2.0, 1.0, 0.01, key=f"{prefix}_mix_r")
            mg = c2.slider("Mix G", 0.0, 2.0, 1.0, 0.01, key=f"{prefix}_mix_g")
            mb = c3.slider("Mix B", 0.0, 2.0, 1.0, 0.01, key=f"{prefix}_mix_b")
        else:
            mr, mg, mb = 1.0, 1.0, 1.0
        gray_params = GrayMethodParams(method, mr, mg, mb)

    return levels, cb, clarity, sharpen, gray_params


# -----------------------------
# Streamlit App
# -----------------------------

def main():
    st.set_page_config(page_title="SAMOILOFF Effect Composer", layout="wide")

    controls = sidebar_controls()

    if controls["reset_clicked"]:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    tabs = st.tabs(["R", "G", "B", "Final (White Light)"])

    # Process R
    with tabs[0]:
        st.header("R Target → Cyan Channel")
        levels_r, cb_r, clarity_r, sharpen_r, gray_r = per_image_controls("R")
        r_gray_lin = None
        r_preview = None
        r_padded = None
        if controls["upload_r"] is not None:
            try:
                file_bytes = controls["upload_r"].read()
                r_gray_lin, r_preview, r_padded = process_target_cached(
                    file_bytes,
                    controls["canvas_px"],
                    controls["padding_mode"],
                    (levels_r.black_point_pct, levels_r.white_point_pct, levels_r.gamma, levels_r.auto_normalize),
                    (cb_r.contrast, cb_r.brightness),
                    (clarity_r.radius_px, clarity_r.amount),
                    (sharpen_r.amount, sharpen_r.radius, sharpen_r.threshold),
                    (gray_r.method, gray_r.mix_r, gray_r.mix_g, gray_r.mix_b),
                )
                st.image(r_preview, caption="R Preview (grayscale of 1 − C)", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing R: {e}")
        else:
            st.info("Upload R-target in sidebar to preview.")

    # Process G
    with tabs[1]:
        st.header("G Target → Magenta Channel")
        levels_g, cb_g, clarity_g, sharpen_g, gray_g = per_image_controls("G")
        g_gray_lin = None
        g_preview = None
        g_padded = None
        if controls["upload_g"] is not None:
            try:
                file_bytes = controls["upload_g"].read()
                g_gray_lin, g_preview, g_padded = process_target_cached(
                    file_bytes,
                    controls["canvas_px"],
                    controls["padding_mode"],
                    (levels_g.black_point_pct, levels_g.white_point_pct, levels_g.gamma, levels_g.auto_normalize),
                    (cb_g.contrast, cb_g.brightness),
                    (clarity_g.radius_px, clarity_g.amount),
                    (sharpen_g.amount, sharpen_g.radius, sharpen_g.threshold),
                    (gray_g.method, gray_g.mix_r, gray_g.mix_g, gray_g.mix_b),
                )
                st.image(g_preview, caption="G Preview (grayscale of 1 − M)", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing G: {e}")
        else:
            st.info("Upload G-target in sidebar to preview.")

    # Process B
    with tabs[2]:
        st.header("B Target → Yellow Channel")
        levels_b, cb_b, clarity_b, sharpen_b, gray_b = per_image_controls("B")
        b_gray_lin = None
        b_preview = None
        b_padded = None
        if controls["upload_b"] is not None:
            try:
                file_bytes = controls["upload_b"].read()
                b_gray_lin, b_preview, b_padded = process_target_cached(
                    file_bytes,
                    controls["canvas_px"],
                    controls["padding_mode"],
                    (levels_b.black_point_pct, levels_b.white_point_pct, levels_b.gamma, levels_b.auto_normalize),
                    (cb_b.contrast, cb_b.brightness),
                    (clarity_b.radius_px, clarity_b.amount),
                    (sharpen_b.amount, sharpen_b.radius, sharpen_b.threshold),
                    (gray_b.method, gray_b.mix_r, gray_b.mix_g, gray_b.mix_b),
                )
                st.image(b_preview, caption="B Preview (grayscale of 1 − Y)", use_column_width=True)
            except Exception as e:
                st.error(f"Error processing B: {e}")
        else:
            st.info("Upload B-target in sidebar to preview.")

    # Final composite
    with tabs[3]:
        st.header("Final (White Light)")
        if all(x is not None for x in [locals().get("r_gray_lin"), locals().get("g_gray_lin"), locals().get("b_gray_lin")]):
            # Previews are 1−gray; recover CMY as C=1−Rgray, etc., then mix
            c_lin = clamp01(1.0 - r_gray_lin)
            m_lin = clamp01(1.0 - g_gray_lin)
            y_lin = clamp01(1.0 - b_gray_lin)
            composite = build_white_light_composite_srgb(c_lin, m_lin, y_lin, density=1.5)
            # Apply preview gamma then brightness (display intent only)
            preview_img = composite
            preview_img = apply_preview_gamma_srgb(preview_img, controls["gamma_preview"])
            preview_img = apply_preview_brightness_srgb(preview_img, controls["preview_brightness"])
            st.image(preview_img, caption="White-light Composite", use_column_width=True)
        else:
            st.info("Upload R/G/B targets to view the composite.")

        if controls["export_clicked"]:
            if all(x is not None for x in [locals().get("r_gray_lin"), locals().get("g_gray_lin"), locals().get("b_gray_lin")]):
                # Create output directory if it doesn't exist
                output_dir = controls["output_dir"]
                os.makedirs(output_dir, exist_ok=True)
                
                # Build exports
                final_img_rgb = build_white_light_composite_srgb(c_lin, m_lin, y_lin, density=1.5)

                # Determine alpha mask for transparent padding (union of non-transparent regions)
                alpha = None
                for pad in [locals().get("r_padded"), locals().get("g_padded"), locals().get("b_padded")]:
                    if isinstance(pad, Image.Image) and pad is not None and pad.mode == "RGBA":
                        a = np.asarray(pad.split()[-1]).astype(np.uint8)
                        alpha = a if alpha is None else np.maximum(alpha, a)
                if controls["padding_mode"] == "Transparent" and alpha is not None:
                    final_rgba = final_img_rgb.convert("RGBA")
                    r, g, b, _ = final_rgba.split()
                    final_rgba = Image.merge("RGBA", (r, g, b, Image.fromarray(alpha)))
                    final_export = final_rgba
                else:
                    final_export = final_img_rgb.convert("RGB")

                # Save final PNG to output directory
                final_path = os.path.join(output_dir, "final_white_light.png")
                final_export.save(final_path, format="PNG", optimize=True)

                # Preview stills (grayscale)
                def save_gray_png(pil_img: Image.Image, filename: str):
                    g = pil_img.convert("L")
                    path = os.path.join(output_dir, filename)
                    g.save(path, format="PNG", optimize=True)

                r_prev_img = locals().get("r_preview") or Image.new("RGB", controls["canvas_px"], (255, 255, 255))
                g_prev_img = locals().get("g_preview") or Image.new("RGB", controls["canvas_px"], (255, 255, 255))
                b_prev_img = locals().get("b_preview") or Image.new("RGB", controls["canvas_px"], (255, 255, 255))

                save_gray_png(r_prev_img, "preview_R.png")
                save_gray_png(g_prev_img, "preview_G.png")
                save_gray_png(b_prev_img, "preview_B.png")

                # Animated GIF (cached)
                try:
                    def to_png_bytes(img: Image.Image) -> bytes:
                        bio = io.BytesIO()
                        img.convert("RGB").save(bio, format="PNG", optimize=True)
                        bio.seek(0)
                        return bio.getvalue()

                    gif_bytes, _ = generate_cycle_gif_cached(
                        to_png_bytes(r_prev_img),
                        to_png_bytes(g_prev_img),
                        to_png_bytes(b_prev_img),
                        controls["gif_hold"],
                        controls["gif_fade"],
                        controls["gif_fps"],
                    )

                    # Save grayscale GIF
                    gif_path = os.path.join(output_dir, "rgb_cycle.gif")
                    with open(gif_path, "wb") as f:
                        f.write(gif_bytes)

                    # Colored stills and GIF
                    # Backlight frames should use pass-through = original gray (not inverted preview)
                    def backlight_from_preview(preview_img: Image.Image, channel_index: int) -> Image.Image:
                        lin_prev = srgb_to_linear(to_numpy_float01(preview_img)[:, :, :3])
                        gray_pass = clamp01(1.0 - lin_prev[:, :, 0])  # invert preview to get pass-through
                        h, w = gray_pass.shape
                        out_lin = np.zeros((h, w, 3), dtype=np.float32)
                        out_lin[:, :, channel_index] = gray_pass
                        return from_numpy_to_pil(linear_to_srgb(out_lin), mode_hint="RGB")

                    r_col_img = backlight_from_preview(r_prev_img, 0)
                    g_col_img = backlight_from_preview(g_prev_img, 1)
                    b_col_img = backlight_from_preview(b_prev_img, 2)

                    # Save colored stills
                    r_col_img.save(os.path.join(output_dir, "preview_R_colored.png"))
                    g_col_img.save(os.path.join(output_dir, "preview_G_colored.png"))
                    b_col_img.save(os.path.join(output_dir, "preview_B_colored.png"))

                    colored_gif_bytes, _ = generate_colored_cycle_gif(
                        r_prev_img, g_prev_img, b_prev_img,
                        controls["gif_hold"], controls["gif_fade"], controls["gif_fps"]
                    )

                    # Save colored GIF
                    colored_gif_path = os.path.join(output_dir, "rgb_cycle_colored.gif")
                    with open(colored_gif_path, "wb") as f:
                        f.write(colored_gif_bytes)

                except Exception as e:
                    st.warning(f"GIF generation failed: {e}")

                st.success(f"All files exported to: {output_dir}")
                st.write("**Exported files:**")
                st.write(f"- `{final_path}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_R.png')}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_G.png')}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_B.png')}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_R_colored.png')}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_G_colored.png')}`")
                st.write(f"- `{os.path.join(output_dir, 'preview_B_colored.png')}`")
                st.write(f"- `{gif_path}`")
                st.write(f"- `{colored_gif_path}`")
            else:
                st.error("Need all R/G/B targets uploaded to export.")


if __name__ == "__main__":
    main()
