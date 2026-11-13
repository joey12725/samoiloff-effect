"""Streamlit UI for the Samoiloff-effect print composer.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st

from samoiloff.core import (
    BLUE,
    DEFAULT_CANVAS_INCHES,
    DEFAULT_DENSITY,
    DPI,
    GRAY_METHODS,
    GREEN,
    RED,
    ClarityParams,
    Composition,
    ContrastBrightnessParams,
    GrayMethodParams,
    LevelsParams,
    SharpenParams,
    TargetSettings,
    apply_preview_brightness_srgb,
    apply_preview_gamma_srgb,
    compose,
    compute_canvas_pixels,
    export_bundle,
    linear_to_pil,
    plate_image,
    process_target,
)

SAMPLE_DIR = Path(__file__).parent / "examples" / "targets"

CHANNELS = [
    # key, light name, ink name, light color, sample file
    ("R", "red", "cyan", RED, "red_moon.png"),
    ("G", "green", "magenta", GREEN, "green_sun.png"),
    ("B", "blue", "yellow", BLUE, "blue_boat.png"),
]


@st.cache_data(show_spinner=False)
def process_target_cached(file_bytes: bytes, canvas_px: tuple[int, int], padding_mode: str, settings: TargetSettings):
    return process_target(file_bytes, canvas_px, padding_mode, settings)


def export_zip(comp: Composition, hold_s: float, fade_s: float, fps: int) -> bytes:
    files = export_bundle(comp, hold_s, fade_s, fps)
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return bio.getvalue()


def sidebar_controls() -> dict:
    sb = st.sidebar
    sb.title("Samoiloff Composer")

    sb.subheader("Targets")
    use_samples = sb.toggle("Use sample images", value=True, help="Bundled demo targets. Turn off to upload your own.")
    uploads = {}
    for key, light, _ink, _rgb, _sample in CHANNELS:
        uploads[key] = sb.file_uploader(
            f"Shows under {light} light",
            type=["png", "jpg", "jpeg"],
            key=f"upload_{key}",
            disabled=use_samples,
        )

    sb.subheader("Canvas")
    colw, colh = sb.columns(2)
    width_in = colw.number_input("Width (in)", 0.5, 40.0, DEFAULT_CANVAS_INCHES[0], 0.25)
    height_in = colh.number_input("Height (in)", 0.5, 40.0, DEFAULT_CANVAS_INCHES[1], 0.25)
    canvas_px = compute_canvas_pixels(width_in, height_in, DPI)
    sb.caption(f"{canvas_px[0]} × {canvas_px[1]} px at {DPI} DPI")
    padding_mode = sb.selectbox("Padding", ["White", "Transparent"])

    sb.subheader("Print")
    density = sb.slider(
        "Ink density (k)",
        0.5,
        3.0,
        DEFAULT_DENSITY,
        0.05,
        help="Ink response curve 1 − (1 − x)^k. Raise it if your prints come out washed out.",
    )
    gamma_preview = sb.slider("Preview gamma", 0.5, 2.5, 1.0, 0.05, help="On-screen only. Exports are unaffected.")
    preview_brightness = sb.slider("Preview brightness (%)", -20, 20, 0, 1, help="On-screen only.")

    sb.subheader("Animation")
    hold_s = sb.number_input("Hold (s)", 0.1, 10.0, 1.5, 0.1)
    fade_s = sb.number_input("Fade (s)", 0.1, 10.0, 0.8, 0.1)
    fps = sb.number_input("FPS", 1, 60, 24, 1)

    if sb.button("Reset all controls"):
        st.session_state.clear()
        st.rerun()

    return {
        "use_samples": use_samples,
        "uploads": uploads,
        "canvas_px": canvas_px,
        "padding_mode": padding_mode,
        "density": density,
        "gamma_preview": gamma_preview,
        "preview_brightness": preview_brightness / 100.0,
        "gif": (hold_s, fade_s, int(fps)),
    }


def per_image_controls(key: str) -> TargetSettings:
    with st.expander("Levels / exposure"):
        c1, c2, c3, c4 = st.columns(4)
        levels = LevelsParams(
            c1.slider("Black %", 0, 100, 0, 1, key=f"{key}_bp"),
            c2.slider("White %", 0, 100, 100, 1, key=f"{key}_wp"),
            c3.slider("Gamma", 0.10, 4.0, 1.0, 0.01, key=f"{key}_gamma"),
            c4.checkbox("Auto-normalize", key=f"{key}_auto"),
        )
    with st.expander("Contrast and brightness"):
        c1, c2 = st.columns(2)
        cb = ContrastBrightnessParams(
            c1.slider("Contrast", -100, 100, 0, 1, key=f"{key}_contrast"),
            c2.slider("Brightness", -100, 100, 0, 1, key=f"{key}_brightness"),
        )
    with st.expander("Clarity (local contrast)"):
        c1, c2 = st.columns(2)
        clarity = ClarityParams(
            c1.number_input("Radius (px)", 0.0, 200.0, 0.0, 0.5, key=f"{key}_clarity_radius"),
            c2.slider("Amount", -100, 100, 0, 1, key=f"{key}_clarity_amt"),
        )
    with st.expander("Sharpen"):
        c1, c2, c3 = st.columns(3)
        sharpen = SharpenParams(
            c1.slider("Amount", 0, 150, 0, 1, key=f"{key}_sharpen_amt"),
            c2.number_input("Radius (px)", 0.1, 3.0, 1.0, 0.1, key=f"{key}_sharpen_radius"),
            int(c3.slider("Threshold", 0, 10, 0, 1, key=f"{key}_sharpen_thresh")),
        )
    with st.expander("Color to gray"):
        method = st.selectbox("Method", GRAY_METHODS, key=f"{key}_gray_method")
        mix = (1.0, 1.0, 1.0)
        if method == "channel_mix":
            c1, c2, c3 = st.columns(3)
            mix = (
                c1.slider("Mix R", 0.0, 2.0, 1.0, 0.01, key=f"{key}_mix_r"),
                c2.slider("Mix G", 0.0, 2.0, 1.0, 0.01, key=f"{key}_mix_g"),
                c3.slider("Mix B", 0.0, 2.0, 1.0, 0.01, key=f"{key}_mix_b"),
            )
        gray = GrayMethodParams(method, *mix)
    return TargetSettings(levels, cb, clarity, sharpen, gray)


def target_bytes(controls: dict, key: str, sample: str) -> bytes | None:
    if controls["use_samples"]:
        return (SAMPLE_DIR / sample).read_bytes()
    upload = controls["uploads"][key]
    return upload.getvalue() if upload is not None else None


def display(img, controls: dict):
    img = apply_preview_gamma_srgb(img, controls["gamma_preview"])
    return apply_preview_brightness_srgb(img, controls["preview_brightness"])


def main() -> None:
    st.set_page_config(page_title="Samoiloff Composer", layout="wide")
    controls = sidebar_controls()

    tabs = st.tabs(["Print", *(f"{light.title()} light" for _, light, *_ in CHANNELS)])
    grays, padded = {}, []

    for tab, (key, light, ink, light_rgb, sample) in zip(tabs[1:], CHANNELS, strict=True):
        with tab:
            st.subheader(f"Image revealed by {light} light, printed in {ink} ink")
            settings = per_image_controls(key)
            data = target_bytes(controls, key, sample)
            if data is None:
                st.info(f"Upload the {light}-light image in the sidebar.")
                continue
            try:
                gray, pad = process_target_cached(data, controls["canvas_px"], controls["padding_mode"], settings)
            except Exception as exc:  # bad or unsupported upload
                st.error(f"Could not process this image: {exc}")
                continue
            grays[key] = gray
            padded.append(pad)
            plate = plate_image(1.0 - gray, tuple(1.0 - c for c in light_rgb))
            col1, col2 = st.columns(2)
            col1.image(display(plate, controls), caption=f"{ink.title()} ink plate", width="stretch")
            lit = linear_to_pil(gray[:, :, None] * np.array(light_rgb, dtype=np.float32))
            col2.image(display(lit, controls), caption=f"Target under {light} light", width="stretch")

    with tabs[0]:
        if len(grays) < 3:
            st.info("Add all three target images to compose the print.")
            return
        comp = compose(grays["R"], grays["G"], grays["B"], density=controls["density"], padded=padded)
        st.image(display(comp.white_light(), controls), caption="The print under white light", width="stretch")
        cols = st.columns(3)
        for col, (_, light, _ink, light_rgb, _s) in zip(cols, CHANNELS, strict=True):
            col.image(display(comp.lit_by(light_rgb), controls), caption=f"Under {light} light", width="stretch")

        st.download_button(
            "Download all exports (.zip)",
            data=lambda: export_zip(comp, *controls["gif"]),
            file_name="samoiloff_exports.zip",
            mime="application/zip",
            type="primary",
        )
        st.caption(
            "The bundle holds the printable image, CMY ink plates, red/green/blue light previews and two "
            "animated GIFs. It renders when you click, so large canvases take a few seconds."
        )


if __name__ == "__main__":
    main()
