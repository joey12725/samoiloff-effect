import io

import numpy as np
import pytest
from PIL import Image

from samoiloff.core import (
    BLUE,
    GREEN,
    RED,
    WHITE,
    ClarityParams,
    ContrastBrightnessParams,
    GrayMethodParams,
    LevelsParams,
    SharpenParams,
    apply_clarity,
    apply_cmy_density,
    apply_contrast_brightness_linear,
    apply_levels_exposure_linear,
    apply_sharpen,
    build_white_light_composite_srgb,
    clamp01,
    compose,
    compute_canvas_pixels,
    contain_center_pad,
    crossfade_gif,
    export_bundle,
    generate_colored_cycle_gif,
    linear_to_srgb,
    open_image,
    process_target,
    srgb_to_linear,
    tint_grayscale_to_color,
    to_grayscale,
    to_numpy_float01,
    under_light,
)


def png(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


@pytest.fixture
def random_grays():
    rng = np.random.default_rng(7)
    return tuple(rng.random((24, 36), dtype=np.float32) for _ in range(3))


# --- color math ---------------------------------------------------------------


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    np.testing.assert_allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_srgb_known_values():
    # sRGB 0.5 is ~21.4% linear light; the toe is linear below 0.04045.
    assert srgb_to_linear(np.array([0.5]))[0] == pytest.approx(0.2140, abs=1e-3)
    assert srgb_to_linear(np.array([0.04]))[0] == pytest.approx(0.04 / 12.92)


def test_clamp01():
    np.testing.assert_array_equal(clamp01(np.array([-0.5, 0.3, 1.2])), [0.0, 0.3, 1.0])


# --- canvas -------------------------------------------------------------------


def test_compute_canvas_pixels():
    assert compute_canvas_pixels(6.0, 4.0, 300) == (1800, 1200)


def test_contain_center_pad_white():
    padded = contain_center_pad(Image.new("RGB", (400, 200), (255, 0, 0)), (600, 600), "White")
    assert padded.size == (600, 600) and padded.mode == "RGB"
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((300, 300)) == (255, 0, 0)


def test_contain_center_pad_transparent():
    padded = contain_center_pad(Image.new("RGB", (200, 400), (0, 255, 0)), (600, 600), "Transparent")
    assert padded.mode == "RGBA"
    assert padded.getpixel((0, 300))[3] == 0
    assert padded.getpixel((300, 300)) == (0, 255, 0, 255)


def test_open_image_handles_palette_and_alpha():
    pal = Image.new("P", (4, 4))
    assert open_image(png(pal)).mode == "RGB"
    la = Image.new("LA", (4, 4), (128, 0))
    assert open_image(png(la)).mode == "RGBA"


def test_open_image_applies_exif_rotation():
    img = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    bio = io.BytesIO()
    img.save(bio, format="JPEG", exif=exif)
    assert open_image(bio.getvalue()).size == (20, 40)


# --- adjustments --------------------------------------------------------------


def test_levels_stretch_black_and_white_points():
    x = np.array([[[0.1, 0.5, 0.9]]], dtype=np.float32)
    out = apply_levels_exposure_linear(x, LevelsParams(10, 90, 1.0, False))
    np.testing.assert_allclose(out[0, 0], [0.0, 0.5, 1.0], atol=1e-5)


def test_contrast_brightness_stays_in_range():
    x = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = apply_contrast_brightness_linear(x, ContrastBrightnessParams(50, 20))
    assert out.min() >= 0 and out.max() <= 1
    np.testing.assert_allclose(out, 0.7, atol=1e-6)


@pytest.mark.parametrize(
    "fn,params",
    [(apply_clarity, ClarityParams(5.0, 0.0)), (apply_sharpen, SharpenParams(0, 1.0, 0))],
)
def test_zero_amount_is_noop(fn, params):
    x = np.full((10, 10, 3), 0.5, dtype=np.float32)
    np.testing.assert_array_equal(fn(x, params), x)


def test_grayscale_methods():
    img = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0.5, 0.5, 0.5]]], dtype=np.float32)
    lum = to_grayscale(img, GrayMethodParams("luminance"))
    assert lum[0, 0] == pytest.approx(0.2126)
    assert lum[0, 1] == pytest.approx(0.7152)
    assert to_grayscale(img, GrayMethodParams("max"))[0, 0] == pytest.approx(1.0)
    assert to_grayscale(img, GrayMethodParams("average"))[1, 1] == pytest.approx(0.5)
    assert to_grayscale(img, GrayMethodParams("channel_mix", 0.5, 0.3, 0.2))[0, 0] == pytest.approx(0.5)
    assert to_grayscale(img, GrayMethodParams("channel_mix", 0, 0, 0))[0, 0] == pytest.approx(1 / 3)


def test_process_target_shapes_and_values():
    gray, padded = process_target(png(Image.new("RGB", (100, 50), (128, 128, 128))), (200, 200))
    assert gray.shape == (200, 200) and gray.dtype == np.float32
    assert padded.size == (200, 200)
    assert gray[100, 100] == pytest.approx(srgb_to_linear(np.array([128 / 255]))[0], abs=1e-3)
    assert gray[0, 0] == pytest.approx(1.0)  # white padding


# --- optics -------------------------------------------------------------------


def test_density_curve_endpoints():
    zeros, ones = np.zeros(3), np.ones(3)
    for out in apply_cmy_density(zeros, ones, np.full(3, 0.5), 1.5):
        assert out.min() >= 0 and out.max() <= 1
    c, m, _ = apply_cmy_density(zeros, ones, zeros, 1.5)
    np.testing.assert_allclose(c, 0)
    np.testing.assert_allclose(m, 1)


def test_each_light_reveals_only_its_target(random_grays):
    """The core claim: under a single primary you see one target and nothing else."""
    r, g, b = random_grays
    comp = compose(r, g, b, density=1.0)
    for light, target, ch in ((RED, r, 0), (GREEN, g, 1), (BLUE, b, 2)):
        seen = under_light(comp.transmittance, light)
        np.testing.assert_allclose(seen[:, :, ch], target, atol=1e-6)
        others = [i for i in range(3) if i != ch]
        assert np.all(seen[:, :, others] == 0)


def test_density_darkens_each_reveal(random_grays):
    r, g, b = random_grays
    comp = compose(r, g, b, density=2.0)
    np.testing.assert_allclose(comp.transmittance[:, :, 0], r**2, atol=1e-5)


def test_white_light_composite_matches_transmittance():
    c = np.array([[0.5, 0.0], [1.0, 0.2]], dtype=np.float32)
    m = np.array([[0.0, 0.8], [0.3, 0.0]], dtype=np.float32)
    y = np.array([[0.7, 0.4], [0.0, 1.0]], dtype=np.float32)
    k = 1.5
    arr = srgb_to_linear(to_numpy_float01(build_white_light_composite_srgb(c, m, y, density=k)))
    for i, ink in enumerate((c, m, y)):
        np.testing.assert_allclose(arr[:, :, i], (1 - ink) ** k, atol=0.01)


def test_compose_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compose(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)))


def test_compose_carries_transparent_padding(random_grays):
    padded = contain_center_pad(Image.new("RGB", (36, 12)), (36, 24), "Transparent")
    comp = compose(*random_grays, padded=[padded])
    out = comp.white_light()
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0))[3] == 0
    assert out.getpixel((18, 12))[3] == 255


def test_plates_are_ink_on_paper(random_grays):
    comp = compose(np.ones((4, 4)), np.zeros((4, 4)), np.ones((4, 4)))
    plates = comp.plates()
    assert plates["cyan"].getpixel((0, 0)) == (255, 255, 255)  # no ink -> paper
    assert plates["magenta"].getpixel((0, 0)) == (255, 0, 255)  # full ink


def test_tint_grayscale_to_color():
    tinted = tint_grayscale_to_color(Image.new("RGB", (4, 4), (255, 255, 255)), (1.0, 0.0, 0.0))
    assert tinted.getpixel((1, 1)) == (255, 0, 0)


# --- animation and export -----------------------------------------------------


def gif_frames(data: bytes):
    im = Image.open(io.BytesIO(data))
    frames = []
    for i in range(im.n_frames):
        im.seek(i)
        frames.append((im.convert("RGB").copy(), im.info["duration"]))
    return frames


def test_crossfade_gif_timing_and_colors():
    stills = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    data, total = crossfade_gif(stills, hold_s=0.5, fade_s=0.25, fps=20)
    frames = gif_frames(data)
    assert sum(d for _, d in frames) == pytest.approx(total * 1000, abs=30)
    assert total == pytest.approx(3 * (0.5 + 0.25), abs=0.05)
    assert frames[0][0].getpixel((0, 0)) == (255, 0, 0)


def test_colored_cycle_starts_under_red_light(random_grays):
    comp = compose(*random_grays)
    data, _ = generate_colored_cycle_gif(comp.transmittance, 0.2, 0.1, 10)
    first = np.asarray(gif_frames(data)[0][0])
    assert first[:, :, 1:].max() == 0
    assert first[:, :, 0].max() > 200


def test_export_bundle_files(random_grays):
    files = export_bundle(compose(*random_grays), 0.1, 0.1, 5)
    assert set(files) == {
        "print_white_light.png",
        "plate_cyan.png",
        "plate_magenta.png",
        "plate_yellow.png",
        "under_red_light.png",
        "under_green_light.png",
        "under_blue_light.png",
        "rgb_cycle_colored.gif",
        "rgb_cycle.gif",
    }
    for name, data in files.items():
        Image.open(io.BytesIO(data)).verify()


def test_white_light_constant_is_neutral():
    t = np.full((2, 2, 3), 0.25, dtype=np.float32)
    np.testing.assert_allclose(under_light(t, WHITE), t)
