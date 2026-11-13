#!/usr/bin/env python3
"""Run the full pipeline headless on three generated sample images.

    python demo.py                     # writes exports to ./output
    python demo.py --samples           # also (re)draws examples/targets/*.png
    python demo.py --hero docs         # also renders docs/hero.png and docs/cycle.gif

The sample targets are drawn procedurally here, so they are free to reuse.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from samoiloff.core import (
    BLUE,
    GREEN,
    RED,
    Composition,
    TargetSettings,
    compose,
    compute_canvas_pixels,
    export_bundle,
    generate_colored_cycle_gif,
    process_target,
)

ROOT = Path(__file__).parent
SAMPLE_DIR = ROOT / "examples" / "targets"
SAMPLE_FILES = ("red_moon.png", "green_sun.png", "blue_boat.png")
SAMPLE_SIZE = (900, 600)
INK = (20, 20, 20)
PAPER = (255, 255, 255)


# -----------------------------------------------------------------------------
# Procedural sample targets
# -----------------------------------------------------------------------------


def draw_moon(size=SAMPLE_SIZE) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    cx, cy, r = int(w * 0.40), h // 2, int(h * 0.34)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=INK)
    off = int(r * 0.55)
    d.ellipse([cx - r + off, cy - r - off // 3, cx + r + off, cy + r - off // 3], fill=PAPER)
    for sx, sy, sr in [(0.70, 0.22, 34), (0.82, 0.55, 24), (0.64, 0.78, 18), (0.15, 0.18, 20), (0.90, 0.85, 14)]:
        draw_star(d, (int(w * sx), int(h * sy)), sr)
    return img


def draw_star(d: ImageDraw.ImageDraw, center, radius, points=5):
    cx, cy = center
    pts = []
    for i in range(points * 2):
        ang = math.pi / points * i - math.pi / 2
        rad = radius if i % 2 == 0 else radius * 0.42
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    d.polygon(pts, fill=INK)


def draw_sun(size=SAMPLE_SIZE) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    cx, cy, r = w // 2, h // 2, int(h * 0.20)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=INK)
    for i in range(12):
        ang = 2 * math.pi * i / 12
        inner, outer, half = r * 1.3, r * 2.15, 0.11
        d.polygon(
            [
                (cx + inner * math.cos(ang - half), cy + inner * math.sin(ang - half)),
                (cx + outer * math.cos(ang), cy + outer * math.sin(ang)),
                (cx + inner * math.cos(ang + half), cy + inner * math.sin(ang + half)),
            ],
            fill=INK,
        )
    return img


def draw_boat(size=SAMPLE_SIZE) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    mast_x, deck_y = int(w * 0.50), int(h * 0.68)
    d.polygon([(mast_x + 8, int(h * 0.12)), (mast_x + 8, deck_y - 20), (int(w * 0.72), deck_y - 20)], fill=INK)
    d.polygon([(mast_x - 8, int(h * 0.20)), (mast_x - 8, deck_y - 20), (int(w * 0.33), deck_y - 20)], fill=INK)
    d.polygon(
        [(int(w * 0.26), deck_y), (int(w * 0.76), deck_y), (int(w * 0.68), deck_y + 52), (int(w * 0.33), deck_y + 52)],
        fill=INK,
    )
    for row, y in enumerate((int(h * 0.86), int(h * 0.94))):
        shift = 40 * row
        for x in range(-80 + shift, w + 80, 120):
            d.arc([x, y - 22, x + 120, y + 22], 200, 340, fill=INK, width=12)
    return img


def sample_targets() -> list[Image.Image]:
    return [draw_moon(), draw_sun(), draw_boat()]


def png_bytes(img: Image.Image) -> bytes:
    bio = io.BytesIO()
    img.save(bio, format="PNG", optimize=True)
    return bio.getvalue()


# -----------------------------------------------------------------------------
# Hero image for the README
# -----------------------------------------------------------------------------


def render_hero(targets: list[Image.Image], comp: Composition, tile_w: int = 360) -> Image.Image:
    """Inputs on the left, the print in the middle, the three reveals on the right."""
    tile_h = int(tile_w * 2 / 3)
    big_w, big_h = int(tile_w * 1.5), int(tile_h * 1.5)
    gap, pad, label_h = 28, 48, 40
    bg, fg, muted = (246, 245, 241), (28, 28, 30), (110, 110, 115)

    col_h = 3 * tile_h + 2 * (gap + label_h)
    width = pad * 2 + tile_w * 2 + big_w + 2 * (gap + 40)
    height = pad * 2 + col_h + 60
    hero = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(hero)
    try:
        font = ImageFont.load_default(size=22)
        title = ImageFont.load_default(size=26)
    except TypeError:  # Pillow < 10.1
        font = title = ImageFont.load_default()

    def tile(img: Image.Image, x: int, y: int, w: int, h: int, label: str):
        hero.paste(img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS), (x, y))
        d.rectangle([x - 1, y - 1, x + w, y + h], outline=(210, 208, 202))
        d.text((x, y + h + 8), label, fill=muted, font=font)

    top = pad + 50
    x0 = pad
    d.text((x0, pad), "Three images in", fill=fg, font=title)
    for i, (img, light) in enumerate(zip(targets, ("red", "green", "blue"), strict=True)):
        tile(img, x0, top + i * (tile_h + gap + label_h), tile_w, tile_h, f"for {light} light")

    x1 = x0 + tile_w + gap + 40
    d.text((x1, pad), "One print", fill=fg, font=title)
    big_y = top + (col_h - big_h) // 2 - label_h // 2
    tile(comp.white_light(), x1, big_y, big_w, big_h, "under white light")
    arrow_y = big_y + big_h // 2
    for ax in (x1 - 40, x1 + big_w + 12):
        d.line([(ax, arrow_y), (ax + 26, arrow_y)], fill=muted, width=4)
        d.polygon([(ax + 26, arrow_y - 9), (ax + 38, arrow_y), (ax + 26, arrow_y + 9)], fill=muted)

    x2 = x1 + big_w + gap + 40
    d.text((x2, pad), "Change the light", fill=fg, font=title)
    for i, (light, name) in enumerate(((RED, "red"), (GREEN, "green"), (BLUE, "blue"))):
        tile(comp.lit_by(light), x2, top + i * (tile_h + gap + label_h), tile_w, tile_h, f"under {name} light")
    return hero


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def run(output_dir: Path, canvas_in=(6.0, 4.0), samples: bool = False, hero_dir: Path | None = None) -> dict:
    targets = sample_targets()
    if samples:
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        for img, name in zip(targets, SAMPLE_FILES, strict=True):
            img.save(SAMPLE_DIR / name, optimize=True)
            print(f"wrote {SAMPLE_DIR / name}")

    canvas_px = compute_canvas_pixels(*canvas_in)
    grays = [process_target(png_bytes(t), canvas_px, "White", TargetSettings())[0] for t in targets]
    comp = compose(*grays)

    output_dir.mkdir(parents=True, exist_ok=True)
    files = export_bundle(comp, hold_s=1.0, fade_s=0.5, fps=12)
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
        print(f"wrote {output_dir / name} ({len(data) / 1024:.0f} KB)")

    if hero_dir is not None:
        hero_dir.mkdir(parents=True, exist_ok=True)
        render_hero(targets, comp).save(hero_dir / "hero.png", optimize=True)
        small = compose(*[process_target(png_bytes(t), (480, 320), "White", TargetSettings())[0] for t in targets])
        gif, _ = generate_colored_cycle_gif(small.transmittance, hold_s=1.0, fade_s=0.6, fps=12)
        (hero_dir / "cycle.gif").write_bytes(gif)
        print(f"wrote {hero_dir / 'hero.png'} and {hero_dir / 'cycle.gif'}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="where exports go (default: output)")
    parser.add_argument("--samples", action="store_true", help="redraw examples/targets/*.png")
    parser.add_argument("--hero", type=Path, metavar="DIR", help="render hero.png and cycle.gif into DIR")
    args = parser.parse_args()
    run(args.output_dir, samples=args.samples, hero_dir=args.hero)


if __name__ == "__main__":
    main()
