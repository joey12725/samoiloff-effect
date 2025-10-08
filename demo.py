#!/usr/bin/env python3
"""
Demo script for SAMOILOFF Effect Composer.
Creates sample images and demonstrates the processing pipeline.
"""

import io
import os
import argparse
from PIL import Image, ImageDraw
import numpy as np
from app import (
    process_target, build_white_light_composite_srgb, generate_cycle_gif,
    LevelsParams, ContrastBrightnessParams, ClarityParams, 
    SharpenParams, GrayMethodParams, compute_canvas_pixels,
    tint_grayscale_to_color, generate_colored_cycle_gif
)

def create_sample_images():
    """Create sample R, G, B target images"""
    size = (200, 200)
    
    # R target: Red gradient
    r_img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(r_img)
    for y in range(size[1]):
        intensity = int(255 * y / size[1])
        draw.line([(0, y), (size[0], y)], fill=(intensity, 0, 0))
    
    # G target: Green circle
    g_img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(g_img)
    center = (size[0]//2, size[1]//2)
    radius = min(size) // 3
    draw.ellipse([center[0]-radius, center[1]-radius, 
                  center[0]+radius, center[1]+radius], 
                 fill=(0, 255, 0))
    
    # B target: Blue checkerboard
    b_img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(b_img)
    checker_size = 20
    for x in range(0, size[0], checker_size):
        for y in range(0, size[1], checker_size):
            if (x // checker_size + y // checker_size) % 2 == 0:
                draw.rectangle([x, y, x+checker_size, y+checker_size], 
                              fill=(0, 0, 255))
    
    return r_img, g_img, b_img

def save_image_bytes(img):
    """Convert PIL Image to bytes"""
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def main():
    """Run the demo"""
    parser = argparse.ArgumentParser(description="SAMOILOFF Effect Composer Demo")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    print("SAMOILOFF Effect Composer Demo")
    print("=" * 40)
    
    # Create sample images
    print("Creating sample images...")
    r_img, g_img, b_img = create_sample_images()
    
    # Save sample images
    r_img.save(os.path.join(outdir, "sample_r.png"))
    g_img.save(os.path.join(outdir, "sample_g.png"))
    b_img.save(os.path.join(outdir, "sample_b.png"))
    print("✓ Sample images saved: sample_r.png, sample_g.png, sample_b.png")
    
    # Set up processing parameters
    canvas_px = compute_canvas_pixels(6.0, 4.0, 300)  # 1800x1200
    print(f"✓ Canvas size: {canvas_px[0]}x{canvas_px[1]} pixels")
    
    levels = LevelsParams(0, 100, 1.0, False)
    contrast_brightness = ContrastBrightnessParams(0, 0)
    clarity = ClarityParams(0.0, 0.0)
    sharpen = SharpenParams(0, 1.0, 0)
    gray_params = GrayMethodParams("luminance", 1.0, 1.0, 1.0)
    
    # Process R target
    print("Processing R target...")
    r_bytes = save_image_bytes(r_img)
    r_gray, r_preview, r_padded = process_target(
        r_bytes, canvas_px, "White", 
        levels, contrast_brightness, clarity, sharpen, gray_params
    )
    r_preview.save(os.path.join(outdir, "demo_r_preview.png"))
    print("✓ R processing complete: demo_r_preview.png")
    
    # Process G target
    print("Processing G target...")
    g_bytes = save_image_bytes(g_img)
    g_gray, g_preview, g_padded = process_target(
        g_bytes, canvas_px, "White",
        levels, contrast_brightness, clarity, sharpen, gray_params
    )
    g_preview.save(os.path.join(outdir, "demo_g_preview.png"))
    print("✓ G processing complete: demo_g_preview.png")
    
    # Process B target
    print("Processing B target...")
    b_bytes = save_image_bytes(b_img)
    b_gray, b_preview, b_padded = process_target(
        b_bytes, canvas_px, "White",
        levels, contrast_brightness, clarity, sharpen, gray_params
    )
    b_preview.save(os.path.join(outdir, "demo_b_preview.png"))
    print("✓ B processing complete: demo_b_preview.png")
    
    # Create final composite
    print("Creating final composite...")
    final_composite = build_white_light_composite_srgb(r_gray, g_gray, b_gray)
    final_composite.save(os.path.join(outdir, "demo_final_composite.png"))
    print("✓ Final composite saved: demo_final_composite.png")
    
    # Generate animated GIFs (grayscale and colored)
    print("Generating animated GIFs...")
    gif_bytes, duration = generate_cycle_gif(r_preview, g_preview, b_preview, 1.0, 0.5, 24)
    with open(os.path.join(outdir, "demo_rgb_cycle.gif"), "wb") as f:
        f.write(gif_bytes)

    colored_gif_bytes, duration2 = generate_colored_cycle_gif(r_preview, g_preview, b_preview, 1.0, 0.5, 24)
    with open(os.path.join(outdir, "demo_rgb_cycle_colored.gif"), "wb") as f:
        f.write(colored_gif_bytes)

    # Save backlight-simulated stills (R/G/B backgrounds with black occlusions)
    from app import srgb_to_linear, linear_to_srgb, to_numpy_float01, from_numpy_to_pil, clamp01

    def gray_to_backlight_frame(gray_img: Image.Image, channel_index: int) -> Image.Image:
        lin = srgb_to_linear(to_numpy_float01(gray_img)[:, :, :3])
        gray_lin = lin[:, :, 0]
        h, w = gray_lin.shape
        out_lin = np.zeros((h, w, 3), dtype=np.float32)
        out_lin[:, :, channel_index] = clamp01(gray_lin)
        return from_numpy_to_pil(linear_to_srgb(out_lin))

    r_col = gray_to_backlight_frame(r_preview, 0)
    g_col = gray_to_backlight_frame(g_preview, 1)
    b_col = gray_to_backlight_frame(b_preview, 2)

    r_col.save(os.path.join(outdir, "demo_r_preview_colored.png"))
    g_col.save(os.path.join(outdir, "demo_g_preview_colored.png"))
    b_col.save(os.path.join(outdir, "demo_b_preview_colored.png"))

    print(f"✓ Animated GIF saved: demo_rgb_cycle.gif (duration: {duration:.2f}s)")
    print(f"✓ Colored GIF saved: demo_rgb_cycle_colored.gif (duration: {duration2:.2f}s)")
    
    print("\n" + "=" * 40)
    print("Demo complete! Generated files:")
    print("- sample_r.png, sample_g.png, sample_b.png (input images)")
    print("- demo_r_preview.png, demo_g_preview.png, demo_b_preview.png (previews)")
    print("- demo_r_preview_colored.png, demo_g_preview_colored.png, demo_b_preview_colored.png (colored previews)")
    print("- demo_final_composite.png (final composite)")
    print("- demo_rgb_cycle.gif (animated cycle)")
    print("- demo_rgb_cycle_colored.gif (animated colored cycle)")
    print("\nYou can now run the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
