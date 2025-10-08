#!/usr/bin/env python3
"""
Integration test for the SAMOILOFF Effect Composer Streamlit app.
This test verifies that the app can be imported and basic functions work.
"""

import sys
import os
import tempfile
import io
from PIL import Image

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_import():
    """Test that the app can be imported without errors"""
    try:
        import app
        print("✓ App module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import app: {e}")
        return False

def test_basic_functions():
    """Test that basic functions are available and work"""
    try:
        import app
        
        # Test color space functions
        import numpy as np
        test_array = np.array([0.5, 0.8, 0.2])
        
        # Test sRGB to linear conversion
        linear = app.srgb_to_linear(test_array)
        assert linear.shape == test_array.shape
        print("✓ sRGB to linear conversion works")
        
        # Test linear to sRGB conversion
        srgb = app.linear_to_srgb(linear)
        assert srgb.shape == test_array.shape
        print("✓ Linear to sRGB conversion works")
        
        # Test canvas pixel calculation
        w, h = app.compute_canvas_pixels(6.0, 4.0, 300)
        assert w == 1800 and h == 1200
        print("✓ Canvas pixel calculation works")
        
        return True
    except Exception as e:
        print(f"✗ Basic functions test failed: {e}")
        return False

def test_image_processing():
    """Test image processing pipeline with sample images"""
    try:
        import app
        
        # Create test images
        test_img = Image.new('RGB', (100, 100), (128, 128, 128))
        
        # Save to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test processing parameters
        levels = app.LevelsParams(0, 100, 1.0, False)
        contrast_brightness = app.ContrastBrightnessParams(0, 0)
        clarity = app.ClarityParams(0.0, 0.0)
        sharpen = app.SharpenParams(0, 1.0, 0)
        gray_params = app.GrayMethodParams("luminance", 1.0, 1.0, 1.0)
        
        # Process image
        gray_channel, preview, padded = app.process_target(
            img_bytes.getvalue(),
            (200, 200),
            "White",
            levels, contrast_brightness, clarity, sharpen, gray_params
        )
        
        # Verify outputs
        assert gray_channel.shape == (200, 200)
        assert preview.size == (200, 200)
        assert padded.size == (200, 200)
        print("✓ Image processing pipeline works")
        
        return True
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False

def test_cmy_composite():
    """Test CMY mapping and composite generation"""
    try:
        import app
        import numpy as np
        
        # Create test CMY channels
        c_channel = np.ones((50, 50), dtype=np.float32) * 0.5
        m_channel = np.ones((50, 50), dtype=np.float32) * 0.3
        y_channel = np.ones((50, 50), dtype=np.float32) * 0.7
        
        # Generate composite
        composite = app.build_white_light_composite_srgb(c_channel, m_channel, y_channel)
        
        # Verify output
        assert isinstance(composite, Image.Image)
        assert composite.size == (50, 50)
        print("✓ CMY composite generation works")
        
        return True
    except Exception as e:
        print(f"✗ CMY composite test failed: {e}")
        return False

def test_gif_generation():
    """Test GIF generation"""
    try:
        import app
        
        # Create test images
        r_img = Image.new('RGB', (50, 50), (255, 0, 0))
        g_img = Image.new('RGB', (50, 50), (0, 255, 0))
        b_img = Image.new('RGB', (50, 50), (0, 0, 255))
        
        # Generate GIF
        gif_bytes, duration = app.generate_cycle_gif(r_img, g_img, b_img, 0.1, 0.1, 10)
        
        # Verify output
        assert isinstance(gif_bytes, bytes)
        assert len(gif_bytes) > 0
        assert isinstance(duration, float)
        assert duration > 0
        print("✓ GIF generation works")
        
        return True
    except Exception as e:
        print(f"✗ GIF generation test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("Running SAMOILOFF Effect Composer Integration Tests")
    print("=" * 50)
    
    tests = [
        test_app_import,
        test_basic_functions,
        test_image_processing,
        test_cmy_composite,
        test_gif_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Integration Tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All integration tests passed!")
        return True
    else:
        print("✗ Some integration tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
