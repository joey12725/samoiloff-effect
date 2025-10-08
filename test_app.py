import io
import unittest
import numpy as np
from PIL import Image
import tempfile
import os

# Import functions from app.py
from app import (
    srgb_to_linear, linear_to_srgb, clamp01,
    compute_canvas_pixels, contain_center_pad,
    apply_levels_exposure_linear, apply_contrast_brightness_linear,
    apply_clarity, apply_sharpen, to_grayscale,
    process_target, build_white_light_composite_srgb,
    generate_cycle_gif, LevelsParams, ContrastBrightnessParams,
    ClarityParams, SharpenParams, GrayMethodParams,
    tint_grayscale_to_color, generate_colored_cycle_gif
)


class TestColorSpaceConversions(unittest.TestCase):
    """Test sRGB ↔ linear color space conversions"""
    
    def test_srgb_to_linear_roundtrip(self):
        """Test that sRGB → linear → sRGB preserves values within tolerance"""
        # Test values
        test_values = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        
        for val in test_values:
            linear = srgb_to_linear(np.array([val]))
            back_to_srgb = linear_to_srgb(linear)
            self.assertAlmostEqual(val, back_to_srgb[0], places=6)
    
    def test_clamp01(self):
        """Test clamping function"""
        test_array = np.array([-0.5, 0.3, 1.2, 0.0, 1.0])
        clamped = clamp01(test_array)
        expected = np.array([0.0, 0.3, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(clamped, expected)


class TestCanvasOperations(unittest.TestCase):
    """Test canvas sizing and padding operations"""
    
    def test_compute_canvas_pixels(self):
        """Test canvas pixel calculation"""
        width_px, height_px = compute_canvas_pixels(6.0, 4.0, 300)
        self.assertEqual(width_px, 1800)
        self.assertEqual(height_px, 1200)
    
    def test_contain_center_pad_landscape(self):
        """Test contain+center padding for landscape image"""
        # Create a landscape test image (400x200)
        test_img = Image.new('RGB', (400, 200), (255, 0, 0))
        
        # Pad to square canvas (600x600)
        padded = contain_center_pad(test_img, (600, 600), "White")
        
        self.assertEqual(padded.size, (600, 600))
        self.assertEqual(padded.mode, 'RGB')
        
        # Check that image is centered
        # Should be 300x150 image centered in 600x600 canvas
        # Top-left corner should be white
        top_left = padded.getpixel((0, 0))
        self.assertEqual(top_left, (255, 255, 255))
        
        # Center should have the original red
        center = padded.getpixel((300, 300))
        self.assertEqual(center, (255, 0, 0))
    
    def test_contain_center_pad_portrait(self):
        """Test contain+center padding for portrait image"""
        # Create a portrait test image (200x400)
        test_img = Image.new('RGB', (200, 400), (0, 255, 0))
        
        # Pad to square canvas (600x600)
        padded = contain_center_pad(test_img, (600, 600), "White")
        
        self.assertEqual(padded.size, (600, 600))
        
        # Center should have the original green
        center = padded.getpixel((300, 300))
        self.assertEqual(center, (0, 255, 0))


class TestImageAdjustments(unittest.TestCase):
    """Test image adjustment functions"""
    
    def setUp(self):
        """Create test image data"""
        # Create a simple test image (10x10) with known values
        self.test_img = np.ones((10, 10, 3), dtype=np.float32) * 0.5  # Mid-gray
    
    def test_levels_exposure(self):
        """Test levels/exposure adjustment"""
        params = LevelsParams(black_point_pct=10, white_point_pct=90, gamma=1.0, auto_normalize=False)
        result = apply_levels_exposure_linear(self.test_img, params)
        
        # Should be normalized between 0.1 and 0.9
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
    
    def test_contrast_brightness(self):
        """Test contrast and brightness adjustment"""
        params = ContrastBrightnessParams(contrast=50, brightness=20)
        result = apply_contrast_brightness_linear(self.test_img, params)
        
        # Should still be in valid range
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))
    
    def test_clarity_zero(self):
        """Test clarity with zero amount (should be no-op)"""
        params = ClarityParams(radius_px=5.0, amount=0.0)
        result = apply_clarity(self.test_img, params)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(result, self.test_img, decimal=6)
    
    def test_sharpen_zero(self):
        """Test sharpen with zero amount (should be no-op)"""
        params = SharpenParams(amount=0, radius=1.0, threshold=0)
        result = apply_sharpen(self.test_img, params)
        
        # Should be unchanged
        np.testing.assert_array_almost_equal(result, self.test_img, decimal=6)


class TestGrayscaleConversion(unittest.TestCase):
    """Test grayscale conversion methods"""
    
    def setUp(self):
        """Create test RGB image"""
        # Create RGB image with known values
        self.test_img = np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # Red, Green
            [[0.0, 0.0, 1.0], [0.5, 0.5, 0.5]]   # Blue, Gray
        ], dtype=np.float32)
    
    def test_luminance_method(self):
        """Test luminance grayscale conversion"""
        params = GrayMethodParams("luminance", 1.0, 1.0, 1.0)
        result = to_grayscale(self.test_img, params)
        
        # Should be 2x2 grayscale
        self.assertEqual(result.shape, (2, 2))
        
        # Red pixel should have luminance value
        red_luminance = 0.2126 * 1.0 + 0.7152 * 0.0 + 0.0722 * 0.0
        self.assertAlmostEqual(result[0, 0], red_luminance, places=6)
    
    def test_max_method(self):
        """Test max grayscale conversion"""
        params = GrayMethodParams("max", 1.0, 1.0, 1.0)
        result = to_grayscale(self.test_img, params)
        
        # Red pixel should be 1.0 (max of [1,0,0])
        self.assertAlmostEqual(result[0, 0], 1.0, places=6)
        
        # Gray pixel should be 0.5 (max of [0.5,0.5,0.5])
        self.assertAlmostEqual(result[1, 1], 0.5, places=6)
    
    def test_average_method(self):
        """Test average grayscale conversion"""
        params = GrayMethodParams("average", 1.0, 1.0, 1.0)
        result = to_grayscale(self.test_img, params)
        
        # Gray pixel should be 0.5 (average of [0.5,0.5,0.5])
        self.assertAlmostEqual(result[1, 1], 0.5, places=6)
    
    def test_channel_mix_method(self):
        """Test channel mix grayscale conversion"""
        params = GrayMethodParams("channel_mix", 0.5, 0.3, 0.2)
        result = to_grayscale(self.test_img, params)
        
        # Red pixel: 0.5*1.0 + 0.3*0.0 + 0.2*0.0 = 0.5
        self.assertAlmostEqual(result[0, 0], 0.5, places=6)


class TestCMYMapping(unittest.TestCase):
    """Test CMY channel mapping and composite generation"""
    
    def test_white_light_composite(self):
        """Test white-light composite generation"""
        # Create test CMY channels
        c_channel = np.array([[0.5, 0.0], [1.0, 0.2]], dtype=np.float32)  # Cyan
        m_channel = np.array([[0.0, 0.8], [0.3, 0.0]], dtype=np.float32)  # Magenta  
        y_channel = np.array([[0.7, 0.4], [0.0, 1.0]], dtype=np.float32)  # Yellow
        
        # Generate composite
        composite = build_white_light_composite_srgb(c_channel, m_channel, y_channel)
        
        # Should be PIL Image
        self.assertIsInstance(composite, Image.Image)
        self.assertEqual(composite.size, (2, 2))
        
        # Convert back to array for testing
        arr = np.array(composite, dtype=np.float32) / 255.0
        
        # The composite applies sRGB encoding, so we need to convert back to linear
        # to compare with our expected linear values
        from app import srgb_to_linear
        
        # Test R = 1 - C (in linear domain)
        expected_r_linear = 1.0 - c_channel
        actual_r_linear = srgb_to_linear(arr[:, :, 0])
        np.testing.assert_array_almost_equal(actual_r_linear, expected_r_linear, decimal=2)
        
        # Test G = 1 - M (in linear domain)
        expected_g_linear = 1.0 - m_channel
        actual_g_linear = srgb_to_linear(arr[:, :, 1])
        np.testing.assert_array_almost_equal(actual_g_linear, expected_g_linear, decimal=2)
        
        # Test B = 1 - Y (in linear domain)
        expected_b_linear = 1.0 - y_channel
        actual_b_linear = srgb_to_linear(arr[:, :, 2])
        np.testing.assert_array_almost_equal(actual_b_linear, expected_b_linear, decimal=2)


class TestGIFGeneration(unittest.TestCase):
    """Test animated GIF generation"""
    
    def test_gif_generation(self):
        """Test basic GIF generation"""
        # Create simple test images
        r_img = Image.new('RGB', (100, 100), (255, 0, 0))
        g_img = Image.new('RGB', (100, 100), (0, 255, 0))
        b_img = Image.new('RGB', (100, 100), (0, 0, 255))
        
        # Generate GIF
        gif_bytes, duration = generate_cycle_gif(r_img, g_img, b_img, 0.1, 0.1, 10)
        
        # Should return bytes and duration
        self.assertIsInstance(gif_bytes, bytes)
        self.assertIsInstance(duration, float)
        self.assertGreater(len(gif_bytes), 0)
        self.assertGreater(duration, 0)


class TestColorization(unittest.TestCase):
    """Test grayscale tinting and colored GIF generation"""

    def test_tint_grayscale_to_color(self):
        # Create a white grayscale image
        img = Image.new('RGB', (10, 10), (255, 255, 255))
        # Tint to red
        tinted = tint_grayscale_to_color(img, (1.0, 0.0, 0.0))
        r, g, b = tinted.split()
        # Expect red channel to be 255, green and blue to be 0
        self.assertEqual(r.getpixel((5, 5)), 255)
        self.assertEqual(g.getpixel((5, 5)), 0)
        self.assertEqual(b.getpixel((5, 5)), 0)

    def test_generate_colored_cycle_gif(self):
        # Create simple grayscale previews (white and black regions)
        r_prev = Image.new('RGB', (20, 20), (255, 255, 255))
        g_prev = Image.new('RGB', (20, 20), (255, 255, 255))
        b_prev = Image.new('RGB', (20, 20), (255, 255, 255))
        gif_bytes, duration = generate_colored_cycle_gif(r_prev, g_prev, b_prev, 0.1, 0.1, 10)
        self.assertIsInstance(gif_bytes, bytes)
        self.assertGreater(len(gif_bytes), 0)
        self.assertIsInstance(duration, float)
        self.assertGreater(duration, 0)


class TestEndToEndProcessing(unittest.TestCase):
    """Test complete processing pipeline"""
    
    def test_process_target_basic(self):
        """Test basic target processing"""
        # Create a simple test image
        test_img = Image.new('RGB', (100, 100), (128, 128, 128))
        
        # Save to bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Process parameters
        levels = LevelsParams(0, 100, 1.0, False)
        contrast_brightness = ContrastBrightnessParams(0, 0)
        clarity = ClarityParams(0.0, 0.0)
        sharpen = SharpenParams(0, 1.0, 0)
        gray_params = GrayMethodParams("luminance", 1.0, 1.0, 1.0)
        
        # Process
        gray_channel, preview, padded = process_target(
            img_bytes.getvalue(),
            (200, 200),
            "White",
            levels, contrast_brightness, clarity, sharpen, gray_params
        )
        
        # Check outputs
        self.assertIsInstance(gray_channel, np.ndarray)
        self.assertEqual(gray_channel.shape, (200, 200))
        self.assertIsInstance(preview, Image.Image)
        self.assertIsInstance(padded, Image.Image)
        self.assertEqual(padded.size, (200, 200))


def run_tests():
    """Run all tests and report results"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestColorSpaceConversions,
        TestCanvasOperations,
        TestImageAdjustments,
        TestGrayscaleConversion,
        TestCMYMapping,
        TestGIFGeneration,
        TestEndToEndProcessing,
        TestColorization
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Report results
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
