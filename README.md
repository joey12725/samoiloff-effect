# SAMOILOFF Effect Composer

A sophisticated Streamlit application for creating multi-color lithophanes using the **Samoiloff Effect** - a technique that combines subtractive color mixing with 3D printing to produce stunning color lithophanes that reveal different images when illuminated with different colored lights.

## What is the Samoiloff Effect?

The Samoiloff Effect is a revolutionary approach to creating color lithophanes that leverages the principles of **subtractive color mixing** in physical materials. Unlike traditional lithophanes that show only grayscale images, this technique allows you to create lithophanes that display different colored images depending on the color of light shining through them.

### The Science Behind It

**Subtractive Color Mixing**: When light passes through colored materials, each material absorbs (subtracts) certain wavelengths and transmits others. The three primary subtractive colors are:
- **Cyan (C)**: Absorbs red light, transmits blue and green
- **Magenta (M)**: Absorbs green light, transmits red and blue  
- **Yellow (Y)**: Absorbs blue light, transmits red and green

**How It Works**: By strategically varying the thickness of cyan, magenta, and yellow materials in a lithophane, you can control how much of each color is absorbed. When white light passes through:
- Areas with more cyan material appear cyan (white - red = cyan)
- Areas with more magenta material appear magenta (white - green = magenta)
- Areas with more yellow material appear yellow (white - blue = yellow)
- Areas with all three materials appear black (white - red - green - blue = black)

**The Magic**: When illuminated with colored lights instead of white light:
- **Red light**: Only cyan areas are visible (red light can't pass through cyan)
- **Green light**: Only magenta areas are visible (green light can't pass through magenta)
- **Blue light**: Only yellow areas are visible (blue light can't pass through yellow)

This allows you to create lithophanes that show completely different images depending on the illumination color!

## How This Application Works

### Input Processing
1. **Three Target Images**: Upload separate images for Red, Green, and Blue channels
2. **Image Processing**: Each image is converted to grayscale using various methods (luminance, max, average, or custom channel mixing)
3. **CMY Mapping**: The grayscale values are mapped to subtractive colors:
   - Red target → Cyan channel (C = 1 - R_gray)
   - Green target → Magenta channel (M = 1 - G_gray)  
   - Blue target → Yellow channel (Y = 1 - B_gray)

### Color Mixing Algorithm
The application uses **continuous subtractive mixing** with density curves to simulate how light behaves when passing through colored materials:

```python
# Density curve: c' = 1 - (1 - c)^k
# Where k controls the density response (default: 1.5)
# Higher k = more dramatic color separation
```

### Why This Works
1. **Physical Accuracy**: The algorithm models real subtractive color mixing that occurs in 3D printed materials
2. **Linear Light Processing**: All calculations are performed in linear light space for accurate color reproduction
3. **Density Curves**: The density parameter controls how aggressively colors separate, matching real-world material behavior
4. **Proper Color Space**: sRGB input/output with linear light processing ensures accurate color representation

## Features

### Core Functionality
- **300 DPI Canvas**: High-resolution output suitable for detailed 3D printing
- **Flexible Canvas Sizing**: Adjustable dimensions with automatic pixel calculation
- **Smart Padding**: White or transparent padding with contain-and-center fitting
- **Real-time Previews**: Live preview of each channel and final composite

### Advanced Image Controls
Each target image has independent controls:
- **Levels/Exposure**: Black point, white point, gamma correction, auto-normalization
- **Contrast & Brightness**: Fine-tune image appearance
- **Clarity**: Local contrast enhancement for better detail
- **Sharpening**: Unsharp mask with adjustable radius, amount, and threshold
- **Grayscale Conversion**: Multiple methods (luminance, max, average, custom channel mixing)

### Export Options
- **Final Composite**: CMY-mapped image ready for lithophane generation
- **Channel Previews**: Individual grayscale previews for each channel
- **Colored Previews**: RGB-tinted versions showing how each channel will appear
- **Animated GIFs**: RGB cycling animations showing the effect in action
- **Multiple Formats**: PNG with optional transparency support

### Animation Features
- **Configurable Timing**: Adjustable hold and fade durations
- **Frame Rate Control**: 1-60 FPS for smooth or efficient animations
- **Linear Light Blending**: Proper color mixing during transitions
- **Dual Formats**: Grayscale and colored versions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone or download the repository
cd SAMOILOFF-EFFECT-Streamlit-App

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- `streamlit`: Web application framework
- `numpy`: Numerical computations
- `PIL` (Pillow): Image processing
- `imageio`: GIF generation

## Usage

### Basic Workflow
1. **Start the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Upload Images**: Use the sidebar to upload three images:
   - R-target: Image that will appear under red light
   - G-target: Image that will appear under green light  
   - B-target: Image that will appear under blue light

3. **Adjust Settings**: Configure canvas size, padding mode, and output directory

4. **Fine-tune Images**: Use the per-image controls to optimize each channel:
   - Adjust levels and exposure for proper contrast
   - Enhance clarity and sharpness for better detail
   - Choose appropriate grayscale conversion method

5. **Preview Results**: View real-time previews in the tabbed interface:
   - Individual channel previews
   - Final white-light composite
   - Animated RGB cycling

6. **Export**: Click "Export All" to save all files to your output directory

### Advanced Usage

#### Optimizing for 3D Printing
- **Canvas Size**: Choose dimensions that match your 3D printer's build volume
- **DPI**: 300 DPI provides excellent detail for most printers
- **Padding**: Use transparent padding for seamless integration with other objects

#### Color Channel Selection
- **High Contrast**: Choose images with strong contrast for dramatic effects
- **Complementary Colors**: Select images that work well together when mixed
- **Detail Level**: Consider the level of detail visible in each color channel

#### Animation Settings
- **Hold Duration**: Longer holds (1.5-2s) emphasize each color
- **Fade Duration**: Shorter fades (0.5-1s) create smooth transitions
- **Frame Rate**: 24 FPS provides smooth animation, 12 FPS reduces file size

## Technical Details

### Color Space Handling
- **Input**: sRGB images (standard web/display format)
- **Processing**: Linear light space for accurate color mixing
- **Output**: sRGB for display and export

### Image Processing Pipeline
1. **Load & Convert**: Convert input images to RGB/RGBA
2. **Resize & Pad**: Fit to canvas with contain-and-center algorithm
3. **Linearize**: Convert from sRGB to linear light space
4. **Apply Adjustments**: Levels, contrast, clarity, sharpening
5. **Grayscale Conversion**: Convert to single channel using selected method
6. **CMY Mapping**: Invert grayscale values for subtractive color mapping
7. **Composite**: Apply subtractive mixing with density curves
8. **Export**: Convert back to sRGB for final output

### Performance Optimizations
- **Caching**: Streamlit caching for expensive operations
- **Efficient Algorithms**: Vectorized NumPy operations
- **Memory Management**: Proper image format handling
- **Lazy Loading**: Process images only when needed

## File Outputs

When you export, the application creates:

### Core Files
- `final_white_light.png`: Main composite image for lithophane generation
- `preview_R.png`: Grayscale preview of red channel
- `preview_G.png`: Grayscale preview of green channel  
- `preview_B.png`: Grayscale preview of blue channel

### Colored Previews
- `preview_R_colored.png`: Red-tinted preview
- `preview_G_colored.png`: Green-tinted preview
- `preview_B_colored.png`: Blue-tinted preview

### Animations
- `rgb_cycle.gif`: Grayscale RGB cycling animation
- `rgb_cycle_colored.gif`: Colored RGB cycling animation

## Troubleshooting

### Common Issues
- **Upload Errors**: Ensure images are PNG or JPEG format
- **Memory Issues**: Reduce canvas size for large images
- **Export Failures**: Check output directory permissions
- **GIF Generation**: Ensure imageio is properly installed

### Performance Tips
- **Canvas Size**: Smaller canvases process faster
- **Image Resolution**: Lower resolution images process quicker
- **Caching**: Clear Streamlit cache if experiencing issues

## Applications

### Creative Uses
- **Artistic Lithophanes**: Create stunning multi-color art pieces
- **Educational Tools**: Demonstrate subtractive color mixing principles
- **Decorative Objects**: Unique lighting fixtures and decorations
- **Photography**: Creative photo presentations

### Technical Applications
- **Color Theory Education**: Visual demonstration of color mixing
- **3D Printing**: Advanced lithophane generation
- **Lighting Design**: Custom colored lighting effects
- **Material Science**: Understanding subtractive color properties

## Contributing

This project welcomes contributions! Areas for improvement:
- Additional image processing algorithms
- More grayscale conversion methods
- Enhanced animation options
- 3D printer integration
- Performance optimizations

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- **Samoiloff Effect**: Based on the innovative lithophane technique
- **Streamlit**: Web application framework
- **PIL/Pillow**: Image processing capabilities
- **NumPy**: Numerical computing foundation
- **ImageIO**: Animation generation support

---

*Create stunning multi-color lithophanes that reveal different images under different colored lights - the magic of subtractive color mixing brought to life through 3D printing!*
