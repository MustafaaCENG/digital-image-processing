# Image Enhancement using Point Operations

This project implements various point processing techniques for image enhancement, focusing on gamma correction and histogram equalization. The implementation allows for analyzing the effects of these operations on different types of images both quantitatively and visually.

## Features

- **Gamma Correction**: Adjust image brightness and contrast using different gamma values
- **Histogram Equalization**: Redistribute pixel intensities to enhance contrast
- **Combined Operations**: Apply operations in different sequences and analyze the effects
- **Color Image Processing**: Process color images using different approaches (RGB channels separately or HSV color space)
- **Statistical Analysis**: Calculate and compare image statistics before and after processing

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- OpenCV (cv2)
- scikit-image

You can install the required packages using pip:

```bash
pip install numpy matplotlib opencv-python scikit-image
```

## Usage

1. Place your test images in the `images` directory:
   - For grayscale processing: `underexposed.jpg`, `overexposed.jpg`, `balanced.jpg`
   - For color processing: `color_underexposed.jpg`, `color_overexposed.jpg`, `color_balanced.jpg`

2. Run the main script:

```bash
python image_enhancement.py
```

3. The results will be saved in the `results` directory, including:
   - Processed images
   - Comparison plots with histograms
   - Statistical data printed to the console

## Implementation Details

### Gamma Correction

The gamma correction function applies the power-law transformation to adjust image brightness:

```
output = input^gamma
```

- Gamma < 1: Brightens darker regions (useful for underexposed images)
- Gamma > 1: Darkens brighter regions (useful for overexposed images)
- Gamma = 1: No change

### Histogram Equalization

The histogram equalization function redistributes pixel intensities to enhance contrast:

1. Compute the histogram of the input image
2. Calculate the cumulative distribution function (CDF)
3. Normalize the CDF to create a mapping function
4. Apply the mapping to create the equalized image

### Combined Operations

The program applies operations in different sequences:
- Gamma correction followed by histogram equalization
- Histogram equalization followed by gamma correction

### Color Image Processing

For color images, the program provides two approaches:
1. Process each RGB channel separately
2. Convert to HSV color space and apply histogram equalization to the V (value) channel only

## Results

The program generates:
- Individual processed images
- Comparison plots showing original and processed images with their histograms
- Statistical data for quantitative analysis

## Example Output

The console output includes statistical information for each image and processing method:

```
Statistics for underexposed:
  Original:
    Mean: 0.1234
    Std Dev: 0.0567
    Min: 0.0000
    Max: 0.7890
  Gamma Correction:
    Mean: 0.3456
    ...
```

## License

This project is open source and available under the MIT License. 