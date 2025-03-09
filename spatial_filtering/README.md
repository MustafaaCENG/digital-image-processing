# Spatial Filtering Techniques

This project implements various spatial filtering techniques for image processing, including Gaussian smoothing, image sharpening, edge detection, and median filtering. The implementation emphasizes practical application, parameter analysis, and quantitative evaluation.

## Features

- **Gaussian Filtering**: Reduce noise using Gaussian kernels with adjustable parameters
- **Image Sharpening**: Enhance details using unsharp masking technique
- **Edge Detection**: Detect edges using Sobel, Prewitt, and Laplacian operators
- **Median Filtering**: Remove salt-and-pepper noise while preserving edges
- **Integrated Pipeline**: Sequential processing combining multiple techniques
- **Comparative Analysis**: Evaluate and compare different filtering techniques

## Requirements

- Python 3.6+
- NumPy
- OpenCV (cv2)
- Matplotlib
- scikit-image

## Installation

1. Clone this repository or download the source code.
2. Install the required dependencies:

```bash
pip install numpy opencv-python matplotlib scikit-image
```

## Usage

1. Create a directory named `test_images` in the project folder (if it doesn't exist already).
2. Place your test images in the `test_images` directory. The program supports common image formats (PNG, JPG, BMP, TIFF).
3. Run the main script:

```bash
python main.py
```

The program will:
- Load all images from the `test_images` directory
- Convert color images to grayscale
- Run demonstrations of each filtering technique
- Display results with side-by-side comparisons
- Provide quantitative evaluation using PSNR and SSIM metrics

## Project Structure

- `main.py`: Main script that demonstrates all filtering techniques
- `spatial_filters.py`: Module containing implementations of all filtering functions
- `utils.py`: Utility functions for visualization and metrics calculation
- `README.md`: This file

## Implementation Details

### Gaussian Filtering
- Custom implementation of Gaussian kernel generation
- Convolution-based filtering with adjustable kernel size and standard deviation
- Evaluation of noise reduction performance

### Image Sharpening
- Unsharp masking technique that enhances details
- Adjustable sharpening strength
- Visual comparison of different parameter settings

### Edge Detection
- Implementation of Sobel, Prewitt, and Laplacian edge detection
- Comparison of edge detection on original and sharpened images
- Visualization of gradient magnitudes

### Median Filtering
- Non-linear filtering for salt-and-pepper noise removal
- Comparison with Gaussian filtering for noise reduction
- Histogram analysis before and after filtering

### Integrated Pipeline
- Sequential application of multiple filtering techniques
- Evaluation of each step's contribution to the final result
- Quantitative assessment of the complete pipeline

## Metrics

The program evaluates filtering performance using:
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures the quality of filtered images compared to the original
- **Structural Similarity Index (SSIM)**: Assesses the perceived quality of filtered images

## License

This project is provided for educational purposes. Feel free to use and modify the code for your own projects.

## Acknowledgments

This project was developed as part of an image processing assignment focusing on spatial filtering techniques. 