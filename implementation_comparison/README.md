# Edge Detection Algorithm Comparison

<div align="center">
  <img src="results/edge_detection/geometric_shapes_comparison.png" alt="Edge Detection Comparison" width="800">
</div>

## Overview

This project implements and compares various edge detection algorithms on different types of images. Edge detection is a fundamental image processing technique used to identify boundaries within digital images where the image brightness changes sharply.

## Implemented Algorithms

| Algorithm | Description | Characteristics |
|-----------|-------------|-----------------|
| **Sobel** | Uses two 3×3 kernels to approximate image gradients | Good for detecting strong edges with some noise suppression |
| **Prewitt** | Similar to Sobel but with uniform weights | Simpler computation with adequate performance |
| **Roberts** | Uses 2×2 kernels for fast computation | Simple but more sensitive to noise |
| **Laplacian of Gaussian (LoG)** | Combines Gaussian filtering with Laplacian operator | Good at finding edge zero crossings |
| **Canny** | Multi-stage algorithm with hysteresis thresholding | Considered optimal for most applications |

## Visual Results

### Sample: Text Image
<div align="center">
  <img src="results/edge_detection/text_sample_comparison.png" alt="Text Sample Comparison" width="800">
</div>

### Sample: Geometric Shapes
<div align="center">
  <img src="results/edge_detection/geometric_shapes_comparison.png" alt="Geometric Shapes Comparison" width="800">
</div>

### Sample: Noisy Gradient
<div align="center">
  <img src="results/edge_detection/noisy_gradient_comparison.png" alt="Noisy Gradient Comparison" width="800">
</div>

## Performance Comparison

The performance of each algorithm varies depending on the image type and content. Below is a computation time comparison:

<div align="center">
  <img src="results/edge_detection/timing_comparison.png" alt="Timing Comparison" width="800">
</div>

## Mathematical Foundations

### Sobel Filter
Uses two 3×3 kernels to compute the gradient in the x and y directions:
```
Gx = [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]]

Gy = [[-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1]]
```
The gradient magnitude is calculated as: G = sqrt(Gx² + Gy²)

### Prewitt Filter
Similar to Sobel but with uniform weights:
```
Gx = [[-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1]]

Gy = [[-1, -1, -1],
      [0, 0, 0],
      [1, 1, 1]]
```

### Roberts Filter
Uses 2×2 kernels:
```
Gx = [[1, 0],
      [0, -1]]

Gy = [[0, 1],
      [-1, 0]]
```

### Laplacian of Gaussian (LoG)
Combines Gaussian smoothing with the Laplacian operator. The Laplacian operator is:
```
L = [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]]
```

### Canny Edge Detection
A multi-stage algorithm that:
1. Applies Gaussian filter to smooth the image
2. Finds the intensity gradients
3. Applies non-maximum suppression
4. Applies double threshold to determine potential edges
5. Tracks edges by hysteresis

## Installation and Usage

### Requirements
- Python 3.6+
- NumPy
- Matplotlib
- scikit-image
- SciPy

### Installation
```bash
pip install numpy matplotlib scikit-image scipy
```

### Running the Code
```bash
python edge_detection_main.py --samples_dir samples/edges --results_dir results/edge_detection
```

Arguments:
- `--samples_dir`: Directory containing sample images (default: samples/edges)
- `--results_dir`: Directory to save results (default: results/edge_detection)

## Algorithm Selection Guide

| Image Type | Recommended Algorithm | Why |
|------------|----------------------|-----|
| Natural images with noise | Canny | Best noise handling with good edge detection |
| Clean geometric shapes | Roberts/Sobel | Fast and accurate for simple shapes |
| Medical/scientific images | LoG | Good at detecting continuous boundaries |
| Text/document images | Prewitt/Sobel | Good balance of speed and accuracy |

## Project Structure
```
.
├── edge_detection.py        # Core implementation of edge detection algorithms
├── edge_detection_main.py   # Main script to run the algorithms
├── README.md                # This file
├── samples/                 # Sample images
│   └── edges/               # Sample images for edge detection
└── results/                 # Generated results
    └── edge_detection/      # Edge detection results
```

## Future Improvements
- Implement more advanced edge detection algorithms (e.g., SUSAN, edge drawing)
- Add interactive parameter tuning
- Implement quantitative evaluation metrics for edge detection quality
- Support for video edge detection

## References
1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6), 679-698.
2. Marr, D., & Hildreth, E. (1980). Theory of Edge Detection. Proceedings of the Royal Society of London, 207(1167), 187-217.
3. Duda, R. O., & Hart, P. E. (1972). Use of the Hough Transformation to Detect Lines and Curves in Pictures. Communications of the ACM, 15(1), 11-15. 