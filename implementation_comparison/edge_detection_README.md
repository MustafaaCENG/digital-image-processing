# Edge Detection Algorithms

This project implements and compares various edge detection algorithms on different types of images. Edge detection is a fundamental image processing technique used to identify boundaries within digital images where the image brightness changes sharply.

## Algorithms Implemented

1. **Sobel Filter** - Uses two 3×3 kernels to approximate the derivatives of the image intensity function. One kernel detects horizontal edges, and the other detects vertical edges.

2. **Prewitt Filter** - Similar to Sobel but uses different kernels. It is also a discrete differentiation operator, computing an approximation of the gradient of the image intensity function.

3. **Roberts Filter** - One of the earliest edge detection operators. It performs a simple, quick to compute, 2-D spatial gradient measurement on an image.

4. **Laplacian of Gaussian (LoG)** - Combines Gaussian filtering with the Laplacian operator. The Gaussian filter reduces noise, and the Laplacian operator detects edges by finding areas where the second derivative of the intensity has a zero crossing.

5. **Canny Edge Detection** - Considered one of the best edge detection algorithms. It involves multiple stages including noise reduction, gradient calculation, non-maximum suppression, and hysteresis thresholding.

6. **Hough Transform** (Bonus) - Used for detecting straight lines in an image after edge detection.

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
The gradient magnitude is then calculated as: G = sqrt(Gx^2 + Gy^2)

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
The image is first convolved with a Gaussian filter to reduce noise, then with the Laplacian kernel.

### Canny Edge Detection
A multi-step algorithm that:
1. Applies Gaussian filter to smooth the image
2. Finds the intensity gradients
3. Applies non-maximum suppression
4. Applies double threshold to determine potential edges
5. Tracks edges by hysteresis: finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges

## Usage

### Requirements
- Python 3.6+
- NumPy
- Matplotlib
- scikit-image

### Installation
```bash
pip install -r requirements.txt
```

### Running the Code
```bash
python edge_detection_main.py --samples_dir samples/edges --results_dir results/edge_detection
```

Arguments:
- `--samples_dir`: Directory containing sample images (default: samples/edges)
- `--results_dir`: Directory to save results (default: results/edge_detection)

If no sample images are found, the program will automatically generate three test images:
1. Geometric shapes
2. Text-like patterns
3. Noisy gradient with shapes

### Output
The program generates:
- Individual edge-detected images for each algorithm and input image
- Comparison images showing all algorithms side by side
- A timing comparison chart
- Console output with computation times

## Results Analysis

The results from different edge detection algorithms can vary significantly based on:

1. **Image Content**: Some algorithms work better on specific types of images. For example:
   - Canny often performs well on natural images with complex edges
   - Sobel is effective for images with strong intensity gradients
   - LoG is good at finding zero crossings which correspond to edges

2. **Noise Sensitivity**: 
   - Roberts is most sensitive to noise
   - LoG and Canny include smoothing steps to reduce noise sensitivity

3. **Computational Efficiency**:
   - Roberts and Prewitt are computationally efficient
   - Canny is more complex and generally takes longer
   - LoG requires two operations (Gaussian smoothing and Laplacian)

## Project Structure
```
.
├── edge_detection.py        # Core implementation of edge detection algorithms
├── edge_detection_main.py   # Main script to run the algorithms
├── edge_detection_README.md # This file
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