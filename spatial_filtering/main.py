import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.util import random_noise

import spatial_filters as sf
import utils

def create_test_images_dir():
    """Create a directory for test images if it doesn't exist."""
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Created 'test_images' directory. Please place your test images there.")
        print("You can download sample images from the internet or use your own.")
        return False
    return True

def load_images(image_paths):
    """Load images from the specified paths."""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            # Read image and convert to grayscale if it's color
            img = io.imread(path)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = color.rgb2gray(img)
                img = (img * 255).astype(np.uint8)
            images.append(img)
        else:
            print(f"Warning: Image not found at {path}")
    return images

def gaussian_filtering_demo(image):
    """Demonstrate Gaussian filtering with different parameters."""
    print("\n=== Gaussian Filtering Demo ===")
    
    # Add Gaussian noise to the image
    noisy_image = sf.add_gaussian_noise(image, mean=0, sigma=25)
    
    # Apply Gaussian filter with different parameters
    filtered_small = sf.apply_gaussian_filter(noisy_image, kernel_size=3, sigma=1.0)
    filtered_medium = sf.apply_gaussian_filter(noisy_image, kernel_size=5, sigma=1.5)
    filtered_large = sf.apply_gaussian_filter(noisy_image, kernel_size=9, sigma=2.0)
    
    # Display results
    images = [image, noisy_image, filtered_small, filtered_medium, filtered_large]
    titles = [
        'Original Image', 
        'Noisy Image (Gaussian Noise)', 
        'Gaussian Filter (3x3, σ=1.0)', 
        'Gaussian Filter (5x5, σ=1.5)', 
        'Gaussian Filter (9x9, σ=2.0)'
    ]
    utils.display_images(images, titles, rows=2, cols=3)
    
    # Evaluate filters
    filtered_images = [filtered_small, filtered_medium, filtered_large]
    filter_names = ['Gaussian 3x3', 'Gaussian 5x5', 'Gaussian 9x9']
    results = utils.evaluate_filters(image, filtered_images, filter_names)
    
    # Print and plot evaluation results
    utils.print_evaluation_results(results)
    utils.plot_evaluation_results(results)
    
    return noisy_image, filtered_medium

def sharpening_demo(image):
    """Demonstrate image sharpening using unsharp masking."""
    print("\n=== Image Sharpening Demo ===")
    
    # Apply unsharp masking with different parameters
    sharpened_mild = sf.unsharp_masking(image, kernel_size=5, sigma=1.0, amount=0.5)
    sharpened_medium = sf.unsharp_masking(image, kernel_size=5, sigma=1.0, amount=1.0)
    sharpened_strong = sf.unsharp_masking(image, kernel_size=5, sigma=1.0, amount=2.0)
    
    # Display results
    images = [image, sharpened_mild, sharpened_medium, sharpened_strong]
    titles = [
        'Original Image', 
        'Sharpened (amount=0.5)', 
        'Sharpened (amount=1.0)', 
        'Sharpened (amount=2.0)'
    ]
    utils.display_images(images, titles, rows=2, cols=2)
    
    return sharpened_medium

def edge_detection_demo(image, sharpened_image):
    """Demonstrate edge detection algorithms."""
    print("\n=== Edge Detection Demo ===")
    
    # Apply edge detection to original image
    sobel_edges, sobel_magnitude, _ = sf.sobel_edge_detection(image)
    prewitt_edges, prewitt_magnitude = sf.prewitt_edge_detection(image)
    laplacian_edges = sf.laplacian_edge_detection(image)
    
    # Apply edge detection to sharpened image
    sobel_edges_sharp, sobel_magnitude_sharp, _ = sf.sobel_edge_detection(sharpened_image)
    prewitt_edges_sharp, _ = sf.prewitt_edge_detection(sharpened_image)
    laplacian_edges_sharp = sf.laplacian_edge_detection(sharpened_image)
    
    # Display results for original image
    images1 = [image, sobel_edges, prewitt_edges, laplacian_edges]
    titles1 = [
        'Original Image', 
        'Sobel Edge Detection', 
        'Prewitt Edge Detection', 
        'Laplacian Edge Detection'
    ]
    utils.display_images(images1, titles1, rows=2, cols=2)
    
    # Display results for sharpened image
    images2 = [sharpened_image, sobel_edges_sharp, prewitt_edges_sharp, laplacian_edges_sharp]
    titles2 = [
        'Sharpened Image', 
        'Sobel Edge Detection (Sharpened)', 
        'Prewitt Edge Detection (Sharpened)', 
        'Laplacian Edge Detection (Sharpened)'
    ]
    utils.display_images(images2, titles2, rows=2, cols=2)
    
    # Display gradient magnitudes
    images3 = [sobel_magnitude, sobel_magnitude_sharp]
    titles3 = ['Sobel Gradient Magnitude', 'Sobel Gradient Magnitude (Sharpened)']
    utils.display_images(images3, titles3, rows=1, cols=2)
    
    return sobel_edges

def median_filtering_demo(image):
    """Demonstrate median filtering for salt and pepper noise removal."""
    print("\n=== Median Filtering Demo ===")
    
    # Add salt and pepper noise
    noisy_image = sf.add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
    
    # Apply median filter with different kernel sizes
    median_3x3 = sf.median_filter(noisy_image, kernel_size=3)
    median_5x5 = sf.median_filter(noisy_image, kernel_size=5)
    
    # Apply Gaussian filter for comparison
    gaussian_filtered = sf.apply_gaussian_filter(noisy_image, kernel_size=5, sigma=1.5)
    
    # Display results
    images = [image, noisy_image, median_3x3, median_5x5, gaussian_filtered]
    titles = [
        'Original Image', 
        'Salt & Pepper Noise', 
        'Median Filter (3x3)', 
        'Median Filter (5x5)', 
        'Gaussian Filter (5x5)'
    ]
    utils.display_images(images, titles, rows=2, cols=3)
    
    # Plot histograms
    utils.plot_histograms(
        [image, noisy_image, median_3x3, median_5x5, gaussian_filtered],
        ['Original Histogram', 'Noisy Histogram', 'Median 3x3 Histogram', 
         'Median 5x5 Histogram', 'Gaussian Histogram'],
        rows=2, cols=3
    )
    
    # Evaluate filters
    filtered_images = [median_3x3, median_5x5, gaussian_filtered]
    filter_names = ['Median 3x3', 'Median 5x5', 'Gaussian 5x5']
    results = utils.evaluate_filters(image, filtered_images, filter_names)
    
    # Print and plot evaluation results
    utils.print_evaluation_results(results)
    utils.plot_evaluation_results(results)
    
    return noisy_image, median_5x5

def integrated_pipeline_demo(image):
    """Demonstrate an integrated image processing pipeline."""
    print("\n=== Integrated Pipeline Demo ===")
    
    # Step 1: Add salt and pepper noise
    noisy_image = sf.add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
    
    # Step 2: Apply median filtering to remove salt and pepper noise
    median_filtered = sf.median_filter(noisy_image, kernel_size=5)
    
    # Step 3: Apply Gaussian smoothing
    gaussian_filtered = sf.apply_gaussian_filter(median_filtered, kernel_size=5, sigma=1.0)
    
    # Step 4: Apply sharpening
    sharpened = sf.unsharp_masking(gaussian_filtered, kernel_size=5, sigma=1.0, amount=1.5)
    
    # Step 5: Apply edge detection
    edges, _, _ = sf.sobel_edge_detection(sharpened)
    
    # Display the pipeline steps
    images = [image, noisy_image, median_filtered, gaussian_filtered, sharpened, edges]
    titles = [
        'Original Image',
        'Step 1: Salt & Pepper Noise',
        'Step 2: Median Filtering',
        'Step 3: Gaussian Smoothing',
        'Step 4: Sharpening',
        'Step 5: Edge Detection'
    ]
    utils.display_images(images, titles, rows=2, cols=3)
    
    # Evaluate each step of the pipeline
    filtered_images = [median_filtered, gaussian_filtered, sharpened]
    filter_names = ['Median Filtering', 'Gaussian Smoothing', 'Sharpening']
    results = utils.evaluate_filters(image, filtered_images, filter_names)
    
    # Print and plot evaluation results
    utils.print_evaluation_results(results)
    utils.plot_evaluation_results(results)

def comparative_analysis(images, image_names):
    """Perform comparative analysis of filtering techniques on multiple images."""
    print("\n=== Comparative Analysis ===")
    
    for i, (image, name) in enumerate(zip(images, image_names)):
        print(f"\nAnalyzing image: {name}")
        
        # Add noise
        noisy_gaussian = sf.add_gaussian_noise(image, mean=0, sigma=20)
        noisy_sp = sf.add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
        
        # Apply filters
        gaussian_filtered = sf.apply_gaussian_filter(noisy_gaussian, kernel_size=5, sigma=1.5)
        median_filtered = sf.median_filter(noisy_sp, kernel_size=5)
        sharpened = sf.unsharp_masking(image, kernel_size=5, sigma=1.0, amount=1.5)
        edges, _, _ = sf.sobel_edge_detection(image)
        
        # Display results
        images_to_display = [
            image, noisy_gaussian, noisy_sp, 
            gaussian_filtered, median_filtered, sharpened, edges
        ]
        titles = [
            f'{name} - Original', 
            'Gaussian Noise', 
            'Salt & Pepper Noise',
            'Gaussian Filtered', 
            'Median Filtered', 
            'Sharpened', 
            'Edge Detection'
        ]
        utils.display_images(images_to_display, titles, rows=2, cols=4)
        
        # Evaluate filters
        filtered_images = [gaussian_filtered, median_filtered, sharpened]
        filter_names = ['Gaussian Filter', 'Median Filter', 'Sharpening']
        results = utils.evaluate_filters(image, filtered_images, filter_names)
        
        # Print and plot evaluation results
        utils.print_evaluation_results(results)
        utils.plot_evaluation_results(results)

def main():
    """Main function to run all demonstrations."""
    # Check if test_images directory exists
    if not create_test_images_dir():
        return
    
    # Look for images in the test_images directory
    image_paths = [os.path.join('test_images', f) for f in os.listdir('test_images') 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not image_paths:
        print("No images found in the 'test_images' directory.")
        print("Please add some images and run the script again.")
        return
    
    # Load images
    images = load_images(image_paths)
    image_names = [os.path.basename(path) for path in image_paths]
    
    if not images:
        print("Failed to load any images. Please check the image files.")
        return
    
    print(f"Loaded {len(images)} images: {', '.join(image_names)}")
    
    # Use the first image for individual demonstrations
    main_image = images[0]
    
    # Run demonstrations
    noisy_gaussian, gaussian_filtered = gaussian_filtering_demo(main_image)
    sharpened_image = sharpening_demo(main_image)
    edge_map = edge_detection_demo(main_image, sharpened_image)
    noisy_sp, median_filtered = median_filtering_demo(main_image)
    integrated_pipeline_demo(main_image)
    
    # Comparative analysis on all images
    comparative_analysis(images, image_names)
    
    print("\nAll demonstrations completed successfully!")

if __name__ == "__main__":
    main() 