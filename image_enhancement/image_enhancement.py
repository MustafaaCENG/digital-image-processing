import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
from typing import Tuple, List, Optional


def load_image(image_path: str, as_gray: bool = True) -> np.ndarray:
    """
    Load an image from the specified path.
    
    Args:
        image_path: Path to the image file
        as_gray: Whether to load the image as grayscale
        
    Returns:
        Loaded image as a numpy array
    """
    if as_gray:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    
    if img is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    
    # Normalize to [0, 1] range if not already
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
        
    return img


def save_image(image: np.ndarray, save_path: str) -> None:
    """
    Save an image to the specified path.
    
    Args:
        image: Image to save (normalized to [0, 1])
        save_path: Path where to save the image
    """
    # Convert to uint8 for saving
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the image
    cv2.imwrite(save_path, image)


def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Args:
        image: Input image (normalized to [0, 1])
        gamma: Gamma value
        
    Returns:
        Gamma-corrected image
    """
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Apply gamma correction: s = r^gamma
    corrected = np.power(image, gamma)
    
    return corrected


def compute_histogram(image: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the histogram of an image.
    
    Args:
        image: Input image (normalized to [0, 1])
        bins: Number of bins for the histogram
        
    Returns:
        Tuple of (histogram, bin_edges)
    """
    # Scale to [0, bins-1] for histogram computation
    if image.max() <= 1.0:
        scaled_image = (image * (bins - 1)).astype(np.uint8)
    else:
        scaled_image = image.astype(np.uint8)
    
    # Compute histogram
    hist, bin_edges = np.histogram(scaled_image, bins=bins, range=(0, bins-1))
    
    return hist, bin_edges


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to an image.
    
    Args:
        image: Input image (normalized to [0, 1])
        
    Returns:
        Histogram-equalized image
    """
    # Ensure image is in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert to uint8 for histogram computation
    img_uint8 = (image * 255).astype(np.uint8)
    
    # Compute histogram
    hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=[0, 256])
    
    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize CDF to [0, 1]
    cdf_normalized = cdf / cdf.max()
    
    # Create lookup table for mapping
    lookup_table = (cdf_normalized * 255).astype(np.uint8)
    
    # Apply mapping to create equalized image
    equalized_img_uint8 = lookup_table[img_uint8]
    
    # Convert back to float [0, 1]
    equalized_img = equalized_img_uint8.astype(np.float32) / 255.0
    
    return equalized_img


def compare_with_builtin_equalization(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compare custom histogram equalization with OpenCV's built-in function.
    
    Args:
        image: Input image (normalized to [0, 1])
        
    Returns:
        Tuple of (custom_equalized, builtin_equalized)
    """
    # Custom implementation
    custom_equalized = histogram_equalization(image)
    
    # OpenCV's implementation
    img_uint8 = (image * 255).astype(np.uint8)
    builtin_equalized_uint8 = cv2.equalizeHist(img_uint8)
    builtin_equalized = builtin_equalized_uint8.astype(np.float32) / 255.0
    
    return custom_equalized, builtin_equalized


def plot_image_with_histogram(image: np.ndarray, title: str, ax=None) -> None:
    """
    Plot an image and its histogram.
    
    Args:
        image: Input image (normalized to [0, 1])
        title: Title for the plot
        ax: Matplotlib axes for plotting (optional)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title(f"{title} - Image")
    ax[0].axis('off')
    
    # Plot histogram
    hist, bin_edges = compute_histogram(image)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax[1].bar(bin_centers, hist, width=1, alpha=0.7)
    ax[1].set_title(f"{title} - Histogram")
    ax[1].set_xlabel("Pixel Intensity")
    ax[1].set_ylabel("Frequency")
    
    plt.tight_layout()


def process_and_compare_images(image_path: str, gamma_values: List[float], 
                              output_dir: str = "results") -> None:
    """
    Process an image with different gamma values and histogram equalization,
    and compare the results.
    
    Args:
        image_path: Path to the input image
        gamma_values: List of gamma values to apply
        output_dir: Directory to save the results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    original_img = load_image(image_path)
    
    # Get image name for saving results
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Gamma Correction
    fig, axes = plt.subplots(len(gamma_values) + 1, 2, figsize=(12, 5 * (len(gamma_values) + 1)))
    
    # Plot original image and histogram
    plot_image_with_histogram(original_img, "Original", axes[0])
    
    # Apply gamma correction with different values
    for i, gamma in enumerate(gamma_values):
        gamma_img = gamma_correction(original_img, gamma)
        plot_image_with_histogram(gamma_img, f"Gamma = {gamma}", axes[i + 1])
        
        # Save gamma-corrected image
        save_path = os.path.join(output_dir, f"{img_name}_gamma_{gamma}.png")
        save_image(gamma_img, save_path)
    
    plt.savefig(os.path.join(output_dir, f"{img_name}_gamma_comparison.png"))
    plt.close()
    
    # 2. Histogram Equalization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original image and histogram
    plot_image_with_histogram(original_img, "Original", axes[0])
    
    # Apply custom histogram equalization
    equalized_img = histogram_equalization(original_img)
    plot_image_with_histogram(equalized_img, "Histogram Equalization", axes[1])
    
    # Save equalized image
    save_path = os.path.join(output_dir, f"{img_name}_equalized.png")
    save_image(equalized_img, save_path)
    
    # Compare with built-in equalization
    custom_eq, builtin_eq = compare_with_builtin_equalization(original_img)
    
    # Calculate mean squared error between custom and built-in implementations
    mse = np.mean((custom_eq - builtin_eq) ** 2)
    print(f"MSE between custom and built-in equalization: {mse:.6f}")
    
    plt.savefig(os.path.join(output_dir, f"{img_name}_equalization.png"))
    plt.close()
    
    # 3. Combined Operations
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    
    # Plot original image and histogram
    plot_image_with_histogram(original_img, "Original", axes[0])
    
    # a. Gamma correction followed by histogram equalization
    gamma_value = 1.5  # Choose a gamma value
    gamma_img = gamma_correction(original_img, gamma_value)
    gamma_then_eq_img = histogram_equalization(gamma_img)
    plot_image_with_histogram(gamma_then_eq_img, f"Gamma ({gamma_value}) -> Equalization", axes[1])
    
    # Save combined image
    save_path = os.path.join(output_dir, f"{img_name}_gamma_then_eq.png")
    save_image(gamma_then_eq_img, save_path)
    
    # b. Histogram equalization followed by gamma correction
    eq_img = histogram_equalization(original_img)
    eq_then_gamma_img = gamma_correction(eq_img, gamma_value)
    plot_image_with_histogram(eq_then_gamma_img, f"Equalization -> Gamma ({gamma_value})", axes[2])
    
    # Save combined image
    save_path = os.path.join(output_dir, f"{img_name}_eq_then_gamma.png")
    save_image(eq_then_gamma_img, save_path)
    
    plt.savefig(os.path.join(output_dir, f"{img_name}_combined_operations.png"))
    plt.close()
    
    # Calculate statistics for comparison
    stats = {
        "Original": {
            "Mean": np.mean(original_img),
            "Std Dev": np.std(original_img),
            "Min": np.min(original_img),
            "Max": np.max(original_img)
        },
        "Gamma Correction": {
            "Mean": np.mean(gamma_img),
            "Std Dev": np.std(gamma_img),
            "Min": np.min(gamma_img),
            "Max": np.max(gamma_img)
        },
        "Histogram Equalization": {
            "Mean": np.mean(equalized_img),
            "Std Dev": np.std(equalized_img),
            "Min": np.min(equalized_img),
            "Max": np.max(equalized_img)
        },
        "Gamma -> Equalization": {
            "Mean": np.mean(gamma_then_eq_img),
            "Std Dev": np.std(gamma_then_eq_img),
            "Min": np.min(gamma_then_eq_img),
            "Max": np.max(gamma_then_eq_img)
        },
        "Equalization -> Gamma": {
            "Mean": np.mean(eq_then_gamma_img),
            "Std Dev": np.std(eq_then_gamma_img),
            "Min": np.min(eq_then_gamma_img),
            "Max": np.max(eq_then_gamma_img)
        }
    }
    
    # Print statistics
    print(f"\nStatistics for {img_name}:")
    for method, values in stats.items():
        print(f"  {method}:")
        for stat, value in values.items():
            print(f"    {stat}: {value:.4f}")
    
    return stats


def process_color_image(image_path: str, output_dir: str = "results") -> None:
    """
    Process a color image using different approaches.
    
    Args:
        image_path: Path to the input color image
        output_dir: Directory to save the results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load color image
    original_img = load_image(image_path, as_gray=False)
    
    # Get image name for saving results
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Process each channel separately
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot original color image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original Color Image")
    axes[0, 0].axis('off')
    
    # Process each channel separately
    channels = []
    for i in range(3):  # RGB channels
        channel = original_img[:, :, i]
        equalized_channel = histogram_equalization(channel)
        channels.append(equalized_channel)
    
    # Combine channels
    rgb_equalized = np.stack(channels, axis=2)
    
    # Plot RGB equalized image
    axes[0, 1].imshow(rgb_equalized)
    axes[0, 1].set_title("RGB Channels Equalized Separately")
    axes[0, 1].axis('off')
    
    # 2. Convert to HSV and equalize V channel
    hsv_img = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Equalize V channel
    v_eq = (histogram_equalization(v / 255.0) * 255).astype(np.uint8)
    
    # Merge channels
    hsv_eq = cv2.merge([h, s, v_eq])
    
    # Convert back to RGB
    hsv_equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB) / 255.0
    
    # Plot HSV equalized image
    axes[1, 0].imshow(hsv_equalized)
    axes[1, 0].set_title("HSV Value Channel Equalized")
    axes[1, 0].axis('off')
    
    # 3. Apply gamma correction to each RGB channel
    gamma_value = 1.5
    gamma_channels = []
    for i in range(3):  # RGB channels
        channel = original_img[:, :, i]
        gamma_channel = gamma_correction(channel, gamma_value)
        gamma_channels.append(gamma_channel)
    
    # Combine channels
    rgb_gamma = np.stack(gamma_channels, axis=2)
    
    # Plot gamma-corrected image
    axes[1, 1].imshow(rgb_gamma)
    axes[1, 1].set_title(f"RGB Gamma Correction (γ={gamma_value})")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_color_processing.png"))
    plt.close()
    
    # Save processed images
    save_image((rgb_equalized * 255).astype(np.uint8), 
               os.path.join(output_dir, f"{img_name}_rgb_equalized.png"))
    save_image((hsv_equalized * 255).astype(np.uint8), 
               os.path.join(output_dir, f"{img_name}_hsv_equalized.png"))
    save_image((rgb_gamma * 255).astype(np.uint8), 
               os.path.join(output_dir, f"{img_name}_rgb_gamma.png"))


def main():
    """
    Main function to run the image enhancement operations.
    """
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define gamma values to test
    gamma_values = [0.5, 1.0, 1.5, 2.0]
    
    # Process grayscale images
    image_paths = [
        "images/underexposed.jpg",  # Underexposed image
        "images/overexposed.jpg",   # Overexposed image
        "images/balanced.jpg"       # Well-balanced image
    ]
    
    # Check if images exist, if not, print a message
    missing_images = [path for path in image_paths if not os.path.exists(path)]
    if missing_images:
        print("The following images are missing:")
        for path in missing_images:
            print(f"  - {path}")
        print("\nPlease add these images to the 'images' directory before running the script.")
        return
    
    # Process each image
    all_stats = {}
    for image_path in image_paths:
        print(f"\nProcessing {image_path}...")
        stats = process_and_compare_images(image_path, gamma_values, results_dir)
        all_stats[os.path.basename(image_path)] = stats
    
    # Process color images (bonus task)
    color_image_paths = [
        "images/color_underexposed.jpg",
        "images/color_overexposed.jpg",
        "images/color_balanced.jpg"
    ]
    
    # Check if color images exist
    missing_color_images = [path for path in color_image_paths if not os.path.exists(path)]
    if not missing_color_images:
        print("\nProcessing color images...")
        for image_path in color_image_paths:
            print(f"\nProcessing {image_path}...")
            process_color_image(image_path, results_dir)
    else:
        print("\nSkipping color image processing as some images are missing.")
    
    print("\nImage enhancement operations completed. Results saved in the 'results' directory.")


if __name__ == "__main__":
    main() 