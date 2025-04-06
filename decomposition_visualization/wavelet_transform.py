import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage import io, color, metrics
from pathlib import Path


def load_image(image_path):
    """
    Load an image and convert it to grayscale
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Grayscale image as a 2D numpy array
    """
    # Read the image
    img = io.imread(image_path)
    
    # Convert to grayscale if the image is RGB
    if len(img.shape) > 2:
        img = color.rgb2gray(img)
        
    return img


def wavelet_decomposition(image, wavelet='haar', level=1):
    """
    Perform wavelet decomposition on an image
    
    Args:
        image: Input grayscale image
        wavelet: Wavelet type (e.g., 'haar', 'db2')
        level: Decomposition level
        
    Returns:
        Coefficients from the wavelet transform
    """
    # Perform the wavelet transform
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    return coeffs


def visualize_coefficients(coeffs, level=1, title=None):
    """
    Visualize the wavelet coefficients
    
    Args:
        coeffs: Coefficients from wavelet decomposition
        level: Decomposition level to visualize (1-indexed)
        title: Title for the figure
    """
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Ensure the requested level does not exceed the available levels
    max_level = len(coeffs) - 1
    if level > max_level:
        level = max_level
        print(f"Warning: Requested level {level} exceeds available levels. Using level {max_level} instead.")
    
    # Plot the approximation coefficients
    cA = coeffs[0]
    plt.subplot(2, 2, 1)
    plt.imshow(cA, cmap='gray')
    plt.title('Approximation (LL)')
    plt.axis('off')
    
    # Plot the detail coefficients for the specified level
    if level >= 1 and level <= max_level:
        (cH, cV, cD) = coeffs[level]
        
        # Horizontal detail
        plt.subplot(2, 2, 2)
        plt.imshow(cH, cmap='gray')
        plt.title(f'Horizontal Detail (LH) - Level {level}')
        plt.axis('off')
        
        # Vertical detail
        plt.subplot(2, 2, 3)
        plt.imshow(cV, cmap='gray')
        plt.title(f'Vertical Detail (HL) - Level {level}')
        plt.axis('off')
        
        # Diagonal detail
        plt.subplot(2, 2, 4)
        plt.imshow(cD, cmap='gray')
        plt.title(f'Diagonal Detail (HH) - Level {level}')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save the figure
    plt.savefig(f'results/{title}.png', dpi=300)
    plt.show()


def reconstruct_image(coeffs, wavelet='haar'):
    """
    Reconstruct the image from wavelet coefficients
    
    Args:
        coeffs: Coefficients from wavelet decomposition
        wavelet: Wavelet type used for decomposition
        
    Returns:
        Reconstructed image
    """
    # Perform the inverse wavelet transform
    reconstructed = pywt.waverec2(coeffs, wavelet)
    
    return reconstructed


def calculate_metrics(original, reconstructed):
    """
    Calculate image quality metrics between original and reconstructed images
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        
    Returns:
        Dictionary containing MSE and PSNR values
    """
    # Ensure the images have the same shape
    min_shape = [min(s1, s2) for s1, s2 in zip(original.shape, reconstructed.shape)]
    original_cropped = original[:min_shape[0], :min_shape[1]]
    reconstructed_cropped = reconstructed[:min_shape[0], :min_shape[1]]
    
    # Calculate MSE
    mse = metrics.mean_squared_error(original_cropped, reconstructed_cropped)
    
    # Calculate PSNR
    psnr = metrics.peak_signal_noise_ratio(original_cropped, reconstructed_cropped)
    
    return {
        'MSE': mse,
        'PSNR': psnr
    }


def threshold_coefficients(coeffs, threshold=0.1):
    """
    Apply thresholding to wavelet coefficients
    
    Args:
        coeffs: Coefficients from wavelet decomposition
        threshold: Threshold value for coefficient filtering
        
    Returns:
        Thresholded coefficients
    """
    # Create a copy of the coefficients
    thresholded_coeffs = [coeffs[0].copy()]  # Keep approximation coefficients unchanged
    
    # Apply thresholding to detail coefficients
    for i in range(1, len(coeffs)):
        detail_coeffs = []
        for detail in coeffs[i]:
            # Apply soft thresholding
            thresholded = pywt.threshold(detail, threshold, 'soft')
            detail_coeffs.append(thresholded)
        thresholded_coeffs.append(tuple(detail_coeffs))
    
    return thresholded_coeffs


def compare_images(original, reconstructed, thresholded=None, title=None):
    """
    Compare original and reconstructed images
    
    Args:
        original: Original image
        reconstructed: Reconstructed image
        thresholded: Reconstructed image after thresholding
        title: Title for the figure
    """
    plt.figure(figsize=(15, 5))
    
    # Plot the original image
    plt.subplot(1, 3 if thresholded is not None else 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot the reconstructed image
    plt.subplot(1, 3 if thresholded is not None else 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    # Plot the thresholded reconstructed image if provided
    if thresholded is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(thresholded, cmap='gray')
        plt.title('Thresholded Reconstruction')
        plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save the figure
    plt.savefig(f'results/comparison_{title}.png', dpi=300)
    plt.show() 