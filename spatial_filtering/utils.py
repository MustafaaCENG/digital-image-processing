import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import os

def display_images(images, titles, rows, cols, figsize=(15, 10)):
    """
    Display multiple images in a grid layout.
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        figsize (tuple): Figure size (width, height)
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            if len(img.shape) == 2 or img.shape[2] == 1:  # Grayscale image
                axes[i].imshow(img, cmap='gray')
            else:  # Color image
                axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_histograms(images, titles, rows, cols, figsize=(15, 10), timestamp=None):
    """
    Plot histograms for multiple images.
    
    Args:
        images (list): List of images
        titles (list): List of titles for each histogram
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        figsize (tuple): Figure size (width, height)
        timestamp (str): Timestamp for saving the plot
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            if len(img.shape) == 3 and img.shape[2] == 3:  # Color image
                # Convert to grayscale for histogram
                gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                axes[i].hist(gray_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
            else:  # Grayscale image
                axes[i].hist(img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
            
            axes[i].set_title(title)
            axes[i].set_xlim([0, 256])
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if timestamp:
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(os.path.join('results', f'histograms_{timestamp}.png'))
    
    plt.show()

def calculate_psnr(original, filtered):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original (numpy.ndarray): Original image
        filtered (numpy.ndarray): Filtered image
        
    Returns:
        float: PSNR value in dB
    """
    return peak_signal_noise_ratio(original, filtered)

def calculate_ssim(original, filtered):
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        original (numpy.ndarray): Original image
        filtered (numpy.ndarray): Filtered image
        
    Returns:
        float: SSIM value (between -1 and 1, higher is better)
    """
    return structural_similarity(original, filtered, data_range=original.max() - original.min())

def evaluate_filters(original, filtered_images, filter_names):
    """
    Evaluate multiple filtered images using PSNR and SSIM metrics.
    
    Args:
        original (numpy.ndarray): Original image
        filtered_images (list): List of filtered images
        filter_names (list): List of filter names
        
    Returns:
        dict: Dictionary containing PSNR and SSIM values for each filter
    """
    results = {}
    
    for name, filtered in zip(filter_names, filtered_images):
        psnr = calculate_psnr(original, filtered)
        ssim = calculate_ssim(original, filtered)
        
        results[name] = {
            'PSNR': psnr,
            'SSIM': ssim
        }
    
    return results

def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results (dict): Dictionary containing PSNR and SSIM values for each filter
    """
    print("\nFilter Evaluation Results:")
    print("-" * 50)
    print(f"{'Filter Name':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("-" * 50)
    
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['PSNR']:>10.2f}  {metrics['SSIM']:>8.3f}")

def plot_evaluation_results(results, timestamp=None):
    """
    Plot evaluation results as bar charts.
    
    Args:
        results (dict): Dictionary containing PSNR and SSIM values for each filter
        timestamp (str): Timestamp for saving the plot
    """
    names = list(results.keys())
    psnr_values = [metrics['PSNR'] for metrics in results.values()]
    ssim_values = [metrics['SSIM'] for metrics in results.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # PSNR plot
    ax1.bar(names, psnr_values)
    ax1.set_title('PSNR Comparison')
    ax1.set_ylabel('PSNR (dB)')
    ax1.tick_params(axis='x', rotation=45)
    
    # SSIM plot
    ax2.bar(names, ssim_values)
    ax2.set_title('SSIM Comparison')
    ax2.set_ylabel('SSIM')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if timestamp:
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(os.path.join('results', f'metrics_comparison_{timestamp}.png'))
    
    plt.show() 