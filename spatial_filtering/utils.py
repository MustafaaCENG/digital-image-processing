import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

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

def plot_histograms(images, titles, rows, cols, figsize=(15, 10)):
    """
    Plot histograms for multiple images.
    
    Args:
        images (list): List of images
        titles (list): List of titles for each histogram
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        figsize (tuple): Figure size (width, height)
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
    Print evaluation results in a formatted table.
    
    Args:
        results (dict): Dictionary containing evaluation results
    """
    print("\nQuantitative Evaluation Results:")
    print("-" * 60)
    print(f"{'Filter':<20} | {'PSNR (dB)':<15} | {'SSIM':<15}")
    print("-" * 60)
    
    for filter_name, metrics in results.items():
        print(f"{filter_name:<20} | {metrics['PSNR']:<15.2f} | {metrics['SSIM']:<15.4f}")
    
    print("-" * 60)

def plot_evaluation_results(results):
    """
    Plot evaluation results as bar charts.
    
    Args:
        results (dict): Dictionary containing evaluation results
    """
    filter_names = list(results.keys())
    psnr_values = [results[name]['PSNR'] for name in filter_names]
    ssim_values = [results[name]['SSIM'] for name in filter_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # PSNR plot
    ax1.bar(filter_names, psnr_values, color='skyblue')
    ax1.set_title('PSNR Comparison')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_ylim(bottom=0)
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # SSIM plot
    ax2.bar(filter_names, ssim_values, color='lightgreen')
    ax2.set_title('SSIM Comparison')
    ax2.set_ylabel('SSIM')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show() 