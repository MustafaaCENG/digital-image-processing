import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os
import cv2

def plot_comparison(images: Dict[str, np.ndarray], title: str, save_path: str) -> None:
    """
    Plot multiple images for comparison.
    
    Args:
        images: Dictionary of images with their labels
        title: Main title for the plot
        save_path: Path to save the plot
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for ax, (label, img) in zip(axes, images.items()):
        if len(img.shape) == 2:  # Grayscale
            ax.imshow(img, cmap='gray')
        else:  # RGB
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_quality(original: np.ndarray, processed: np.ndarray) -> Tuple[float, float]:
    """
    Analyze image quality using MSE and PSNR metrics.
    
    Args:
        original: Original image
        processed: Processed image
    
    Returns:
        Tuple of (MSE, PSNR) values
    """
    from interpolation import calculate_mse, calculate_psnr
    mse = calculate_mse(original, processed)
    psnr = calculate_psnr(original, processed)
    return mse, psnr

def save_results(results: Dict[str, Dict[str, float]], save_path: str) -> None:
    """
    Save analysis results to a text file.
    
    Args:
        results: Dictionary containing analysis results
        save_path: Path to save the results
    """
    with open(save_path, 'w') as f:
        f.write("Image Quality Analysis Results\n")
        f.write("=" * 30 + "\n\n")
        
        for method, metrics in results.items():
            f.write(f"{method}:\n")
            f.write(f"  MSE:  {metrics['mse']:.4f}\n")
            f.write(f"  PSNR: {metrics['psnr']:.4f} dB\n\n")

def log_processing(message: str, log_file: str) -> None:
    """
    Log processing steps and results.
    
    Args:
        message: Message to log
        log_file: Path to log file
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n") 