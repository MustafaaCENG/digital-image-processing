import os
import cv2
import numpy as np
from typing import List, Dict
from interpolation import downsample_direct, downsample_interpolation, upsample
from analysis import plot_comparison, analyze_quality, save_results, log_processing

# Configuration
FACTORS = [2, 4, 8]  # Downsampling factors
INTERPOLATION_METHODS = ['nearest', 'bilinear', 'bicubic']
LOG_FILE = 'logs/log.txt'

def process_image(image_path: str, is_grayscale: bool = False) -> None:
    """
    Process a single image through all downsampling and upsampling methods.
    
    Args:
        image_path: Path to the input image
        is_grayscale: Whether the image is grayscale
    """
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR)
    if img is None:
        log_processing(f"Error: Could not read image {image_path}", LOG_FILE)
        return
    
    # Check image size
    if img.shape[0] < 512 or img.shape[1] < 512:
        log_processing(f"Warning: Image {image_path} is smaller than 512x512 pixels. Skipping.", LOG_FILE)
        return
    
    original_size = img.shape[:2]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    log_processing(f"Processing image: {image_path}", LOG_FILE)
    
    results = {}
    
    # Process for each downsampling factor
    for factor in FACTORS:
        # Direct downsampling
        downsampled_direct = downsample_direct(img, factor)
        
        # Interpolation-based downsampling
        downsampled_interp = downsample_interpolation(img, factor, 'bilinear')
        
        # Save downsampled images
        cv2.imwrite(f"results/downsampled/{image_name}_direct_f{factor}.png", downsampled_direct)
        cv2.imwrite(f"results/downsampled/{image_name}_interp_f{factor}.png", downsampled_interp)
        
        # Upsampling with different methods
        for method in INTERPOLATION_METHODS:
            # Process direct downsampled image
            upsampled_direct = upsample(downsampled_direct, (original_size[1], original_size[0]), method)
            
            # Process interpolation downsampled image
            upsampled_interp = upsample(downsampled_interp, (original_size[1], original_size[0]), method)
            
            # Save upsampled images
            cv2.imwrite(f"results/upsampled/{image_name}_direct_f{factor}_{method}.png", upsampled_direct)
            cv2.imwrite(f"results/upsampled/{image_name}_interp_f{factor}_{method}.png", upsampled_interp)
            
            # Analyze quality
            key_direct = f"Factor {factor} - Direct - {method}"
            key_interp = f"Factor {factor} - Interpolated - {method}"
            
            mse_direct, psnr_direct = analyze_quality(img, upsampled_direct)
            mse_interp, psnr_interp = analyze_quality(img, upsampled_interp)
            
            results[key_direct] = {'mse': mse_direct, 'psnr': psnr_direct}
            results[key_interp] = {'mse': mse_interp, 'psnr': psnr_interp}
            
            # Create comparison plots
            plot_comparison(
                {
                    'Original': img,
                    'Downsampled (Direct)': cv2.resize(downsampled_direct, (original_size[1], original_size[0])),
                    f'Upsampled ({method})': upsampled_direct
                },
                f"{image_name} - Factor {factor} - Direct - {method}",
                f"results/analysis/{image_name}_direct_f{factor}_{method}_comparison.png"
            )
            
            plot_comparison(
                {
                    'Original': img,
                    'Downsampled (Interpolated)': cv2.resize(downsampled_interp, (original_size[1], original_size[0])),
                    f'Upsampled ({method})': upsampled_interp
                },
                f"{image_name} - Factor {factor} - Interpolated - {method}",
                f"results/analysis/{image_name}_interp_f{factor}_{method}_comparison.png"
            )
    
    # Save analysis results
    save_results(results, f"results/analysis/{image_name}_analysis.txt")
    log_processing(f"Completed processing image: {image_path}", LOG_FILE)

def process_directory(directory: str, is_grayscale: bool = False) -> int:
    """
    Process all images in a directory.
    
    Args:
        directory: Directory path
        is_grayscale: Whether to process images as grayscale
    
    Returns:
        Number of images processed
    """
    if not os.path.exists(directory):
        log_processing(f"Warning: Directory {directory} does not exist.", LOG_FILE)
        return 0
        
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(os.path.join(directory, filename), is_grayscale)
            count += 1
    return count

def main():
    """Main function to process all images."""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/downsampled', exist_ok=True)
    os.makedirs('results/upsampled', exist_ok=True)
    os.makedirs('results/analysis', exist_ok=True)
    os.makedirs('images/grayscale', exist_ok=True)
    os.makedirs('images/rgb', exist_ok=True)
    
    log_processing("Starting image processing", LOG_FILE)
    
    # Process images
    grayscale_count = process_directory('images/grayscale', is_grayscale=True)
    rgb_count = process_directory('images/rgb', is_grayscale=False)
    
    if grayscale_count == 0 and rgb_count == 0:
        log_processing("No images found to process. Please add images to the following directories:", LOG_FILE)
        log_processing("- Grayscale images (512x512 or larger): images/grayscale/", LOG_FILE)
        log_processing("- RGB images (512x512 or larger): images/rgb/", LOG_FILE)
        return
    
    log_processing(f"Processed {grayscale_count} grayscale images and {rgb_count} RGB images", LOG_FILE)
    log_processing("Completed all image processing", LOG_FILE)

if __name__ == "__main__":
    main() 