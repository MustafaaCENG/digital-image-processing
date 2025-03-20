import numpy as np
import cv2
from typing import Tuple, Union

def downsample_direct(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample image by direct pixel deletion.
    
    Args:
        image: Input image array
        factor: Downsampling factor (2 for half size, 4 for quarter, etc.)
    
    Returns:
        Downsampled image
    """
    return image[::factor, ::factor]

def downsample_interpolation(image: np.ndarray, factor: int, method: str = 'bilinear') -> np.ndarray:
    """
    Downsample image using interpolation.
    
    Args:
        image: Input image array
        factor: Downsampling factor (2 for half size, 4 for quarter, etc.)
        method: Interpolation method ('bilinear' or 'bicubic')
    
    Returns:
        Downsampled image
    """
    new_size = (image.shape[1] // factor, image.shape[0] // factor)
    interpolation = cv2.INTER_LINEAR if method == 'bilinear' else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interpolation)

def upsample(image: np.ndarray, size: Tuple[int, int], method: str = 'nearest') -> np.ndarray:
    """
    Upsample image using specified interpolation method.
    
    Args:
        image: Input image array
        size: Target size (width, height)
        method: Interpolation method ('nearest', 'bilinear', or 'bicubic')
    
    Returns:
        Upsampled image
    """
    interpolation_methods = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }
    
    interpolation = interpolation_methods.get(method, cv2.INTER_LINEAR)
    return cv2.resize(image, size, interpolation=interpolation)

def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original: Original image array
        processed: Processed image array
    
    Returns:
        MSE value
    """
    return np.mean((original - processed) ** 2)

def calculate_psnr(original: np.ndarray, processed: np.ndarray, max_pixel: int = 255) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image array
        processed: Processed image array
        max_pixel: Maximum pixel value (default: 255)
    
    Returns:
        PSNR value in dB
    """
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse)) 