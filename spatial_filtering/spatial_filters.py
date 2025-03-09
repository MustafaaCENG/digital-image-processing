import numpy as np
import cv2
from scipy import ndimage

def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel with specified size and standard deviation.
    
    Args:
        size (int): Size of the kernel (must be odd)
        sigma (float): Standard deviation of the Gaussian
        
    Returns:
        numpy.ndarray: 2D Gaussian kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure size is odd
    
    # Create a 1D coordinate array from -size//2 to size//2
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    # Create a meshgrid from the coordinate array
    xx, yy = np.meshgrid(ax, ax)
    # Calculate the 2D Gaussian kernel
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    # Normalize the kernel to ensure the sum is 1
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, kernel_size, sigma):
    """
    Apply Gaussian filter to an image.
    
    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian
        
    Returns:
        numpy.ndarray: Filtered image
    """
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)
    # Apply the filter using convolution
    return ndimage.convolve(image, kernel, mode='reflect')

def unsharp_masking(image, kernel_size, sigma, amount):
    """
    Apply unsharp masking to sharpen an image.
    
    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian
        amount (float): Strength of sharpening effect
        
    Returns:
        numpy.ndarray: Sharpened image
    """
    # Apply Gaussian blur
    blurred = apply_gaussian_filter(image, kernel_size, sigma)
    
    # Calculate the high-frequency components (detail)
    detail = image - blurred
    
    # Add scaled detail to the original image
    sharpened = image + amount * detail
    
    # Clip values to valid range
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Edge map
        numpy.ndarray: Gradient magnitude
        numpy.ndarray: Gradient direction
    """
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Apply Sobel operators
    grad_x = ndimage.convolve(image.astype(float), sobel_x, mode='reflect')
    grad_y = ndimage.convolve(image.astype(float), sobel_y, mode='reflect')
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Normalize magnitude to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    # Create binary edge map (simple thresholding)
    threshold = 30  # Adjustable threshold
    edge_map = (magnitude > threshold).astype(np.uint8) * 255
    
    return edge_map, magnitude, direction

def prewitt_edge_detection(image):
    """
    Apply Prewitt edge detection to an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Edge map
        numpy.ndarray: Gradient magnitude
    """
    # Define Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply Prewitt operators
    grad_x = ndimage.convolve(image.astype(float), prewitt_x, mode='reflect')
    grad_y = ndimage.convolve(image.astype(float), prewitt_y, mode='reflect')
    
    # Calculate gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize magnitude to 0-255 range
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    
    # Create binary edge map (simple thresholding)
    threshold = 30  # Adjustable threshold
    edge_map = (magnitude > threshold).astype(np.uint8) * 255
    
    return edge_map, magnitude

def laplacian_edge_detection(image):
    """
    Apply Laplacian edge detection to an image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Edge map
    """
    # Define Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Apply Laplacian operator
    result = ndimage.convolve(image.astype(float), laplacian, mode='reflect')
    
    # Take absolute value and normalize to 0-255 range
    result = np.abs(result)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Create binary edge map (simple thresholding)
    threshold = 30  # Adjustable threshold
    edge_map = (result > threshold).astype(np.uint8) * 255
    
    return edge_map

def median_filter(image, kernel_size):
    """
    Apply median filter to an image.
    
    Args:
        image (numpy.ndarray): Input image
        kernel_size (int): Size of the median filter kernel
        
    Returns:
        numpy.ndarray: Filtered image
    """
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure size is odd
    
    return ndimage.median_filter(image, size=kernel_size)

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    """
    Add salt and pepper noise to an image.
    
    Args:
        image (numpy.ndarray): Input image
        salt_prob (float): Probability of salt noise (white pixels)
        pepper_prob (float): Probability of pepper noise (black pixels)
        
    Returns:
        numpy.ndarray: Noisy image
    """
    noisy_image = np.copy(image)
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image.
    
    Args:
        image (numpy.ndarray): Input image
        mean (float): Mean of the Gaussian noise
        sigma (float): Standard deviation of the Gaussian noise
        
    Returns:
        numpy.ndarray: Noisy image
    """
    # Generate Gaussian noise
    noise = np.random.normal(mean, sigma, image.shape)
    
    # Add noise to image
    noisy_image = image.astype(np.float64) + noise
    
    # Clip values to valid range
    return np.clip(noisy_image, 0, 255).astype(np.uint8) 