import os
import cv2
import numpy as np

def apply_gamma(image, gamma):
    """
    Apply gamma correction to an image.
    
    Args:
        image: Input image
        gamma: Gamma value
        
    Returns:
        Gamma-corrected image
    """
    # Ensure image is in [0, 1] range
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    
    # Apply gamma correction
    corrected = np.power(image, gamma)
    
    # Convert back to uint8
    if corrected.max() <= 1.0:
        corrected = (corrected * 255).astype(np.uint8)
    
    return corrected

def prepare_images():
    """
    Prepare sample images by creating underexposed and overexposed versions.
    """
    # Create images directory
    images_dir = "images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Prepare grayscale images
    balanced_path = os.path.join(images_dir, "balanced.jpg")
    underexposed_path = os.path.join(images_dir, "underexposed.jpg")
    overexposed_path = os.path.join(images_dir, "overexposed.jpg")
    
    # Check if balanced image exists
    if os.path.exists(balanced_path):
        # Load balanced image
        balanced_img = cv2.imread(balanced_path)
        
        if balanced_img is not None:
            # Create underexposed version (gamma > 1)
            underexposed_img = apply_gamma(balanced_img, 2.5)
            cv2.imwrite(underexposed_path, underexposed_img)
            print(f"Created underexposed version: {underexposed_path}")
            
            # Create overexposed version (gamma < 1)
            overexposed_img = apply_gamma(balanced_img, 0.4)
            cv2.imwrite(overexposed_path, overexposed_img)
            print(f"Created overexposed version: {overexposed_path}")
        else:
            print(f"Failed to load balanced image: {balanced_path}")
    else:
        print(f"Balanced image not found: {balanced_path}")
    
    # Prepare color images
    color_balanced_path = os.path.join(images_dir, "color_balanced.jpg")
    color_underexposed_path = os.path.join(images_dir, "color_underexposed.jpg")
    color_overexposed_path = os.path.join(images_dir, "color_overexposed.jpg")
    
    # Check if color balanced image exists
    if os.path.exists(color_balanced_path):
        # Load color balanced image
        color_balanced_img = cv2.imread(color_balanced_path)
        
        if color_balanced_img is not None:
            # Create underexposed version (gamma > 1)
            color_underexposed_img = apply_gamma(color_balanced_img, 2.5)
            cv2.imwrite(color_underexposed_path, color_underexposed_img)
            print(f"Created color underexposed version: {color_underexposed_path}")
            
            # Create overexposed version (gamma < 1)
            color_overexposed_img = apply_gamma(color_balanced_img, 0.4)
            cv2.imwrite(color_overexposed_path, color_overexposed_img)
            print(f"Created color overexposed version: {color_overexposed_path}")
        else:
            print(f"Failed to load color balanced image: {color_balanced_path}")
    else:
        print(f"Color balanced image not found: {color_balanced_path}")

if __name__ == "__main__":
    prepare_images()
    print("\nSample image preparation completed.") 