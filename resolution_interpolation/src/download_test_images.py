import os
import urllib.request
import cv2
import numpy as np

def download_image(url: str, save_path: str) -> bool:
    """Download an image from URL and save it to the specified path."""
    try:
        print(f"Downloading {os.path.basename(save_path)}...")
        # Create a request with a user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        req = urllib.request.Request(url, headers=headers)
        
        # Download the image
        with urllib.request.urlopen(req) as response:
            with open(save_path, 'wb') as f:
                f.write(response.read())
        
        # Verify the image can be opened
        img = cv2.imread(save_path)
        if img is None:
            os.remove(save_path)
            return False
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

def create_test_image(save_path: str, size: tuple = (512, 512), is_grayscale: bool = True) -> None:
    """Create a synthetic test image when download fails."""
    if is_grayscale:
        # Create a grayscale test pattern
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        X, Y = np.meshgrid(x, y)
        img = np.uint8(255 * (X * Y + np.sin(X * 10) * np.sin(Y * 10) * 0.2))
    else:
        # Create an RGB test pattern
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        x = np.linspace(0, 1, size[0])
        y = np.linspace(0, 1, size[1])
        X, Y = np.meshgrid(x, y)
        
        # Red channel
        img[:, :, 0] = np.uint8(255 * np.sin(X * 5) * np.sin(Y * 5))
        # Green channel
        img[:, :, 1] = np.uint8(255 * X)
        # Blue channel
        img[:, :, 2] = np.uint8(255 * Y)
    
    cv2.imwrite(save_path, img)
    print(f"Created synthetic test image: {os.path.basename(save_path)}")

def main():
    # Create directories if they don't exist
    os.makedirs('images/grayscale', exist_ok=True)
    os.makedirs('images/rgb', exist_ok=True)
    
    # Standard test images
    grayscale_images = {
        'cameraman': 'https://homepages.cae.wisc.edu/~ece533/images/cameraman.png',
        'boat': 'https://homepages.cae.wisc.edu/~ece533/images/boat.png'
    }
    
    rgb_images = {
        'baboon': 'https://homepages.cae.wisc.edu/~ece533/images/baboon.png',
        'peppers': 'https://homepages.cae.wisc.edu/~ece533/images/peppers.png'
    }
    
    # Try downloading grayscale images, create synthetic ones if download fails
    for name, url in grayscale_images.items():
        save_path = f'images/grayscale/{name}.png'
        if not download_image(url, save_path):
            create_test_image(save_path, (512, 512), is_grayscale=True)
    
    # Try downloading RGB images, create synthetic ones if download fails
    for name, url in rgb_images.items():
        save_path = f'images/rgb/{name}.png'
        if not download_image(url, save_path):
            create_test_image(save_path, (512, 512), is_grayscale=False)

if __name__ == "__main__":
    print("Setting up test images...")
    main()
    print("\nSetup complete! You can now run main.py to process the images.") 