import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, transform
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte, img_as_float
from skimage.filters import sobel, prewitt, roberts, laplace
from scipy import ndimage as ndi
import os
import time

class EdgeDetector:
    """
    A class that implements various edge detection algorithms and provides
    tools for comparing their performance on different images.
    """
    
    def __init__(self, img_paths):
        """
        Initialize the EdgeDetector with a list of image paths.
        
        Parameters:
        -----------
        img_paths : list
            List of paths to the images to be processed.
        """
        self.img_paths = img_paths
        self.images = []
        self.image_names = []
        self.load_images()
        
    def load_images(self):
        """Load images from the provided paths."""
        self.images = []
        self.image_names = []
        for path in self.img_paths:
            img = io.imread(path)
            # Convert to grayscale if the image is RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = rgb2gray(img)
            self.images.append(img)
            # Extract filename without extension
            self.image_names.append(os.path.splitext(os.path.basename(path))[0])
    
    def apply_sobel(self, img=None):
        """
        Apply Sobel edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
            
        Returns:
        --------
        ndarray
            The edge-detected image.
        """
        if img is None:
            img = self.images[0]
        
        # Apply Sobel filter
        edges = sobel(img)
        return edges
    
    def apply_prewitt(self, img=None):
        """
        Apply Prewitt edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
            
        Returns:
        --------
        ndarray
            The edge-detected image.
        """
        if img is None:
            img = self.images[0]
        
        # Apply Prewitt filter
        edges = prewitt(img)
        return edges
    
    def apply_roberts(self, img=None):
        """
        Apply Roberts edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
            
        Returns:
        --------
        ndarray
            The edge-detected image.
        """
        if img is None:
            img = self.images[0]
        
        # Apply Roberts filter
        edges = roberts(img)
        return edges
    
    def apply_log(self, img=None, sigma=1.0):
        """
        Apply Laplacian of Gaussian (LoG) edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
        sigma : float, optional
            Standard deviation of the Gaussian filter.
            
        Returns:
        --------
        ndarray
            The edge-detected image.
        """
        if img is None:
            img = self.images[0]
        
        # Apply Gaussian filter to smooth the image
        smoothed = ndi.gaussian_filter(img, sigma=sigma)
        
        # Apply Laplacian filter to detect edges
        edges = laplace(smoothed)
        
        # Normalize to range [0, 1]
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        
        return edges
    
    def apply_canny(self, img=None, sigma=1.0, low_threshold=0.1, high_threshold=0.2):
        """
        Apply Canny edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
        sigma : float, optional
            Standard deviation of the Gaussian filter.
        low_threshold : float, optional
            Lower threshold for the hysteresis procedure.
        high_threshold : float, optional
            Higher threshold for the hysteresis procedure.
            
        Returns:
        --------
        ndarray
            The edge-detected image.
        """
        if img is None:
            img = self.images[0]
        
        # Apply Canny edge detection
        edges = feature.canny(
            img,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold
        )
        
        return edges
    
    def apply_hough_transform(self, img=None, edge_method='canny', threshold=50):
        """
        Apply Hough transform to detect lines after edge detection.
        
        Parameters:
        -----------
        img : ndarray, optional
            The image to process. If None, the first loaded image is used.
        edge_method : str, optional
            Edge detection method to use before Hough transform.
            Options: 'sobel', 'prewitt', 'roberts', 'log', 'canny'.
        threshold : int, optional
            Threshold for detecting lines in the Hough transform.
            
        Returns:
        --------
        tuple
            Original image, edge-detected image, and image with detected lines.
        """
        if img is None:
            img = self.images[0]
        
        # Apply specified edge detection method
        if edge_method == 'sobel':
            edges = self.apply_sobel(img)
        elif edge_method == 'prewitt':
            edges = self.apply_prewitt(img)
        elif edge_method == 'roberts':
            edges = self.apply_roberts(img)
        elif edge_method == 'log':
            edges = self.apply_log(img)
        else:  # Default to Canny
            edges = self.apply_canny(img)
        
        # Convert to binary image if not already
        binary_edges = edges > 0.09
        
        # Compute Hough transform
        tested_angles = np.linspace(-np.pi/2, np.pi/2, 180, endpoint=False)
        h, theta, d = transform.hough_line(binary_edges, theta=tested_angles)
        
        # Create a new image to draw detected lines
        line_img = np.zeros((*img.shape, 3))
        if len(img.shape) == 2:
            # Grayscale image
            line_img[:, :, 0] = line_img[:, :, 1] = line_img[:, :, 2] = img
        else:
            # Already RGB
            line_img = img.copy()
        
        # Find the peaks in the Hough transform
        hspace, angles, dists = transform.hough_line_peaks(h, theta, d, 
                                                        num_peaks=np.inf,
                                                        threshold=threshold)
        
        # Draw detected lines
        for _, angle, dist in zip(hspace, angles, dists):
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - img.shape[1] * np.cos(angle)) / np.sin(angle)
            
            # Check for vertical lines to avoid division by zero
            if np.isclose(np.sin(angle), 0):
                x0 = dist / np.cos(angle)
                x1 = dist / np.cos(angle)
                y0 = 0
                y1 = img.shape[0]
            else:
                x0, x1 = 0, img.shape[1]
            
            # Draw the line
            if 0 <= y0 <= img.shape[0] and 0 <= y1 <= img.shape[0]:
                transform.hough_line_peaks
                line_img = transform.draw_line(x0, y0, x1, y1, line_img, color=[1, 0, 0])
        
        return img, binary_edges, line_img
    
    def compare_all_methods(self, index=0, save_path=None):
        """
        Apply all edge detection methods to a single image and display the results.
        
        Parameters:
        -----------
        index : int, optional
            Index of the image to process.
        save_path : str, optional
            Path to save the result figure. If None, the figure is displayed.
            
        Returns:
        --------
        dict
            Dictionary containing the computation times for each method.
        """
        if index >= len(self.images):
            raise ValueError(f"Image index {index} out of range.")
        
        img = self.images[index]
        img_name = self.image_names[index]
        
        # Apply all edge detection methods and measure computation time
        times = {}
        
        start_time = time.time()
        sobel_edges = self.apply_sobel(img)
        times['Sobel'] = time.time() - start_time
        
        start_time = time.time()
        prewitt_edges = self.apply_prewitt(img)
        times['Prewitt'] = time.time() - start_time
        
        start_time = time.time()
        roberts_edges = self.apply_roberts(img)
        times['Roberts'] = time.time() - start_time
        
        start_time = time.time()
        log_edges = self.apply_log(img)
        times['LoG'] = time.time() - start_time
        
        start_time = time.time()
        canny_edges = self.apply_canny(img)
        times['Canny'] = time.time() - start_time
        
        # Create a figure to display the results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Display the original image
        axes[0, 0].imshow(img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Display the edge-detected images
        axes[0, 1].imshow(sobel_edges, cmap='gray')
        axes[0, 1].set_title(f'Sobel ({times["Sobel"]:.3f}s)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prewitt_edges, cmap='gray')
        axes[0, 2].set_title(f'Prewitt ({times["Prewitt"]:.3f}s)')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(roberts_edges, cmap='gray')
        axes[1, 0].set_title(f'Roberts ({times["Roberts"]:.3f}s)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(log_edges, cmap='gray')
        axes[1, 1].set_title(f'LoG ({times["LoG"]:.3f}s)')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(canny_edges, cmap='gray')
        axes[1, 2].set_title(f'Canny ({times["Canny"]:.3f}s)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Edge Detection Comparison - {img_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return times
    
    def process_all_images(self, output_dir='results'):
        """
        Apply all edge detection methods to all loaded images.
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save the result figures.
            
        Returns:
        --------
        dict
            Dictionary containing the computation times for each image and method.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        all_times = {}
        
        for i, (img, name) in enumerate(zip(self.images, self.image_names)):
            save_path = os.path.join(output_dir, f'{name}_comparison.png')
            times = self.compare_all_methods(i, save_path)
            all_times[name] = times
            
            # Save individual edge detection results
            plt.figure(figsize=(8, 8))
            plt.imshow(img, cmap='gray')
            plt.title(f'Original Image - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_original.png'))
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(self.apply_sobel(img), cmap='gray')
            plt.title(f'Sobel Edge Detection - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_sobel.png'))
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(self.apply_prewitt(img), cmap='gray')
            plt.title(f'Prewitt Edge Detection - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_prewitt.png'))
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(self.apply_roberts(img), cmap='gray')
            plt.title(f'Roberts Edge Detection - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_roberts.png'))
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(self.apply_log(img), cmap='gray')
            plt.title(f'LoG Edge Detection - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_log.png'))
            plt.close()
            
            plt.figure(figsize=(8, 8))
            plt.imshow(self.apply_canny(img), cmap='gray')
            plt.title(f'Canny Edge Detection - {name}')
            plt.axis('off')
            plt.savefig(os.path.join(output_dir, f'{name}_canny.png'))
            plt.close()
            
            # Apply Hough Transform (optional)
            try:
                _, edges, lines = self.apply_hough_transform(img)
                plt.figure(figsize=(8, 8))
                plt.imshow(lines)
                plt.title(f'Hough Line Detection - {name}')
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'{name}_hough.png'))
                plt.close()
            except Exception as e:
                print(f"Hough transform failed for {name}: {e}")
        
        return all_times 