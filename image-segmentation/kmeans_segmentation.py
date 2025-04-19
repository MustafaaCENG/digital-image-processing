import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error
import zipfile
import tempfile
import shutil

def load_image(image_path):
    """Load an image and return it."""
    return cv2.imread(image_path)

def convert_to_grayscale(image):
    """Convert an image to grayscale if it's not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def segment_grayscale_image(image, k):
    """
    Segment a grayscale image using k-means clustering.
    
    Args:
        image: Grayscale image as 2D numpy array
        k: Number of clusters
        
    Returns:
        Segmented image with each pixel replaced by its cluster centroid
    """
    # Reshape the image to a 1D array of pixels
    pixels = image.reshape(-1, 1)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Replace each pixel with its centroid value
    centroids = kmeans.cluster_centers_
    segmented = centroids[labels].reshape(image.shape).astype(np.uint8)
    
    return segmented

def segment_color_image(image, k):
    """
    Segment a color (RGB) image using k-means clustering.
    
    Args:
        image: Color image as 3D numpy array (height, width, 3)
        k: Number of clusters
        
    Returns:
        Segmented image with each pixel replaced by its cluster centroid
    """
    # Convert from BGR to RGB for better visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)
    
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Replace each pixel with its centroid value
    centroids = kmeans.cluster_centers_
    segmented = centroids[labels].reshape(image_rgb.shape).astype(np.uint8)
    
    return segmented

def calculate_mse(original, segmented):
    """Calculate Mean Squared Error between original and segmented images."""
    return mean_squared_error(original, segmented)

def calculate_normalized_mse(original, segmented):
    """Calculate normalized MSE."""
    mse = calculate_mse(original, segmented)
    return mse / (255.0 ** 2)  # Normalize by max possible pixel value squared

def save_image(image, filename, is_rgb=True):
    """Save an image to a file."""
    if is_rgb:
        # Convert from RGB to BGR for OpenCV saving
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image
    cv2.imwrite(filename, image_to_save)

def calculate_compression_ratio(file_path):
    """
    Calculate the compression ratio when zipping a file.
    
    Returns:
        Compression ratio = compressed_size / original_size
    """
    # Get the original file size
    original_size = os.path.getsize(file_path)
    
    # Create a temporary directory for zipping
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, 'compressed.zip')
    
    # Create a ZIP file containing the image
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, os.path.basename(file_path))
    
    # Get the compressed file size
    compressed_size = os.path.getsize(zip_path)
    
    # Calculate compression ratio
    compression_ratio = compressed_size / original_size
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    return compression_ratio

def process_image(image_path, k_values, output_dir, is_grayscale=False):
    """
    Process an image with different k values, calculate metrics and save results.
    
    Args:
        image_path: Path to the input image
        k_values: List of k values to use for segmentation
        output_dir: Directory to save output images and results
        is_grayscale: Whether to process the image as grayscale
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = load_image(image_path)
    
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Process as grayscale if specified or if image is already grayscale
    if is_grayscale or len(image.shape) == 2:
        if len(image.shape) == 3:
            image = convert_to_grayscale(image)
        
        # Save original image
        original_path = os.path.join(output_dir, 'original.png')
        cv2.imwrite(original_path, image)
        
        # Process with different k values
        results = []
        
        for k in k_values:
            # Segment the image
            segmented = segment_grayscale_image(image, k)
            
            # Save segmented image
            segmented_path = os.path.join(output_dir, f'segmented_k{k}.png')
            cv2.imwrite(segmented_path, segmented)
            
            # Calculate metrics
            n_mse = calculate_normalized_mse(image, segmented)
            comp_ratio = calculate_compression_ratio(segmented_path)
            orig_comp_ratio = calculate_compression_ratio(original_path)
            
            # Store results
            results.append({
                'k': k,
                'normalized_mse': n_mse,
                'compression_ratio': comp_ratio,
                'original_compression_ratio': orig_comp_ratio
            })
            
            print(f"Grayscale image, k={k}: NMSE={n_mse:.6f}, Compression Ratio={comp_ratio:.6f}")
        
        return results
    
    else:  # Process as color image
        # Convert from BGR to RGB for better visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original image
        original_path = os.path.join(output_dir, 'original.png')
        save_image(image_rgb, original_path)
        
        # Process with different k values
        results = []
        
        for k in k_values:
            # Segment the image
            segmented = segment_color_image(image, k)
            
            # Save segmented image
            segmented_path = os.path.join(output_dir, f'segmented_k{k}.png')
            save_image(segmented, segmented_path)
            
            # Calculate metrics
            n_mse = calculate_normalized_mse(image_rgb, segmented)
            comp_ratio = calculate_compression_ratio(segmented_path)
            orig_comp_ratio = calculate_compression_ratio(original_path)
            
            # Store results
            results.append({
                'k': k,
                'normalized_mse': n_mse,
                'compression_ratio': comp_ratio,
                'original_compression_ratio': orig_comp_ratio
            })
            
            print(f"Color image, k={k}: NMSE={n_mse:.6f}, Compression Ratio={comp_ratio:.6f}")
        
        return results

def visualize_results(image_path, k_values, output_dir, is_grayscale=False):
    """
    Visualize original and segmented images side by side.
    
    Args:
        image_path: Path to the input image
        k_values: List of k values used for segmentation
        output_dir: Directory containing output images
        is_grayscale: Whether the image is grayscale
    """
    # Load the original image
    image = load_image(image_path)
    
    if is_grayscale and len(image.shape) == 3:
        image = convert_to_grayscale(image)
    elif not is_grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a figure
    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(4 * (len(k_values) + 1), 4))
    
    # Display original image
    if is_grayscale:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Display segmented images
    for i, k in enumerate(k_values):
        segmented_path = os.path.join(output_dir, f'segmented_k{k}.png')
        segmented = cv2.imread(segmented_path)
        
        if is_grayscale:
            segmented = convert_to_grayscale(segmented)
            axes[i+1].imshow(segmented, cmap='gray')
        else:
            segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
            axes[i+1].imshow(segmented)
        
        axes[i+1].set_title(f'k = {k}')
        axes[i+1].axis('off')
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization.png'), dpi=300)
    plt.close()

def main():
    # Set up paths
    data_dir = os.path.join('data', 'input')
    results_dir = os.path.join('data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Define k values
    k_values = [2, 4, 8, 16]
    
    # Process grayscale image
    grayscale_image = os.path.join(data_dir, 'camera_gray.png')
    grayscale_results_dir = os.path.join(results_dir, 'grayscale')
    grayscale_results = process_image(grayscale_image, k_values, grayscale_results_dir, is_grayscale=True)
    visualize_results(grayscale_image, k_values, grayscale_results_dir, is_grayscale=True)
    
    # Process color image
    color_image = os.path.join(data_dir, 'astronaut_color.png')
    color_results_dir = os.path.join(results_dir, 'color')
    color_results = process_image(color_image, k_values, color_results_dir, is_grayscale=False)
    visualize_results(color_image, k_values, color_results_dir, is_grayscale=False)
    
    # Print summary
    print("\nSummary of Results:")
    print("\nGrayscale Image:")
    for result in grayscale_results:
        print(f"k={result['k']}, NMSE={result['normalized_mse']:.6f}, Compression Ratio={result['compression_ratio']:.6f}")
    
    print("\nColor Image:")
    for result in color_results:
        print(f"k={result['k']}, NMSE={result['normalized_mse']:.6f}, Compression Ratio={result['compression_ratio']:.6f}")
    
    # Generate plots comparing metrics across k values
    plot_metrics(grayscale_results, color_results, results_dir)

def plot_metrics(grayscale_results, color_results, output_dir):
    """Generate plots comparing metrics across k values."""
    # Extract data for plotting
    k_values = [result['k'] for result in grayscale_results]
    grayscale_nmse = [result['normalized_mse'] for result in grayscale_results]
    grayscale_comp = [result['compression_ratio'] for result in grayscale_results]
    color_nmse = [result['normalized_mse'] for result in color_results]
    color_comp = [result['compression_ratio'] for result in color_results]
    
    # Create NMSE plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, grayscale_nmse, 'o-', label='Grayscale')
    plt.plot(k_values, color_nmse, 's-', label='Color')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Normalized Mean Squared Error')
    plt.title('Image Quality vs. Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'nmse_comparison.png'), dpi=300)
    plt.close()
    
    # Create Compression Ratio plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, grayscale_comp, 'o-', label='Grayscale')
    plt.plot(k_values, color_comp, 's-', label='Color')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio vs. Number of Clusters')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'compression_comparison.png'), dpi=300)
    plt.close()
    
    # Create MSE vs Compression plot
    plt.figure(figsize=(10, 6))
    plt.scatter(grayscale_nmse, grayscale_comp, s=100, marker='o', label='Grayscale')
    plt.scatter(color_nmse, color_comp, s=100, marker='s', label='Color')
    
    # Annotate points with k values
    for i, k in enumerate(k_values):
        plt.annotate(f'k={k}', (grayscale_nmse[i], grayscale_comp[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'k={k}', (color_nmse[i], color_comp[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Normalized Mean Squared Error')
    plt.ylabel('Compression Ratio')
    plt.title('Quality vs. Compression Tradeoff')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'quality_compression_tradeoff.png'), dpi=300)
    plt.close()

if __name__ == '__main__':
    main() 