import os
import matplotlib.pyplot as plt
import numpy as np
from edge_detection import EdgeDetector
from skimage import io, color
import argparse

def main():
    """
    Main function to run edge detection algorithms on sample images.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Edge Detection Algorithm Comparison')
    parser.add_argument('--samples_dir', type=str, default='samples/edges',
                        help='Directory containing sample images')
    parser.add_argument('--results_dir', type=str, default='results/edge_detection',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check if sample images exist, if not, download or create some
    sample_images = [os.path.join(args.samples_dir, f) for f in os.listdir(args.samples_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if len(sample_images) < 3:
        print("Not enough sample images found. Creating sample images...")
        create_sample_images(args.samples_dir)
        # Update the list of sample images
        sample_images = [os.path.join(args.samples_dir, f) for f in os.listdir(args.samples_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    # Initialize edge detector with sample images
    detector = EdgeDetector(sample_images)
    
    # Process all images and save results
    print("Processing images...")
    all_times = detector.process_all_images(args.results_dir)
    
    # Print timing results
    print("\nComputation time results (seconds):")
    for img_name, times in all_times.items():
        print(f"\nImage: {img_name}")
        for method, time_taken in times.items():
            print(f"  {method}: {time_taken:.5f}s")
    
    # Generate a timing comparison chart
    generate_timing_chart(all_times, os.path.join(args.results_dir, 'timing_comparison.png'))
    
    print(f"\nAll results saved to {args.results_dir}")
    print("Edge detection completed successfully.")

def create_sample_images(output_dir):
    """
    Create or download sample images if none are provided.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save sample images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a sample image with geometric shapes
    # 1. Create a black image with white shapes
    shapes = np.zeros((512, 512), dtype=np.uint8)
    
    # Add a white rectangle
    shapes[100:200, 100:400] = 255
    
    # Add a white circle
    center = (350, 350)
    radius = 80
    y, x = np.ogrid[:512, :512]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    shapes[mask] = 255
    
    # Add a white triangle
    from skimage.draw import polygon
    r = np.array([50, 150, 100])
    c = np.array([400, 450, 300])
    rr, cc = polygon(r, c)
    shapes[rr, cc] = 255
    
    # Save the shapes image
    io.imsave(os.path.join(output_dir, 'geometric_shapes.png'), shapes)
    
    # 2. Create a sample text image
    text = np.ones((512, 512), dtype=np.uint8) * 255
    
    # Add some black text-like shapes
    for i in range(10):
        start_y = 50 + i * 40
        width = np.random.randint(200, 400)
        text[start_y:start_y+10, 50:50+width] = 0
    
    # Add some vertical lines to simulate more complex text
    for i in range(5):
        start_x = 100 + i * 80
        height = np.random.randint(100, 300)
        text[200:200+height, start_x:start_x+5] = 0
    
    # Save the text image
    io.imsave(os.path.join(output_dir, 'text_sample.png'), text)
    
    # 3. Create a sample noisy image with edges
    noisy = np.zeros((512, 512), dtype=np.uint8)
    
    # Add a gradient
    x, y = np.meshgrid(np.linspace(0, 1, 512), np.linspace(0, 1, 512))
    gradient = (x + y) * 127.5
    noisy = gradient.astype(np.uint8)
    
    # Add some squares of different intensities
    noisy[50:150, 50:150] = 200
    noisy[200:300, 200:300] = 100
    noisy[350:450, 350:450] = 50
    
    # Add noise
    noise = np.random.normal(0, 15, (512, 512))
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    
    # Save the noisy image
    io.imsave(os.path.join(output_dir, 'noisy_gradient.png'), noisy)
    
    print(f"Created 3 sample images in {output_dir}")

def generate_timing_chart(all_times, save_path):
    """
    Generate a bar chart comparing timing results for different methods.
    
    Parameters:
    -----------
    all_times : dict
        Dictionary containing timing results for each image and method.
    save_path : str
        Path to save the chart.
    """
    # Extract methods and images
    methods = list(next(iter(all_times.values())).keys())
    images = list(all_times.keys())
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set bar width and positions
    bar_width = 0.15
    positions = np.arange(len(methods))
    
    # Plot bars for each image
    for i, img_name in enumerate(images):
        times = [all_times[img_name][method] for method in methods]
        ax.bar(positions + i * bar_width, times, bar_width, label=img_name)
    
    # Set labels and title
    ax.set_xlabel('Edge Detection Method')
    ax.set_ylabel('Computation Time (seconds)')
    ax.set_title('Computation Time Comparison of Edge Detection Methods')
    ax.set_xticks(positions + bar_width * (len(images) - 1) / 2)
    ax.set_xticklabels(methods)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Timing comparison chart saved to {save_path}")

if __name__ == "__main__":
    main() 