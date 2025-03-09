"""
Run Image Enhancement Workflow

This script runs the entire workflow:
1. Download sample images
2. Prepare them (create underexposed and overexposed versions)
3. Run the image enhancement operations
"""

import os
import subprocess
import sys

def run_script(script_name):
    """
    Run a Python script and print its output.
    
    Args:
        script_name: Name of the script to run
    """
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}\n")
    
    # Get the Python executable path
    python_exe = sys.executable
    
    # Run the script
    result = subprocess.run([python_exe, script_name], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True)
    
    # Print output
    print(result.stdout)
    
    # Print errors if any
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    return result.returncode == 0

def main():
    """
    Main function to run the workflow.
    """
    # Check if we're in the right directory
    if not os.path.exists("image_enhancement.py"):
        # Try to change to the image_enhancement directory
        if os.path.exists("image_enhancement") and os.path.isdir("image_enhancement"):
            os.chdir("image_enhancement")
        else:
            print("Error: Could not find the image_enhancement.py file.")
            print("Please run this script from the image_enhancement directory.")
            return
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Download sample images
    if run_script("download_sample_images.py"):
        print("Sample images downloaded successfully.")
    else:
        print("Failed to download sample images.")
        return
    
    # Step 2: Prepare sample images
    if run_script("prepare_sample_images.py"):
        print("Sample images prepared successfully.")
    else:
        print("Failed to prepare sample images.")
        return
    
    # Step 3: Run image enhancement operations
    if run_script("image_enhancement.py"):
        print("Image enhancement operations completed successfully.")
    else:
        print("Failed to run image enhancement operations.")
        return
    
    print("\nWorkflow completed successfully.")
    print("Results are saved in the 'results' directory.")

if __name__ == "__main__":
    main() 