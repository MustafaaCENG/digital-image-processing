import os
import numpy as np
from skimage import data
from skimage.io import imsave

# Create data/input directory if it doesn't exist
os.makedirs('data/input', exist_ok=True)

# Download and save the cameraman image (grayscale)
camera = data.camera()
imsave('data/input/camera_gray.png', camera)

# Download and save the astronaut image (color)
astronaut = data.astronaut()
imsave('data/input/astronaut_color.png', astronaut)

print("Images downloaded successfully to data/input directory.") 