"""
Test script for the depth-ordered pixel processing in parallax_generator.py
This script should be run from the ComfyUI environment.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the current directory to the path so we can import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our module
from parallax_generator import Parallax_Generator_by_SamSeen

def test_parallax_generator():
    """Test the parallax generator with a simple image"""
    print("Testing Parallax Generator with depth-ordered pixel processing...")
    
    # Create a simple test image (a gradient from black to white)
    width, height = 256, 256
    gradient = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create a horizontal gradient
    for y in range(height):
        for x in range(width):
            gradient[y, x, :] = x / width
    
    # Convert to tensor
    image_tensor = torch.tensor(gradient).unsqueeze(0)  # Add batch dimension
    
    # Create a parallax generator
    parallax_gen = Parallax_Generator_by_SamSeen()
    
    # Generate frames with a small number of frames for testing
    frames, depth_map = parallax_gen.process(
        image_tensor,
        num_frames=5,
        horizontal_shift=20.0,
        vertical_shift=5.0,
        blur_radius=3,
        invert_depth=False,
        clockwise=True,
        loop_type="Forward"
    )
    
    print(f"Generated {frames.shape[0]} frames")
    print("Test completed successfully!")

if __name__ == "__main__":
    test_parallax_generator()
