"""
Test script for the edge fixing options in parallax_generator.py
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

def test_edge_fixing():
    """Test the parallax generator with different edge fixing methods"""
    print("Testing Parallax Generator with edge fixing options...")

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

    # Test each edge fixing method
    edge_fix_methods = ["None", "Crop and Zoom", "Ease-In"]

    # Define test cases with different horizontal and vertical shift values
    test_cases = [
        {"name": "balanced", "h_shift": 30.0, "v_shift": 30.0},
        {"name": "horizontal_dominant", "h_shift": 50.0, "v_shift": 10.0},
        {"name": "vertical_dominant", "h_shift": 10.0, "v_shift": 50.0}
    ]

    for method in edge_fix_methods:
        print(f"\nTesting edge fix method: {method}")

        for test_case in test_cases:
            case_name = test_case["name"]
            h_shift = test_case["h_shift"]
            v_shift = test_case["v_shift"]

            print(f"\nTest case: {case_name}")
            print(f"Testing with horizontal_shift={h_shift}, vertical_shift={v_shift}")

            # Generate frames with the current edge fix method
            frames, depth_map = parallax_gen.process(
                image_tensor,
                num_frames=5,
                horizontal_shift=h_shift,
                vertical_shift=v_shift,
                blur_radius=3,
                invert_depth=False,
                clockwise=True,
                loop_type="Forward",
                edge_fix_method=method
            )

            print(f"Generated {frames.shape[0]} frames with {method} edge fixing")

            # Create output directory if it doesn't exist
            output_dir = f"edge_fix_test_{method.replace(' ', '_').lower()}_{case_name}"
            os.makedirs(output_dir, exist_ok=True)

            # Save the frames
            for i in range(frames.shape[0]):
                frame_np = frames[i].cpu().numpy() * 255.0
                frame_img = Image.fromarray(frame_np.astype(np.uint8))
                frame_img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

            # Save the depth map
            depth_np = depth_map[0].cpu().numpy() * 255.0
            depth_img = Image.fromarray(depth_np.astype(np.uint8))
            depth_img.save(os.path.join(output_dir, "depth_map.png"))

    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_edge_fixing()
