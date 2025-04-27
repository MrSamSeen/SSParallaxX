#!/usr/bin/env python3
"""
Test script for the Parallax Generator node.
This script tests the basic functionality of the parallax generator.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a mock ProgressBar class to avoid import errors
class MockProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0
        print(f"Progress: 0/{total}")

    def update(self, n=1):
        self.current += n
        print(f"Progress: {self.current}/{self.total}")

# Create a more sophisticated mock DepthEstimator class
class MockDepthEstimator:
    def __init__(self):
        self.blur_radius = 15
        self.edge_weight = 0.5
        self.gradient_weight = 0.0
        print("Initialized mock depth estimator")

    def load_model(self):
        """Mock method to match the original DepthEstimator interface"""
        return False

    def predict_depth(self, image):
        """
        Generate a depth map that mimics the output of Depth-Anything-V2.
        Lighter areas represent objects closer to the camera, darker areas are farther away.
        """
        print("Generating realistic mock depth map...")

        # Make sure blur_radius is odd (required by GaussianBlur)
        blur_radius = self.blur_radius
        if blur_radius % 2 == 0:
            blur_radius += 1

        # Get image dimensions
        if len(image.shape) == 3:
            h, w, _ = image.shape
        else:
            h, w = image.shape

        # Convert to grayscale if needed and ensure it's uint8 format
        if len(image.shape) == 3:
            # Convert to uint8 if it's not already
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
            else:
                image_uint8 = image
            gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        else:
            # Convert to uint8 if it's not already
            if image.dtype != np.uint8:
                gray = (image * 255).astype(np.uint8)
            else:
                gray = image.copy()

        # Step 1: Create a base depth map using multiple cues

        # 1.1: Use brightness as a depth cue (brighter areas often closer)
        brightness = gray.astype(np.float32) / 255.0

        # 1.2: Use blur detection as a depth cue (blurrier areas often farther)
        # Laplacian variance is lower for blurry regions
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_map = cv2.GaussianBlur(np.abs(laplacian), (blur_radius, blur_radius), 0)
        blur_map = blur_map / (np.max(blur_map) + 1e-5)  # Normalize

        # 1.3: Use edge detection (edges are usually object boundaries)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges.astype(np.float32) / 255.0

        # Dilate edges to make them more prominent
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.GaussianBlur(edges, (blur_radius, blur_radius), 0)

        # 1.4: Use focus detection (in-focus areas often closer)
        # Sobel filters to detect high-frequency content
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        focus_map = np.sqrt(sobelx**2 + sobely**2)
        focus_map = focus_map / (np.max(focus_map) + 1e-5)  # Normalize

        # 1.5: Use center-bias (objects in center often closer)
        center_y, center_x = h // 2, w // 2
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_bias = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # Calculate distance from center (normalized to 0-1)
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2) / max_dist
                # Closer to center = closer to viewer (higher value)
                center_bias[y, x] = 1.0 - dist

        # Step 2: Combine all depth cues with appropriate weights
        # These weights can be adjusted based on what works best
        depth = (
            brightness * 0.2 +        # Brightness cue
            focus_map * 0.3 +         # Focus cue
            edges * 0.3 +             # Edge cue
            center_bias * 0.2         # Center bias
        )

        # Step 3: Post-process the depth map

        # 3.1: Normalize to [0, 1] range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)

        # 3.2: Apply Gaussian blur to smooth the depth map
        depth = cv2.GaussianBlur(depth, (blur_radius, blur_radius), 0)

        # 3.3: Enhance contrast to make the depth map more pronounced
        depth = np.power(depth, 0.8)  # Gamma correction to enhance contrast

        # 3.4: Final normalization
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)

        print(f"Realistic mock depth map generated with shape: {depth.shape}")
        return depth

# Create a mock folder_paths class
class MockFolderPaths:
    def get_output_directory(self):
        return "output"
    def get_temp_directory(self):
        return "temp"
    def get_input_directory(self):
        return "input"

# Create a mock tqdm function
def mock_tqdm(iterable):
    return iterable

# Add the mocks to sys.modules to avoid import errors
sys.modules['comfy.utils'] = type('MockComfyUtils', (), {'ProgressBar': MockProgressBar})
sys.modules['depth_estimator'] = type('MockDepthEstimatorModule', (), {'DepthEstimator': MockDepthEstimator})
sys.modules['folder_paths'] = MockFolderPaths()
sys.modules['tqdm'] = type('MockTqdm', (), {'tqdm': mock_tqdm})

# Now try to import our Parallax Generator node
try:
    from parallax_generator import Parallax_Generator_by_SamSeen
    print("Successfully imported Parallax_Generator_by_SamSeen")
except ImportError as e:
    print(f"Error importing Parallax_Generator_by_SamSeen: {e}")
    sys.exit(1)

def main():
    """
    Test the Parallax Generator node with a sample image.
    """
    # Check if an image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Look for any image in the current directory
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if image_files:
            image_path = image_files[0]
            print(f"Using image: {image_path}")
        else:
            print("No image files found in the current directory.")
            print("Please provide an image path as an argument.")
            sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # Load the image
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    print(f"Image shape: {img_np.shape}")

    # Convert to tensor format (simulate ComfyUI input)
    import torch
    img_tensor = torch.tensor(img_np.astype(np.float32) / 255.0).unsqueeze(0)  # Add batch dimension

    # Create an instance of our Parallax Generator node
    parallax_node = Parallax_Generator_by_SamSeen()

    # Set parameters
    num_frames = 10
    max_shift = 30.0
    blur_radius = 15
    invert_depth = False
    direction = "Horizontal"
    loop_type = "Bounce"
    output_format = "Frames"  # Just generate frames, don't create a video
    fps = 24

    # Process the image
    print(f"Processing image with parameters: num_frames={num_frames}, max_shift={max_shift}, blur_radius={blur_radius}, direction={direction}, loop_type={loop_type}")
    frames, depth_map = parallax_node.process(
        img_tensor,
        num_frames,
        max_shift,
        blur_radius,
        invert_depth,
        direction,
        loop_type,
        output_format,
        fps
    )

    # Create output directory if it doesn't exist
    output_dir = "parallax_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # Save the frames
    print(f"Saving {frames.shape[0]} frames to {output_dir}...")
    for i in range(frames.shape[0]):
        frame_np = frames[i].cpu().numpy() * 255.0
        frame_img = Image.fromarray(frame_np.astype(np.uint8))
        frame_img.save(os.path.join(output_dir, f"frame_{i:03d}.png"))

    # Save the depth map
    depth_np = depth_map[0].cpu().numpy() * 255.0
    depth_img = Image.fromarray(depth_np.astype(np.uint8))
    depth_img.save(os.path.join(output_dir, "depth_map.png"))
    print(f"Depth map saved to: {os.path.join(output_dir, 'depth_map.png')}")

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
