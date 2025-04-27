#!/usr/bin/env python3
"""
Test script for the video utility nodes.
This script tests the basic functionality of the video nodes.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from video_utils import SSImageUploader, SSImagesToVideoCombiner
    print("Successfully imported video utility nodes")
except Exception as e:
    print(f"Error importing video utility nodes: {e}")
    sys.exit(1)

# Video uploader test removed as it's no longer used

def test_image_uploader():
    """Test the SSImageUploader class"""
    print("\nTesting SSImageUploader...")

    # Create a dummy image file for testing
    # This would normally be in the input directory
    print("Note: This test requires an image file in the input directory")

    # Print the INPUT_TYPES to verify it's working
    input_types = SSImageUploader.INPUT_TYPES()
    print(f"INPUT_TYPES: {input_types}")

    print("SSImageUploader test completed")

def test_video_combiner():
    """Test the SSImagesToVideoCombiner class"""
    print("\nTesting SSImagesToVideoCombiner...")

    # Print the INPUT_TYPES to verify it's working
    input_types = SSImagesToVideoCombiner.INPUT_TYPES()
    print(f"INPUT_TYPES: {input_types}")

    # Create a dummy tensor for testing
    dummy_tensor = torch.zeros((5, 64, 64, 3))

    print("SSImagesToVideoCombiner test completed")

def main():
    """Main function"""
    print("Testing video utility nodes...")

    test_image_uploader()
    test_video_combiner()

    print("\nAll tests completed")

if __name__ == "__main__":
    main()
