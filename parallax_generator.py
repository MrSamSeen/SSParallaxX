import torch
from PIL import Image
import numpy as np
import tqdm
import cv2
from comfy.utils import ProgressBar

# Import depth estimation if available
try:
    from .depth_estimator import DepthEstimator
    print("Successfully imported DepthEstimator")
except ImportError:
    print("Warning: DepthEstimator not found, using fallback depth map generation")

    # Create a more sophisticated fallback depth estimator based on the original code
    class FallbackDepthEstimator:
        def __init__(self):
            self.blur_radius = 15
            self.edge_weight = 0.5
            self.gradient_weight = 0.0
            print("Initialized fallback depth estimator")

        def load_model(self):
            """Mock method to match the original DepthEstimator interface"""
            return False

        def predict_depth(self, image):
            """
            Generate a depth map that mimics the output of Depth-Anything-V2.
            Lighter areas represent objects closer to the camera, darker areas are farther away.
            """
            print("Generating realistic fallback depth map...")

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

                # No need for color version
            else:
                # Convert to uint8 if it's not already
                if image.dtype != np.uint8:
                    gray = (image * 255).astype(np.uint8)
                else:
                    gray = image.copy()

                # No need for color version

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

            print(f"Realistic fallback depth map generated with shape: {depth.shape}")
            return depth

    DepthEstimator = FallbackDepthEstimator

# No need for video utilities since we're only outputting frames

class Parallax_Generator_by_SamSeen:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model = None
        self.original_depths = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "num_frames": ("INT", {"default": 120, "min": 5, "max": 1200, "step": 1}),
                "horizontal_shift": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "vertical_shift": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "blur_radius": ("INT", {"default": 3, "min": 1, "max": 51, "step": 2}),
                "invert_depth": ("BOOLEAN", {"default": True}),
                "clockwise": ("BOOLEAN", {"default": True}),
                "loop_type": (["Forward", "Bounce"], {"default": "Forward"}),
            },
            "optional": {
                "external_depth_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("frames", "depth_map")
    FUNCTION = "process"
    CATEGORY = "👀 SamSeen"
    DESCRIPTION = "Create stunning parallax animations with elliptical motion using depth maps. Generates a sequence of frames where objects appear to move in a realistic 3D way based on their depth. Use with SS_Images_to_Video_Combiner to create videos or GIFs."

    def generate_depth_map(self, image_tensor):
        """
        Generate a depth map from an image tensor.

        Parameters:
        - image_tensor: tensor representing the image(s) with shape [B,H,W,C].

        Returns:
        - depth_map: tensor representing the depth map(s).
        """
        # Initialize depth model if needed
        if self.depth_model is None:
            print("Initializing depth model...")
            self.depth_model = DepthEstimator()

        # Get batch size
        B = image_tensor.shape[0]

        # Process each image in the batch
        out = []
        self.original_depths = []
        pbar = ProgressBar(B)

        for b in range(B):
            # Get current image from batch
            current_image = image_tensor[b].cpu().numpy()

            # Check if the image is in float format (0-1 range)
            if current_image.dtype == np.float32 or current_image.dtype == np.float64:
                # Check if values are in 0-1 range
                if np.max(current_image) <= 1.0:
                    print(f"Image is in float format (0-1 range), converting to uint8 for depth estimation")
                    # No need to convert here, the predict_depth method will handle it
                else:
                    print(f"Image has unusual value range: min={np.min(current_image)}, max={np.max(current_image)}")

            # Generate depth map
            depth = self.depth_model.predict_depth(current_image)

            # Save the original depth map for the parallax generation
            self.original_depths.append(depth.copy())

            # Convert to tensor format - keep as grayscale
            depth_tensor = torch.from_numpy(depth).float().unsqueeze(0)  # Add channel dimension

            out.append(depth_tensor)
            pbar.update(1)

        # Stack the results to create a batched tensor
        depth_map_batch = torch.stack(out)

        return depth_map_batch

    def process(self, base_image, num_frames, horizontal_shift, vertical_shift, blur_radius,
                invert_depth=False, clockwise=True, loop_type="Bounce", external_depth_map=None):
        """
        Create a parallax animation from a standard image with elliptical motion.

        Parameters:
        - base_image: tensor representing the base image(s) with shape [B,H,W,C].
        - num_frames: number of frames to generate.
        - horizontal_shift: maximum horizontal pixel shift for the parallax effect.
        - vertical_shift: maximum vertical pixel shift for the parallax effect.
        - blur_radius: integer controlling the smoothness of the depth map.
        - invert_depth: boolean to invert the depth map (swap foreground/background).
        - clockwise: boolean to control the direction of the elliptical motion.
        - loop_type: type of loop for the animation ("Forward" or "Bounce").
        - external_depth_map: optional external depth map.

        Returns:
        - frames: the generated frames as an image sequence.
        - depth_map: the depth map used for parallax.
        """
        # Update the depth model parameters if using internal depth model
        if self.depth_model is not None:
            self.depth_model.blur_radius = blur_radius

        # Generate or use external depth map
        if external_depth_map is not None:
            print("Using external depth map...")
            depth_map = external_depth_map

            # Store the depth map for parallax generation
            B = base_image.shape[0]
            self.original_depths = []

            for b in range(B):
                current_depth = depth_map[b].cpu().numpy()

                # If we have a colored depth map, use the red channel
                if len(current_depth.shape) == 3 and current_depth.shape[2] == 3:
                    current_depth = current_depth[:, :, 0]

                self.original_depths.append(current_depth)
        else:
            print(f"Generating depth map with blur_radius={blur_radius}, invert_depth={invert_depth}...")
            depth_map = self.generate_depth_map(base_image)

        # Get batch size
        B = base_image.shape[0]

        # Process each image in the batch
        all_frames = []
        enhanced_depth_maps = []

        for b in range(B):
            # Get current image from batch
            current_image = base_image[b].cpu().numpy()
            current_image_pil = Image.fromarray((current_image * 255).astype(np.uint8))

            # Get the current depth map
            if hasattr(self, 'original_depths') and len(self.original_depths) > b:
                # Use the original grayscale depth map for this image in the batch
                depth_for_parallax = self.original_depths[b].copy()
                print(f"Using depth map for image {b+1}/{B}: shape={depth_for_parallax.shape}, min={np.min(depth_for_parallax)}, max={np.max(depth_for_parallax)}")
            else:
                # If original depth is not available, extract from the colored version
                current_depth_map = depth_map[b].cpu().numpy()

                # If we have a colored depth map, use the red channel
                if len(current_depth_map.shape) == 3 and current_depth_map.shape[2] == 3:
                    depth_for_parallax = current_depth_map[:, :, 0].copy()
                else:
                    depth_for_parallax = current_depth_map.copy()

            # Invert depth if requested (swap foreground/background)
            if invert_depth:
                print("Inverting depth map (swapping foreground/background)")
                depth_for_parallax = 1.0 - depth_for_parallax

            # Convert the depth map to a PIL image for processing
            depth_map_img = Image.fromarray((depth_for_parallax * 255).astype(np.uint8), mode='L')

            # Get dimensions and resize depth map to match base image
            width, height = current_image_pil.size
            depth_map_img = depth_map_img.resize((width, height), Image.NEAREST)

            # Calculate the number of frames based on loop type
            actual_frames = num_frames
            if loop_type == "Bounce":
                # For bounce, we need half the frames (we'll mirror them later)
                actual_frames = max(2, num_frames // 2)

            # Generate frames with increasing parallax effect
            frames = []
            pbar = ProgressBar(actual_frames)
            print(f"Generating {actual_frames} frames with horizontal_shift={horizontal_shift}, vertical_shift={vertical_shift}...")

            # Calculate angle step for each frame (full circle = 2*pi)
            angle_step = 2 * np.pi / num_frames

            for frame_idx in range(actual_frames):
                # Calculate the current angle based on frame index
                # If clockwise is True, angle increases; otherwise, it decreases
                if clockwise:
                    current_angle = frame_idx * angle_step
                else:
                    current_angle = 2 * np.pi - (frame_idx * angle_step)

                # Calculate horizontal and vertical shift components using sine and cosine
                # to create an elliptical motion pattern
                current_h_shift = horizontal_shift * np.cos(current_angle)
                current_v_shift = vertical_shift * np.sin(current_angle)

                # Create a new image for the current frame
                frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Apply parallax effect based on depth map
                for y in range(height):
                    for x in range(width):
                        # Get depth value at this pixel
                        depth_value = depth_map_img.getpixel((x, y))
                        if isinstance(depth_value, tuple):
                            depth_value = depth_value[0]

                        # Normalize depth value to 0-1
                        depth_value = depth_value / 255.0

                        # Calculate pixel shift based on depth and current elliptical components
                        h_pixel_shift = depth_value * current_h_shift
                        v_pixel_shift = depth_value * current_v_shift

                        # Apply shift in both directions
                        new_x = int(x + h_pixel_shift)
                        new_y = int(y + v_pixel_shift)

                        # Ensure coordinates are within bounds
                        new_x = max(0, min(width - 1, new_x))
                        new_y = max(0, min(height - 1, new_y))

                        # Copy pixel from original image to new position
                        frame[new_y, new_x] = current_image_pil.getpixel((x, y))

                # Fill any holes in the frame (pixels that weren't assigned)
                frame = self.fill_holes(frame)

                # Convert to tensor
                frame_tensor = torch.tensor(frame.astype(np.float32) / 255.0)
                frames.append(frame_tensor)
                pbar.update(1)

            # If bounce mode, add reversed frames (excluding the last frame to avoid duplication)
            if loop_type == "Bounce" and len(frames) > 1:
                reversed_frames = frames[-2::-1]  # Exclude the last frame and reverse
                frames.extend(reversed_frames)

            # Add to our batch lists
            all_frames.extend(frames)

            # Create a properly formatted depth map for output
            # Make sure it's normalized to [0,1]
            if np.min(depth_for_parallax) < 0 or np.max(depth_for_parallax) > 1:
                depth_gray = cv2.normalize(depth_for_parallax, None, 0, 1, cv2.NORM_MINMAX)
            else:
                depth_gray = depth_for_parallax

            # Convert to 3-channel grayscale (all channels have same value)
            depth_3ch = np.stack([depth_gray, depth_gray, depth_gray], axis=-1)

            # Convert to tensor format
            enhanced_depth_map = torch.tensor(depth_3ch)
            enhanced_depth_maps.append(enhanced_depth_map)

        # Stack the results to create batched tensors
        frames_batch = torch.stack(all_frames)
        enhanced_depth_maps_batch = torch.stack(enhanced_depth_maps)

        # Print final output stats
        print(f"Generated {frames_batch.shape[0]} frames with shape {frames_batch.shape[1:]}")
        print(f"Depth map shape: {enhanced_depth_maps_batch.shape}")

        return (frames_batch, enhanced_depth_maps_batch)

    def fill_holes(self, frame):
        """
        Fill holes in the frame caused by parallax shifting.
        Uses a simple nearest neighbor approach.

        Parameters:
        - frame: numpy array representing the frame.

        Returns:
        - filled_frame: numpy array with holes filled.
        """
        # Create a mask of empty pixels (all zeros)
        mask = np.all(frame == 0, axis=2)

        # If no holes, return the original frame
        if not np.any(mask):
            return frame

        # Create a copy of the frame
        filled_frame = frame.copy()

        # Get coordinates of empty pixels
        y_empty, x_empty = np.where(mask)

        # For each empty pixel, find the nearest non-empty pixel
        for y, x in zip(y_empty, x_empty):
            # Simple approach: check neighbors in a small window
            window_size = 3
            best_dist = float('inf')
            best_color = None

            for dy in range(-window_size, window_size + 1):
                for dx in range(-window_size, window_size + 1):
                    ny, nx = y + dy, x + dx

                    # Check if neighbor is within bounds and not empty
                    if (0 <= ny < frame.shape[0] and 0 <= nx < frame.shape[1] and
                        not np.all(frame[ny, nx] == 0)):

                        # Calculate distance
                        dist = dx*dx + dy*dy

                        if dist < best_dist:
                            best_dist = dist
                            best_color = frame[ny, nx]

            # If a neighbor was found, use its color
            if best_color is not None:
                filled_frame[y, x] = best_color
            else:
                # If no neighbor was found, use a more aggressive search
                # This is a fallback and should rarely happen
                non_empty_y, non_empty_x = np.where(~mask)
                if len(non_empty_y) > 0:
                    # Find the nearest non-empty pixel
                    distances = (non_empty_y - y)**2 + (non_empty_x - x)**2
                    nearest_idx = np.argmin(distances)
                    filled_frame[y, x] = frame[non_empty_y[nearest_idx], non_empty_x[nearest_idx]]

        return filled_frame
