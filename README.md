# SSParallaxX - 3D Parallax Animation for ComfyUI

<p>Transform any 2D image into stunning parallax animations with depth-based motion!</p>

<div style="background-color: #f0f7ff; border-left: 5px solid #0078d7; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h3 style="color: #0078d7; margin-top: 0;">🎬 Create Stunning Parallax Animations with Depth-Based Motion!</h3>
  <p>With our advanced depth estimation and integrated video processing tools, creating parallax animations is now easier than ever! Simply:</p>
  <ol>
    <li>Use the <strong>👀 SS Image Uploader</strong> to load your image</li>
    <li>Process it through the <strong>👀 3D Parallax Animation</strong> node to generate frames with depth-based motion</li>
    <li>Combine the frames into a video or GIF with the <strong>👀 SS Images to Video Combiner</strong></li>
  </ol>
  <p>No external depth maps or separate video processing tools needed - everything is handled automatically in one seamless workflow!</p>
</div>
<img src="https://github.com/MrSamSeen/SSParallaxX/blob/main/SS_Video_00001.gif?raw=true" alt="Parallax Animation Example - Transform any 2D image into stunning parallax animations with depth-based motion">
<p>Welcome to the SSParallaxX repository for ComfyUI! 🚀 Create stunning parallax animations that bring your images to life with realistic 3D motion! This powerful depth-estimation node transforms ordinary 2D images into immersive animations where objects move based on their depth in the scene. Perfect for creating eye-catching content that leaps off the screen with realistic depth!</p>

## Introduction

<img src="https://github.com/MrSamSeen/SSParallaxX/blob/main/Screenshot.png?raw=true" alt="Parallax Animation Example - Transform any 2D image into stunning parallax animations with depth-based motion">

<p>The SSParallaxX node is a cutting-edge 2D-to-3D animation tool designed for ComfyUI to generate professional-quality parallax animations with built-in depth map generation. Powered by advanced AI depth estimation, it automatically creates high-quality depth maps from your images, eliminating the need for external depth sources. This tool creates immersive animations where objects appear to move in 3D space based on their depth in the scene, creating a stunning visual effect that brings your images to life!</p>

## Available Nodes

<p>This repository provides powerful parallax animation and video handling nodes for creating stunning 3D motion effects:</p>

### Core Animation Node

<ul>
  <li><strong>👀 3D Parallax Animation by SamSeen</strong>: Our flagship AI-powered depth estimation node that creates stunning parallax animations. Just feed it your standard 2D images and watch as it automatically creates immersive animations with realistic depth perception.</li>
</ul>

### Video Processing Nodes

<ul>
  <li><strong>👀 SS Image Uploader by SamSeen</strong>: A simple yet powerful tool for loading and resizing single images with customizable dimensions, perfect for creating consistent 3D content from various image sources.</li>
  <li><strong>👀 SS Images to Video Combiner by SamSeen</strong>: Combine your processed image sequences back into stunning videos or GIFs with support for multiple formats (MP4, WebM, GIF) and audio integration for complete immersive experiences.</li>
</ul>

## Installation

<p>To install this parallax animation extension for ComfyUI, clone the repository and add it to the custom_nodes folder in your ComfyUI installation directory:</p>

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SamSeen/SSParallaxX.git
cd SSParallaxX
pip install -r requirements.txt
```

<p>After installation, restart ComfyUI or reload the web interface to see the new nodes in the node browser under the "👀 SamSeen" category.</p>

## Detailed Functionality

### 👀 3D Parallax Animation by SamSeen

<p>This node uses advanced depth estimation to automatically generate depth maps from your images, then creates parallax animations where objects move based on their depth:</p>

#### Input Parameters:

<ul>
  <li><strong>base_image</strong>: The input image to animate</li>
  <li><strong>num_frames</strong>: Number of frames to generate (default: 20)</li>
  <li><strong>horizontal_shift</strong>: Maximum horizontal pixel shift (default: 30.0)</li>
  <li><strong>vertical_shift</strong>: Maximum vertical pixel shift (default: 10.0)</li>
  <li><strong>blur_radius</strong>: Smooths the depth map for more natural results (default: 3)</li>
  <li><strong>invert_depth</strong>: Swaps foreground/background if needed (default: false)</li>
  <li><strong>clockwise</strong>: Direction of the elliptical motion (default: true)</li>
  <li><strong>loop_type</strong>: Choose between "Forward" or "Bounce" animation styles</li>
  <li><strong>external_depth_map</strong>: Optional custom depth map for precise control</li>
</ul>

#### Outputs:

<ul>
  <li><strong>frames</strong>: The generated animation frames as an image sequence</li>
  <li><strong>depth_map</strong>: The generated depth map for further processing or analysis</li>
</ul>

### 👀 SS Image Uploader

<p>This node loads and resizes a single image:</p>

#### Input Parameters:

<ul>
  <li><strong>image</strong>: Select an image file to upload</li>
  <li><strong>max_width</strong>: Maximum width for resizing (default: 512)</li>
  <li><strong>max_height</strong>: Maximum height for resizing (0 = auto)</li>
</ul>

#### Outputs:

<ul>
  <li><strong>image</strong>: The loaded and resized image</li>
</ul>

### 👀 SS Images to Video Combiner

<p>This node combines a sequence of images into a video or GIF:</p>

#### Input Parameters:

<ul>
  <li><strong>images</strong>: The input image sequence to combine</li>
  <li><strong>frame_rate</strong>: Frames per second for the output video (default: 30)</li>
  <li><strong>filename_prefix</strong>: Prefix for the output filename</li>
  <li><strong>format</strong>: Output format (mp4, webm, gif)</li>
  <li><strong>save_output</strong>: Whether to save the output to the output directory</li>
  <li><strong>audio</strong>: Optional audio to add to the video</li>
</ul>

#### Outputs:

<ul>
  <li><strong>video_path</strong>: Path to the generated video file</li>
</ul>

## AI Depth Estimation

<p>The 3D Parallax Animation node uses advanced AI models for depth estimation. The depth map is used to determine how much each pixel should move in the parallax animation, with closer objects (lighter in the depth map) moving more than farther objects (darker in the depth map).</p>

<p>The depth estimation model will be downloaded automatically when you first use the node. This requires an internet connection during the initial run.</p>

### Manual Depth Model Installation

<p>If the automatic depth map extraction model download fails, you can manually install the depth estimation model:</p>

<ol>
  <li>Visit the model card on Hugging Face</li>
  <li>Download the model files</li>
  <li>Place them in the appropriate directory</li>
  <li>Restart ComfyUI or reload the web interface</li>
</ol>

### Dependencies

<p>This extension requires the following Python packages:</p>

```
torch>=2.0.0          # Neural network processing
pillow                # Image processing
numpy                 # Numerical processing
opencv-python         # Computer vision
tqdm                  # Progress tracking
```

## Optimizing Your Parallax Animation

<p>To achieve professional-quality parallax animations with realistic depth effects:</p>

<ol>
  <li><strong>Horizontal and Vertical Shift</strong>: Adjust the horizontal_shift and vertical_shift parameters to control the intensity of the parallax effect. The default values (30 horizontal, 10 vertical) create a nice elliptical motion, but you can experiment with different ratios.</li>
  <li><strong>Blur Radius Optimization</strong>: Adjust the blur_radius to smooth depth transitions. Higher values create softer depth boundaries, while lower values maintain sharper depth edges.</li>
  <li><strong>Depth Inversion</strong>: If your scene looks "inside out" (background appears to move more than foreground), toggle the invert_depth option to correct the parallax effect.</li>
  <li><strong>Frame Count</strong>: Increase the num_frames parameter for smoother animations, especially when using the "Bounce" loop type.</li>
  <li><strong>Direction and Loop Type</strong>: Experiment with clockwise/counter-clockwise motion and Forward/Bounce loop types to find the most appealing effect for your specific image.</li>
</ol>

## Transform Your Images with Parallax Animation

<p>With our advanced AI-powered depth map extraction technology, creating professional-quality parallax animations has never been more accessible:</p>

<ul>
  <li>🎬 <strong>Stunning Animations</strong>: Transform static images into dynamic parallax animations with realistic 3D motion</li>
  <li>🔄 <strong>Customizable Effects</strong>: Fine-tune the parallax effect with adjustable horizontal and vertical shift, clockwise or counter-clockwise motion, and different loop types</li>
  <li>🎮 <strong>Content Enhancement</strong>: Create eye-catching animations for social media, presentations, websites, and more</li>
  <li>🖼️ <strong>Depth Visualization</strong>: Use the depth map output to visualize and understand the 3D structure of your images</li>
  <li>📱 <strong>Cross-Platform Compatibility</strong>: Create animations in various formats (GIF, MP4, WebM) for use across different platforms</li>
</ul>

## Example Workflow

<p>Our integrated processing tools make creating parallax animations simple and efficient:</p>

<ol>
  <li><strong>Image Input</strong>: Use the 👀 SS Image Uploader to load your image with customizable resolution</li>
  <li><strong>Parallax Processing</strong>: Process the image with the 👀 3D Parallax Animation node to generate frames with depth-based motion</li>
  <li><strong>Video Output</strong>: Combine the frames into a video or GIF with the 👀 SS Images to Video Combiner</li>
</ol>

## Examples

<div align="center">
  <p><em>Examples of parallax animations created with the SSParallaxX node</em></p>
</div>

## Troubleshooting

<p>If you encounter any issues with the parallax animation:</p>

<ol>
  <li><strong>Depth Map Quality</strong>: If the depth map doesn't accurately represent the scene's depth, try adjusting the blur_radius parameter or providing a custom depth map.</li>
  <li><strong>Motion Intensity</strong>: If the motion is too strong or weak, adjust the horizontal_shift and vertical_shift parameters to find the optimal effect.</li>
  <li><strong>Inverted Depth Perception</strong>: If foreground and background appear swapped, toggle the invert_depth option to correct the parallax effect.</li>
  <li><strong>Model Download Issues</strong>: If the depth model fails to download automatically, follow the manual installation instructions in the documentation.</li>
  <li><strong>Memory Limitations</strong>: For processing large images, consider reducing the input resolution to avoid memory constraints.</li>
</ol>

## Contributing

<p>We welcome contributions to enhance the SSParallaxX nodes! To contribute:</p>

<ol>
  <li>Fork the repository</li>
  <li>Create a feature branch</li>
  <li>Implement your changes</li>
  <li>Submit a pull request</li>
</ol>

<p>Please ensure your code follows our style guidelines and includes appropriate tests.</p>

## License and Attribution

<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>

## Acknowledgments

<ul>
  <li>Thanks to the ComfyUI team for creating an amazing platform for AI image processing</li>
  <li>Special thanks to the creators of the depth estimation models that power this node</li>
</ul>
