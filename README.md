# SS Parallax Animation Nodes for ComfyUI

<p>SS Parallax Animation nodes for ComfyUI by SamSeen - Transform any 2D image into stunning parallax animations!</p>

<div style="background-color: #f0f7ff; border-left: 5px solid #0078d7; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h3 style="color: #0078d7; margin-top: 0;">🎬 Create Stunning Parallax Animations with Depth-Based Motion!</h3>
  <p>With our depth map generation and integrated video processing tools, creating parallax animations is now easier than ever! Simply:</p>
  <ol>
    <li>Use the <strong>👀 SS Image Uploader</strong> to load your image</li>
    <li>Process it through the <strong>👀 3D Parallax Animation</strong> node to generate frames with depth-based motion</li>
    <li>Combine the frames into a video or GIF with the <strong>👀 SS Images to Video Combiner</strong></li>
  </ol>
  <p>No external depth maps or separate video processing tools needed - everything is handled automatically in one seamless workflow!</p>
</div>

<img width="1254" alt="simple_workflow" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/a7ae4c8b-6c38-47b2-a280-84cd851ef254">

<img width="1254" alt="simple_workflow" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/45016b16-81b7-430c-81a5-ec63627398e8">

<p>Welcome to the SS Parallax Animation repository for ComfyUI! 🚀 Create stunning parallax animations that bring your images to life with realistic 3D motion! This powerful depth-estimation node transforms ordinary 2D images into immersive animations where objects move based on their depth in the scene. Perfect for creating eye-catching content that leaps off the screen with realistic depth!</p>

<h2>Introduction</h2>
<p>The SS Parallax Animation Node is a cutting-edge 2D-to-3D animation tool designed for ComfyUI to generate professional-quality parallax animations with built-in depth map generation. Powered by advanced AI depth estimation, it automatically creates high-quality depth maps from your images, eliminating the need for external depth sources. This tool creates immersive animations where objects appear to move in 3D space based on their depth in the scene, creating a stunning visual effect that brings your images to life!</p>

<h2>Available Nodes</h2>
<p>This repository provides powerful parallax animation and video handling nodes for creating stunning 3D motion effects:</p>

<h3>Core Animation Node</h3>
<ul>
  <li><strong>👀 3D Parallax Animation by SamSeen</strong>: Our flagship AI-powered depth estimation node that creates stunning parallax animations. Just feed it your standard 2D images and watch as it automatically creates immersive animations with realistic depth perception.</li>
</ul>

<h3>Video Processing Nodes</h3>
<ul>
  <li><strong>👀 SS Image Uploader by SamSeen</strong>: A simple yet powerful tool for loading and resizing single images with customizable dimensions, perfect for creating consistent 3D content from various image sources.</li>
  <li><strong>👀 SS Images to Video Combiner by SamSeen</strong>: Combine your processed image sequences back into stunning videos or GIFs with support for multiple formats (MP4, WebM, GIF) and audio integration for complete immersive experiences.</li>
</ul>

<h2>Installation</h2>
<p>To install this parallax animation extension for ComfyUI, clone the repository and add it to the custom_nodes folder in your ComfyUI installation directory:</p>
<pre>git clone https://github.com/SamSeen/SSParallaxX.git
cd SSParallaxX
pip install -r requirements.txt
</pre>

<p>After installation, restart ComfyUI or reload the web interface to see the new nodes in the node browser under the "👀 SamSeen" category.</p>

<h2>Detailed Functionality</h2>

<h3>👀 3D Parallax Animation by SamSeen</h3>
<p>This node uses advanced depth estimation to automatically generate depth maps from your images, then creates parallax animations where objects move based on their depth:</p>

<p><strong>Input Parameters:</strong></p>
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

<p><strong>Outputs:</strong></p>
<ul>
  <li><strong>frames</strong>: The generated animation frames as an image sequence</li>
  <li><strong>depth_map</strong>: The generated depth map for further processing or analysis</li>
</ul>

<h3>👀 SS Images to Video Combiner</h3>
<p>This node combines a sequence of images into a video or GIF:</p>

<p><strong>Input Parameters:</strong></p>
<ul>
  <li><strong>images</strong>: The input image sequence to combine</li>
  <li><strong>frame_rate</strong>: Frames per second for the output video (default: 30)</li>
  <li><strong>filename_prefix</strong>: Prefix for the output filename</li>
  <li><strong>format</strong>: Output format (mp4, webm, gif)</li>
  <li><strong>save_output</strong>: Whether to save the output to the output directory</li>
  <li><strong>audio</strong>: Optional audio to add to the video</li>
</ul>

<p><strong>Outputs:</strong></p>
<ul>
  <li><strong>video_path</strong>: Path to the generated video file</li>
</ul>

<h2>AI Depth Estimation</h2>

<p>The 3D Parallax Animation node uses advanced AI models for depth estimation. The depth map is used to determine how much each pixel should move in the parallax animation, with closer objects (lighter in the depth map) moving more than farther objects (darker in the depth map).</p>

<p>The depth estimation model will be downloaded automatically when you first use the node. This requires an internet connection during the initial run.</p>

<h3>Manual Depth Model Installation</h3>
<p>If the automatic depth map extraction model download fails, you can manually install the depth estimation model:</p>

<ol>
  <li>Visit the model card on Hugging Face</li>
  <li>Download the model files</li>
  <li>Place them in the appropriate directory</li>
  <li>Restart ComfyUI or reload the web interface</li>
</ol>

<h3>Dependencies</h3>
<p>This extension requires the following Python packages:</p>
<pre>
torch>=2.0.0          # Neural network processing
pillow                # Image processing
numpy                 # Numerical processing
opencv-python         # Computer vision
tqdm                  # Progress tracking
</pre>

<h2>Optimizing Your Parallax Animation</h2>
<p>To achieve professional-quality parallax animations with realistic depth effects:</p>
<ol>
  <li><strong>Horizontal and Vertical Shift</strong>: Adjust the horizontal_shift and vertical_shift parameters to control the intensity of the parallax effect. The default values (30 horizontal, 10 vertical) create a nice elliptical motion, but you can experiment with different ratios.</li>
  <li><strong>Blur Radius Optimization</strong>: Adjust the blur_radius to smooth depth transitions. Higher values create softer depth boundaries, while lower values maintain sharper depth edges.</li>
  <li><strong>Depth Inversion</strong>: If your scene looks "inside out" (background appears to move more than foreground), toggle the invert_depth option to correct the parallax effect.</li>
  <li><strong>Frame Count</strong>: Increase the num_frames parameter for smoother animations, especially when using the "Bounce" loop type.</li>
  <li><strong>Direction and Loop Type</strong>: Experiment with clockwise/counter-clockwise motion and Forward/Bounce loop types to find the most appealing effect for your specific image.</li>
</ol>

<h2>Transform Your Images with Parallax Animation</h2>
<p>With our advanced AI-powered depth map extraction technology, creating professional-quality parallax animations has never been more accessible:</p>
<ul>
  <li>🎬 <strong>Stunning Animations:</strong> Transform static images into dynamic parallax animations with realistic 3D motion</li>
  <li>🔄 <strong>Customizable Effects:</strong> Fine-tune the parallax effect with adjustable horizontal and vertical shift, clockwise or counter-clockwise motion, and different loop types</li>
  <li>🎮 <strong>Content Enhancement:</strong> Create eye-catching animations for social media, presentations, websites, and more</li>
  <li>🖼️ <strong>Depth Visualization:</strong> Use the depth map output to visualize and understand the 3D structure of your images</li>
  <li>📱 <strong>Cross-Platform Compatibility:</strong> Create animations in various formats (GIF, MP4, WebM) for use across different platforms</li>
</ul>

<h2>Experiencing Your Stereoscopic 3D Creations</h2>

<p>There are multiple ways to enjoy the immersive depth perception in your side-by-side stereoscopic content:</p>

<ol>
  <li><strong>Cross-eyed Free-Viewing Technique</strong>: Focus on a point between the two side-by-side images and gradually cross your eyes until the images merge into a single 3D image with realistic depth in the center - a popular no-glasses 3D viewing method.</li>
  <li><strong>Parallel Viewing for Binocular Vision</strong>: For larger stereoscopic images, relax your eyes as if looking at a distant object, allowing the side-by-side images to naturally merge into a depth-rich 3D scene.</li>
  <li><strong>Virtual Reality Headsets</strong>: Load your parallax animations into any VR viewer app for a fully immersive depth perception experience with proper spatial awareness.</li>
  <li><strong>Specialized 3D Displays</strong>: View your stereoscopic content on 3D-capable monitors, TVs, or projectors that support side-by-side 3D format for professional-quality depth visualization.</li>
  <li><strong>Mobile 3D Applications</strong>: Many smartphone apps can transform your side-by-side content into immersive 3D experiences using various stereoscopic viewing methods, perfect for sharing your 3D creations.</li>
  <li><strong>Stereoscopic Viewers</strong>: Use dedicated stereoscopic viewers or simple cardboard viewers that help your eyes focus on the correct side-by-side image pairs for enhanced depth perception.</li>
</ol>

<h2>Troubleshooting Stereoscopic Generation Issues</h2>

<p>If you encounter any challenges with your 3D content creation, try these solutions:</p>

<ol>
  <li><strong>Depth Estimation Model Download Issues</strong>: Verify your internet connection or follow the manual AI model installation instructions above for proper depth map extraction.</li>
  <li><strong>GPU Memory Limitations</strong>: For high-resolution stereoscopic processing, reduce your image dimensions or switch to CPU mode if you have limited graphics memory for AI depth estimation.</li>
  <li><strong>Depth Map Quality Enhancement</strong>: Fine-tune the blur_radius parameter to achieve smoother depth transitions and more natural binocular vision effects in your stereoscopic output.</li>
  <li><strong>Depth Perception Inversion</strong>: If foreground and background elements appear with incorrect spatial relationships, toggle the invert_depth parameter to correct the stereoscopic effect.</li>
  <li><strong>ComfyUI Compatibility</strong>: For optimal 3D content generation, ensure you're using ComfyUI version 1.5.0 or higher with all required dependencies for proper depth-aware processing.</li>
  <li><strong>Stereoscopic Comfort</strong>: If viewing causes eye strain, reduce the depth_scale parameter to create more comfortable side-by-side 3D content with less extreme depth differences.</li>
</ol>

<h2>Stereoscopic 3D Workflow Examples</h2>
<p>Create professional-quality side-by-side 3D content with these example ComfyUI workflows:</p>
<img width="1254" alt="3D conversion workflow with depth estimation" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/6b972838-07ca-4e64-a3a4-32277ddcf4c7">
<p><em>Basic stereoscopic workflow: Generate depth maps automatically and create immersive 3D content with AI-powered depth perception</em></p>

<h2>Advanced 3D Content Creation Workflow</h2>
<img width="696" alt="Advanced stereoscopic processing pipeline" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/1272a5ad-9d12-4b86-8284-08bbe48bd116">
<p><em>Enhanced 3D generation workflow: Combine multiple processing steps for professional stereoscopic results with precise depth control</em></p>

<h2>Complete 3D Video Processing Workflow</h2>
<p>Our integrated video processing tools make creating stereoscopic 3D videos simple and efficient:</p>
<ol>
  <li><strong>Image Input</strong>: Use the 👀 SS Image Uploader to load your image with customizable resolution</li>
  <li><strong>Parallax Processing</strong>: Process the image with the 👀 3D Parallax Animation node to generate frames with depth-based motion</li>
  <li><strong>Video Output</strong>: Combine the frames into a video or GIF with the 👀 SS Images to Video Combiner</li>
</ol>

<h2>License and Attribution</h2>
<p>This project is licensed under the MIT License - see the LICENSE file for details.</p>

<h2>Acknowledgments</h2>
<ul>
  <li>Thanks to the ComfyUI team for creating an amazing platform for AI image processing</li>
  <li>Special thanks to the creators of the depth estimation models that power this node</li>
</ul>
<p><em>This end-to-end workflow eliminates the need for external video processing tools, making professional 3D video creation accessible to everyone!</em></p>

<h2>Stereoscopic 3D Gallery</h2>
<p>Examples of side-by-side 3D images created with our depth-aware image processing technology:</p>
<img width="1254" alt="Stereoscopic landscape with depth perception" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/ee30f773-bb90-420f-a1a2-25459b678bbe">
<p><em>Landscape with AI-generated depth map for immersive stereoscopic viewing</em></p>

<img width="1254" alt="Side-by-side 3D portrait with binocular vision effect" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/e702dde6-b675-4ff6-842c-1c32116be313">
<p><em>Portrait with enhanced depth perception for realistic 3D effect</em></p>

<img width="1254" alt="Virtual reality compatible stereoscopic image" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/022223be-fd55-400a-a1ba-38628265a119">
<p><em>VR-ready stereoscopic image with optimized depth for comfortable viewing</em></p>

<img width="1254" alt="Cross-eyed viewing 3D image with depth map" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/e711ba22-a393-4580-ad37-43934373e835">
<p><em>Detailed scene with precise depth estimation for cross-eyed free-viewing</em></p>

<img width="1254" alt="Side-by-side 3D animation frame with depth" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/59ff5c1c-ceda-42d9-a8eb-b27ffbb0e8ca">
<p><em>Animation frame converted to stereoscopic 3D with automatic depth extraction</em></p>

<img width="1254" alt="Immersive stereoscopic scene with AI depth" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/8b6d7c5e-aefc-4f22-9872-73b690514497">
<p><em>Complex scene with AI-powered depth perception for immersive stereoscopic experience</em></p>

<img width="1254" alt="Professional 3D content for specialized displays" src="https://github.com/MrSamSeen/ComfyUI_SSStereoscope/assets/8541558/46182b48-f5cb-4d6f-9ae9-ba727da92569">
<p><em>Professional-quality stereoscopic image optimized for 3D displays and VR headsets</em></p>

<p><a href="https://civitai.com/models/546410?modelVersionId=607731">Explore more stereoscopic 3D examples on CivitAI</a></p>

<h2>Join Our 3D Content Creation Community</h2>

<p>We welcome contributions to enhance this stereoscopic imaging tool! Whether you're experienced with depth estimation algorithms, 3D visualization techniques, or just passionate about creating immersive content, feel free to submit pull requests or open issues with your suggestions for improving the depth-aware processing capabilities.</p>

<h2>Open Source License</h2>

<p>This stereoscopic 3D conversion project is licensed under the MIT License - see the LICENSE file for complete details. Feel free to use our depth estimation technology in your own 3D content creation projects while providing appropriate attribution.</p>

<h2>About This Project</h2>

<p>This ComfyUI custom node represents the cutting edge of AI-powered stereoscopic 3D content creation. By leveraging advanced depth estimation technology, it enables seamless 2D to 3D conversion without requiring specialized hardware or external depth maps. Whether you're creating immersive virtual reality experiences, enhancing your photography with depth perception, or producing eye-catching side-by-side 3D animations, this tool streamlines your stereoscopic workflow.</p>

<p>The depth-aware imaging capabilities automatically identify foreground and background elements, creating realistic depth perception that brings your images to life. Perfect for 3D visualization projects, cross-eyed viewing techniques, or parallel view stereograms that don't require special glasses. From single images to batch processing entire animation sequences, transform your media creation process with professional-quality binocular vision effects.</p>
