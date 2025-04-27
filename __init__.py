print("Initializing SS Parallax nodes for immersive 3D content creation...")


# Import the Parallax Generator
try:
    from .parallax_generator import Parallax_Generator_by_SamSeen
    print("Successfully imported Parallax_Generator_by_SamSeen")
except Exception as e:
    print(f"Error importing Parallax_Generator_by_SamSeen: {e}")
    # Create a placeholder class
    class Parallax_Generator_by_SamSeen:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"error": ("STRING", {"default": "Error loading Parallax_Generator_by_SamSeen"})}}
        RETURN_TYPES = ("STRING",)
        FUNCTION = "error"
        CATEGORY = "👀 SamSeen"
        def error(self, error):
            return (f"ERROR: {error}",)

# Import the video utility nodes
try:
    # First try to import directly
    from video_utils import SSImageUploader, SSImagesToVideoCombiner
    print("Successfully imported SS utility nodes")
except ImportError:
    try:
        # Then try relative import
        from .video_utils import SSImageUploader, SSImagesToVideoCombiner
        print("Successfully imported SS utility nodes")
    except Exception as e:
        print(f"Error importing SS utility nodes: {e}")
        # Create placeholder classes

        class SSImageUploader:
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"error": ("STRING", {"default": "Error loading SS Image Uploader"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error"
            CATEGORY = "👀 SamSeen"
            def error(self, error):
                return (f"ERROR: {error}",)

        class SSImagesToVideoCombiner:
            @classmethod
            def INPUT_TYPES(s):
                return {"required": {"error": ("STRING", {"default": "Error loading SS Images to Video Combiner"})}}
            RETURN_TYPES = ("STRING",)
            FUNCTION = "error"
            CATEGORY = "👀 SamSeen"
            def error(self, error):
                return (f"ERROR: {error}",)

# Define the node mappings
NODE_CLASS_MAPPINGS = {
    "SS_Image_Uploader": SSImageUploader,
    "SS_Images_to_Video_Combiner": SSImagesToVideoCombiner,
    "Parallax_Generator_by_SamSeen": Parallax_Generator_by_SamSeen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SS_Image_Uploader": "👀 SS Image Uploader by SamSeen",
    "SS_Images_to_Video_Combiner": "👀 SS Images to Video Combiner by SamSeen",
    "Parallax_Generator_by_SamSeen": "👀 3D Parallax Animation by SamSeen"
}

# Define the web directory for custom UI components
import os
WEB_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "web")

# Define the JavaScript files to load
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("SS Parallax nodes initialized successfully! Ready to create amazing 3D animations!")
