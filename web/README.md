# SS Parallax Web Components

This directory contains the web components for the SS Parallax nodes.

## Files

- `js/ss_upload.js`: JavaScript code for the file upload widget
- `index.html`: Documentation page for the nodes

## How it works

The JavaScript code adds custom file upload widgets to the SS Image Uploader node. These widgets allow users to upload files directly from their computer to the ComfyUI input directory.

The file upload widget is implemented using the ComfyUI API and adds a button to the node that opens a file selection dialog. When a file is selected, it is uploaded to the ComfyUI input directory and the node's file selection widget is updated to use the uploaded file.

For image files, a preview of the image is shown in the node. For video files, a thumbnail is shown when available.
