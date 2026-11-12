# Visualizations
Contains scripts to run inference using trained Scene3D models (`.pth`, `.pt`, or `.onnx`) and create single/batch image visualizations or video visualizations.

# Examples

## image_visualization.py
Read an image file or directory of images, run the Scene3D network, and save or display the viridis depth map overlay.

### Example usage

```bash
python3 Models/visualizations/image_visualization.py \
    -p /path/to/scene3d.pth \
    -i Media/ \
    -o results/images/
```

Or on a single image (displays in OpenCV window if `-o` is omitted):

```bash
python3 Models/visualizations/image_visualization.py \
    -p /path/to/scene3d.pth \
    -i Media/highway_normal_1.png
```

### Parameters:

- `-p`, `--model_checkpoint_path` : path to model weights `.pth`, `.pt`, or `.onnx` file
- `-i`, `--input_image_dirpath` : path to input image file or directory of images (`.jpg`, `.jpeg`, `.png`)
- `-o`, `--output_image_dirpath` : output directory where visualizations (`*_result.png`) will be saved

## video_visualization.py
Read a video file using OpenCV, run the Scene3D network frame-by-frame, and save the visualization frames as a video in `.avi` format, optionally displaying in-progress frames in an OpenCV window.

### Example usage

```bash
python3 Models/visualizations/video_visualization.py \
    -p /path/to/scene3d.pth \
    -i Media/highway_normal.mp4 \
    -o results/highway_normal_output.avi \
    -v
```

### Parameters:

- `-p`, `--model_checkpoint_path` : path to model weights `.pth`, `.pt`, or `.onnx` file
- `-i`, `--input_video_filepath` : path to input video which will be processed
- `-o`, `--output_video_filepath` : path to output video visualization file (`.avi` appended if omitted)
- `-v`, `--vis` : flag for whether to show frame-by-frame visualization while processing is occurring
