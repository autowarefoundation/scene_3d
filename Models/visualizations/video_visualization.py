# %%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.visualizations.image_visualization import (
    FRAME_INF_SIZE,
    load_model,
    make_visualization,
)

OUTPUT_VIDEO_SIZE = (1280, 720)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        dest="model_checkpoint_path",
        required=True,
        help="Path to PyTorch (.pth/.pt) checkpoint or ONNX (.onnx) model file.",
    )
    parser.add_argument(
        "-i",
        "--input_video_filepath",
        dest="input_video_filepath",
        required=True,
        help="Path to input video which will be processed by Scene3D.",
    )
    parser.add_argument(
        "-o",
        "--output_video_filepath",
        dest="output_video_filepath",
        required=True,
        help="Path to output video visualization file, must include output file name.",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        help="Flag for whether to show frame by frame visualization while processing is occurring.",
    )
    args = parser.parse_args()

    model = load_model(args.model_checkpoint_path)
    print("Scene3D model loaded.")

    video_filepath = args.input_video_filepath
    cap = cv2.VideoCapture(video_filepath)
    if not cap.isOpened():
        print("Error opening video stream or file.")
        return

    output_filepath_obj = args.output_video_filepath
    if not output_filepath_obj.lower().endswith(".avi"):
        output_filepath_obj = output_filepath_obj + ".avi"

    out_dir = os.path.dirname(output_filepath_obj)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer_obj = cv2.VideoWriter(
        output_filepath_obj,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        OUTPUT_VIDEO_SIZE,
    )

    alpha = 0.97
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_iter = (
        tqdm(range(frame_count), desc="Processing video frames: ", unit="frames", colour="green")
        if frame_count > 0
        else iter(int, 1)
    )

    print("Processing started.")
    for _ in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb).resize(FRAME_INF_SIZE)

        prediction = model.inference(image_pil)

        frame_resized = cv2.resize(frame, OUTPUT_VIDEO_SIZE)
        image_vis_obj = make_visualization(frame_resized, prediction, alpha=alpha)

        if args.vis:
            cv2.imshow("Depth", image_vis_obj)
            cv2.waitKey(10)

        writer_obj.write(image_vis_obj)

    cap.release()
    writer_obj.release()
    cv2.destroyAllWindows()
    print(f"Visualization video saved to: {output_filepath_obj}")


if __name__ == "__main__":
    main()
# %%
