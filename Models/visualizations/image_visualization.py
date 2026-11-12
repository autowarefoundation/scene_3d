# %%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cmapy
import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.inference.scene_3d_infer import Scene3DNetworkInfer, Scene3DONNXInfer

FRAME_INF_SIZE = (640, 320)


def load_model(checkpoint_path: str):
    if checkpoint_path and checkpoint_path.endswith(".onnx"):
        return Scene3DONNXInfer(checkpoint_path)
    return Scene3DNetworkInfer(checkpoint_path=checkpoint_path or "")


def make_visualization(
    frame_bgr: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.97,
) -> np.ndarray:
    """Colorize a predicted depth map with viridis and blend onto a BGR frame."""
    prediction_resized = cv2.resize(
        prediction,
        (frame_bgr.shape[1], frame_bgr.shape[0]),
    )
    min_val = np.min(prediction_resized)
    max_val = np.max(prediction_resized)
    denom = max_val - min_val
    if denom > 0:
        prediction_norm = 255.0 * ((prediction_resized - min_val) / denom)
    else:
        prediction_norm = np.zeros_like(prediction_resized)

    prediction_uint8 = prediction_norm.astype(np.uint8)
    prediction_color = cv2.applyColorMap(prediction_uint8, cmapy.cmap("viridis"))
    return cv2.addWeighted(prediction_color, alpha, frame_bgr, 1 - alpha, 0)


def process_single_image(
    model,
    input_image_filepath: str,
    output_dir: str = None,
    alpha: float = 0.97,
):
    print(f"Reading Image: {input_image_filepath}")
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read image: {input_image_filepath}")

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb).resize(FRAME_INF_SIZE)

    prediction = model.inference(image_pil)
    image_vis_obj = make_visualization(frame, prediction, alpha=alpha)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_id = Path(input_image_filepath).stem.zfill(3)
        output_image_filepath = os.path.join(output_dir, f"{img_id}_result.png")
        cv2.imwrite(output_image_filepath, image_vis_obj)
        print(f"Saved visualization to: {output_image_filepath}")
    else:
        cv2.imshow("Depth", image_vis_obj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        dest="model_checkpoint_path",
        required=False,
        help="Path to PyTorch (.pth/.pt) checkpoint or ONNX (.onnx) model file.",
    )
    parser.add_argument(
        "-i",
        "--input_image_dirpath",
        dest="input_image_dirpath",
        required=True,
        help="Path to input image file or directory of images to be processed by Scene3D.",
    )
    parser.add_argument(
        "-o",
        "--output_image_dirpath",
        dest="output_image_dirpath",
        required=False,
        help="Path to output image visualization directory where results will be saved.",
    )
    args = parser.parse_args()

    input_path = args.input_image_dirpath
    output_dir = args.output_image_dirpath
    model_checkpoint_path = args.model_checkpoint_path or ""

    model = load_model(model_checkpoint_path)
    print("Scene3D model loaded.")

    if os.path.isfile(input_path):
        process_single_image(model, input_path, output_dir)
    elif os.path.isdir(input_path):
        if not output_dir:
            raise ValueError("--output_image_dirpath (-o) is required when input is a directory")
        os.makedirs(output_dir, exist_ok=True)
        for filename in sorted(os.listdir(input_path)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                process_single_image(
                    model,
                    os.path.join(input_path, filename),
                    output_dir,
                )
            else:
                print(f"Skipping non-image file: {filename}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
# %%
