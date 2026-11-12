# %%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import inspect
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import onnx
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.model_components.scene_3d.scene_3d_network import Scene3DNetwork
from Models.model_components.scene_3d.scene_seg_network import SceneSegNetwork


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        dest="model_checkpoint_path",
        required=True,
        help="path to pytorch checkpoint file to load model dict",
    )
    parser.add_argument(
        "-o",
        "--onnx_model_path",
        dest="onnx_model_path",
        required=True,
        help="path to converted ONNX model, must include output file name with .onnx extension",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )
    args = parser.parse_args()

    model_checkpoint_path = args.model_checkpoint_path
    onnx_model_path = args.onnx_model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")

    scene_seg_network = SceneSegNetwork(pretrained=False)
    model = Scene3DNetwork(scene_seg_network)

    print("Loading Network")
    if len(model_checkpoint_path) > 0:
        checkpoint = torch.load(
            model_checkpoint_path,
            weights_only=False,
            map_location=device,
        )
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            if hasattr(checkpoint["model"], "state_dict"):
                model.load_state_dict(checkpoint["model"].state_dict())
            else:
                model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise ValueError("No path to checkpoint file provided in class initialization")

    model = model.to(device).eval()

    # Scene3D input shape: (batch, channels, height, width) = (1, 3, 320, 640)
    input_shape = (1, 3, 320, 640)
    input_data = torch.randn(input_shape).to(device)

    print("Testing inference")
    with torch.no_grad():
        _ = model(input_data)

    out_dir = os.path.dirname(onnx_model_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Converting model to ONNX at FP32 and exporting")
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["external_data"] = False

    torch.onnx.export(model, input_data, onnx_model_path, **export_kwargs)

    onnx_network = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_network)
    print("Checks passed - export complete")


if __name__ == "__main__":
    main()
# %%
