#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

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
        "--output_pt_trace_filepath",
        dest="output_pt_trace_filepath",
        required=True,
        help="path to *.pt output trace file generated",
    )
    args = parser.parse_args()

    model_checkpoint_path = args.model_checkpoint_path
    traced_model_save_path = args.output_pt_trace_filepath

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using {device} for tracing.")

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

    input_shape = (1, 3, 320, 640)
    input_data = torch.randn(input_shape).to(device)

    out_dir = os.path.dirname(traced_model_save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Tracing model")
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, input_data)
    traced_script_module.save(traced_model_save_path)
    print("Torch Trace Export file generated successfully.")


if __name__ == "__main__":
    main()
