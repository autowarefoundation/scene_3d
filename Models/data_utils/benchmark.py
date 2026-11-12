# %%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from pytorch_model_summary import summary
from thop import profile

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.model_components.scene_3d.scene_3d_network import Scene3DNetwork
from Models.model_components.scene_3d.scene_seg_network import SceneSegNetwork


def benchmark(model, input_data, dtype="fp32", nwarmup=50, nruns=1000):
    if dtype == "fp16":
        input_data = input_data.half()
        model = model.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            _ = model(input_data)

    if input_data.device.type == "cuda":
        torch.cuda.synchronize()

    print("Start timing ...")
    timings = []

    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            output = model(input_data)
            if input_data.device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()

            timings.append(end_time - start_time)

            if i % 100 == 0:
                print(
                    "Iteration %d/%d, avg batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        dest="network_name",
        default="Scene3D",
        help="Specify the name of the network to benchmark (default: Scene3D)",
    )
    parser.add_argument(
        "-p",
        "--precision",
        dest="precision",
        default="fp32",
        help="Set precision of the network: fp32 or fp16",
    )
    args = parser.parse_args()

    model_name = args.network_name
    precision = args.precision

    if precision not in ("fp16", "fp32"):
        raise ValueError("Please specify precision as fp32 or fp16")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for inference")

    if model_name == "Scene3D":
        scene_seg_network = SceneSegNetwork(pretrained=False)
        model = Scene3DNetwork(scene_seg_network)
    elif model_name == "SceneSeg":
        model = SceneSegNetwork(pretrained=False)
    else:
        raise ValueError(f"Unsupported model name '{model_name}'. Use Scene3D or SceneSeg.")

    print(summary(model, torch.zeros((1, 3, 320, 640)), show_input=True))
    model = model.to(device).eval()

    input_shape = (1, 3, 320, 640)
    input_data = torch.randn(input_shape).to(device)

    total_ops, _ = profile(model, (input_data,), verbose=False)
    print(model_name, "FLOPs(G): ", "%.2f" % (total_ops / (1000 ** 3)))

    benchmark(model, input_data, precision, 50, 1000)


if __name__ == "__main__":
    main()
# %%
