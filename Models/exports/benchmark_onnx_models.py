#!/usr/bin/env python3
"""Benchmark PyTorch and ONNX Scene3D models on relative depth validation datasets."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_scene_3d import LoadDataScene3D
from Models.inference.scene_3d_infer import Scene3DNetworkInfer, Scene3DONNXInfer

GX_FILTER = torch.tensor(
    [[0.125, 0, -0.125], [0.25, 0, -0.25], [0.125, 0, -0.125]],
    dtype=torch.float32,
).view(1, 1, 3, 3)

GY_FILTER = torch.tensor(
    [[0.125, 0.25, 0.125], [0, 0, 0], [-0.125, -0.25, -0.125]],
    dtype=torch.float32,
).view(1, 1, 3, 3)


def get_ssi_nom_tensor(tensor: torch.Tensor) -> torch.Tensor:
    denom = (torch.max(tensor) - torch.mean(tensor)).clamp_min(1e-6)
    return (tensor - torch.min(tensor)) / denom


def calc_mae_ssi_loss_robust(pred_ssi: torch.Tensor, gt_ssi: torch.Tensor) -> float:
    mae = torch.abs(pred_ssi - gt_ssi)
    mae_robust_val = torch.quantile(mae, 0.9, interpolation="linear")
    mae_robust = mae[mae < mae_robust_val]
    if mae_robust.numel() == 0:
        return float(torch.mean(mae).item())
    return float(torch.mean(mae_robust).item())


def calc_edge_ssi_loss(pred_ssi: torch.Tensor, gt_ssi: torch.Tensor) -> torch.Tensor:
    gx_pred = nn.functional.conv2d(pred_ssi, GX_FILTER, padding=1)
    gy_pred = nn.functional.conv2d(pred_ssi, GY_FILTER, padding=1)
    gx_gt = nn.functional.conv2d(gt_ssi, GX_FILTER, padding=1)
    gy_gt = nn.functional.conv2d(gt_ssi, GY_FILTER, padding=1)
    return torch.mean(torch.abs(gx_pred - gx_gt) + torch.abs(gy_pred - gy_gt))


def calc_multi_scale_ssi_edge_loss(pred_ssi: torch.Tensor, gt_ssi: torch.Tensor) -> float:
    downsample = nn.AvgPool2d(2, stride=2)
    p1 = downsample(pred_ssi)
    p2 = downsample(p1)
    p3 = downsample(p2)
    g1 = downsample(gt_ssi)
    g2 = downsample(g1)
    g3 = downsample(g2)
    total = (
        calc_edge_ssi_loss(pred_ssi, gt_ssi)
        + calc_edge_ssi_loss(p1, g1)
        + calc_edge_ssi_loss(p2, g2)
        + calc_edge_ssi_loss(p3, g3)
    ) / 4.0
    return float(total.item())


def load_validation_samples(dataset_root: str, num_samples: int):
    images_dir = os.path.join(dataset_root, "Diverse", "image")
    labels_dir = os.path.join(dataset_root, "Diverse", "relative-depth")
    loader = LoadDataScene3D(labels_dir, images_dir)
    _, n_vals = loader.getItemCount()
    count = min(n_vals, num_samples)
    samples = []
    for idx in range(count):
        img_np, gt_hw1 = loader.getItemVal(idx)
        gt_resized = cv2.resize(gt_hw1.squeeze(-1), (640, 320))
        samples.append(
            {
                "image": Image.fromarray(img_np).convert("RGB"),
                "gt": gt_resized.astype(np.float32),
            }
        )
    return samples


def benchmark_model(name: str, model, model_path: str, samples: list, warmup_runs: int):
    if not samples:
        raise ValueError("No validation samples loaded for benchmarking.")

    for i in range(warmup_runs):
        _ = model.inference(samples[i % len(samples)]["image"])

    latencies_ms = []
    mae_scores = []
    edge_scores = []

    for sample in samples:
        t0 = time.perf_counter()
        pred_hw1 = model.inference(sample["image"])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        pred_t = torch.from_numpy(pred_hw1.squeeze(-1)).float().unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(sample["gt"]).float().unsqueeze(0).unsqueeze(0)

        pred_ssi = get_ssi_nom_tensor(pred_t)
        gt_ssi = get_ssi_nom_tensor(gt_t)

        mae_scores.append(calc_mae_ssi_loss_robust(pred_ssi, gt_ssi))
        edge_scores.append(calc_multi_scale_ssi_edge_loss(pred_ssi, gt_ssi))

    mean_mae = float(np.mean(mae_scores))
    mean_edge = float(np.mean(edge_scores))
    total_loss = mean_mae + 4.0 * mean_edge
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n=== {name} ===")
    print(f"Model size               : {size_mb:.2f} MB")
    print(
        f"Mean latency             : {np.mean(latencies_ms):.2f} ms ({1000.0 / np.mean(latencies_ms):.1f} FPS)"
    )
    print(f"Scale Invariant MAE      : {mean_mae:.5f}")
    print(f"Edge Preservation Loss   : {mean_edge:.5f}")
    print(f"Total Loss (MAE + 4*Edge): {total_loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX Scene3D models")
    parser.add_argument("--pytorch_model_path", type=str, default=None, help="Path to .pth or .pt model")
    parser.add_argument("--fp32_model_path", type=str, default=None, help="Path to FP32 .onnx model")
    parser.add_argument("--int8_model_path", type=str, default=None, help="Path to INT8 .onnx model")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing Diverse/image and Diverse/relative-depth",
    )
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--warmup_runs", type=int, default=3)
    args = parser.parse_args()

    samples = load_validation_samples(args.dataset_root, args.num_samples)
    print(f"Loaded {len(samples)} validation samples from {args.dataset_root}")

    if args.pytorch_model_path:
        pt_model = Scene3DNetworkInfer(args.pytorch_model_path)
        benchmark_model("PyTorch FP32", pt_model, args.pytorch_model_path, samples, args.warmup_runs)

    if args.fp32_model_path:
        onnx_fp32 = Scene3DONNXInfer(args.fp32_model_path)
        benchmark_model("ONNX FP32", onnx_fp32, args.fp32_model_path, samples, args.warmup_runs)

    if args.int8_model_path:
        onnx_int8 = Scene3DONNXInfer(args.int8_model_path)
        benchmark_model("ONNX INT8", onnx_int8, args.int8_model_path, samples, args.warmup_runs)


if __name__ == "__main__":
    main()
