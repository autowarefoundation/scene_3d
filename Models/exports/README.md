# Export and benchmark Scene3D model

## Export to FP32 ONNX (`convert_pytorch_to_onnx.py`)
Convert a trained Scene3D PyTorch `.pth` checkpoint to FP32 ONNX format:

```bash
python3 Models/exports/convert_pytorch_to_onnx.py \
    -p <PATH TO THE PYTORCH FILE> \
    -o <PATH TO THE OUTPUT ONNX FILE>
```

## Export to TorchScript (`trace_pytorch_model.py`)
Trace and save a trained Scene3D PyTorch `.pth` checkpoint as a TorchScript `.pt` model:

```bash
python3 Models/exports/trace_pytorch_model.py \
    -p <PATH TO THE PYTORCH FILE> \
    -o <PATH TO THE OUTPUT PT FILE>
```

## Benchmark PyTorch and ONNX models (`benchmark_onnx_models.py`)
Evaluate latency and validation depth losses (Scale Invariant MAE & Multi-Scale Edge Preservation Loss) on validation samples:

```bash
python3 Models/exports/benchmark_onnx_models.py \
    --pytorch_model_path /path/to/scene3d.pth \
    --fp32_model_path /path/to/scene3d.onnx \
    --dataset_root /path/to/scene3d_dataset_root
```
