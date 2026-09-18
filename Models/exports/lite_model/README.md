# Lite Models Evaluation

This section describes how to evaluate Lite models using either:

- PyTorch checkpoints (`.pth`)
- Exported ONNX models (`.onnx`)

You can freely adjust `--height` and `--width` depending on:

- Dataset resolution  
- Model configuration  
- Desired evaluation scale  

All results are saved by default in:

```
runs/eval/
```

You can override this by specifying a custom output directory:

```
--out_dir <path>
```
---

# Scene3DLite Evaluation

**Used for depth / 3D evaluation on supported datasets.**

### Evaluate using PyTorch checkpoint

```bash
python Models/exports/lite_models/eval_scene3d_lite_model.py \
    --checkpoint /path/to/scene3dlite.pth \
    --datasets TUSimple CurveLanes \
    --height 320 --width 640
```

### Evaluate using ONNX model

```bash
python Models/exports/lite_models/eval_scene3d_lite_model.py \
    --onnx /path/to/scene3dlite.onnx \
    --datasets TUSimple CurveLanes \
    --height 320 --width 640
```

---

# Available Datasets

Supported datasets include:

- `acdc`
- `mapillary`
- `muses`
- `idda`
- `bdd100k`
- `cityscapes`
- `curvelanes` (lane detection)
- `tusimple` (lane detection)

> Height and width should be selected according to the dataset resolution and the model configuration used during training.
