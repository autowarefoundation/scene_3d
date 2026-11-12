# Scene3D Training

## train_scene_3d.py

Main script for training the Scene3D neural network on relative depth datasets (`Diverse/image/` and `Diverse/relative-depth/`).

### Example usage

**Train Scene3D with pre-trained upstream SceneSeg backbone weights:**

```bash
python3 Models/training/train_scene_3d.py \
    -s /path/to/save_models/ \
    -r /path/to/scene3d_dataset_root/ \
    -t /path/to/save_test_images/ \
    -m /path/to/sceneseg_checkpoint.pth
```

**Resume or fine-tune from a saved Scene3D checkpoint:**

```bash
python3 Models/training/train_scene_3d.py \
    -s /path/to/save_models/ \
    -r /path/to/scene3d_dataset_root/ \
    -t /path/to/save_test_images/ \
    -l \
    -c /path/to/scene3d_checkpoint.pth
```

### Main parameters:

- `-s`, `--model_save_root_path` : directory where PyTorch `.pth` checkpoints are saved (required)
- `-m`, `--pretrained_checkpoint_path` : path to upstream SceneSeg `.pth` weights for backbone initialization
- `-c`, `--checkpoint_path` : path to Scene3D `.pth` checkpoint when resuming/fine-tuning with `-l`
- `-r`, `--root` : dataset root directory containing `Diverse/image/`, `Diverse/relative-depth/`, and `Test/` (required)
- `-t`, `--test_images_save_root_path` : directory where periodic test-image depth visualizations are saved (required)
- `-l`, `--load_from_save` : flag to resume from a Scene3D checkpoint (`-c`) instead of upstream SceneSeg (`-m`)
- `--start_epoch` : starting epoch index (default: `2`)
- `--num_epochs` : total epoch upper bound (default: `5`)
- `--val_interval` : step interval for checkpoint saving and full validation (default: `92000`)

## scene_3d_trainer.py

Helper class (`Scene3DTrainer`) implementing model initialization, data loading, `DEPTH` augmentations, Scale Invariant robust MAE loss + Multi-Scale Edge Preservation loss, TensorBoard logging, validation, and visualization export.
