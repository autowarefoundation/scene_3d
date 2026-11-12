## scene_3d_infer.py

Helper classes for inference using the Scene3D neural network:

- `Scene3DNetworkInfer`: PyTorch `.pth` checkpoint or traced `.pt` TorchScript inference wrapper (`Scene3DNetwork`).
- `Scene3DONNXInfer`: ONNX Runtime `.onnx` inference wrapper.

### Example usage

```python
from PIL import Image
from Models.inference.scene_3d_infer import Scene3DNetworkInfer

model = Scene3DNetworkInfer("/path/to/scene3d.pth")
img = Image.open("Media/highway_normal_1.png").convert("RGB")
depth_map = model.inference(img)  # shape: (320, 640, 1)
```
