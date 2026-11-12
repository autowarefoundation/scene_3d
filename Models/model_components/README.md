# model_components

Contains PyTorch classes for all components of the `Scene3DNetwork` monocular relative depth estimation model:

- `scene_3d/backbone.py` — EfficientNet-B0 encoder extracting 5 multi-scale feature maps (`[l0, l2, l3, l4, l8]`).
- `scene_3d/scene_seg_network.py` — Upstream `SceneSegNetwork` wrapper providing the pretrained road-scene encoder backbone.
- `scene_3d/pre_trained_backbone.py` — Frozen wrapper (`PreTrainedBackbone`) around the upstream `SceneSegNetwork` backbone.
- `scene_3d/depth_context.py` — Global pooling, MLP context bottleneck, spatial reshaping, and residual attention (`context * features + features`).
- `scene_3d/scene_3d_neck.py` — Decoder neck with progressive transposed-convolution upsampling and encoder skip connections (`features[3]`, `features[2]`, `features[1]`).
- `scene_3d/scene_3d_head.py` — Depth prediction head with final upsampling, `features[0]` skip connection, and 3 convolution layers producing a 1-channel dense relative depth map.
- `scene_3d/scene_3d_network.py` — Full end-to-end `Scene3DNetwork` module.
- `scene_3d/scene_3d_upstream.py` — Frozen upstream feature and context extractor wrapper (`Scene3DUpstream`).
