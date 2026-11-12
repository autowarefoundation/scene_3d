## lidar_depth_fill.py
In order to train the Scene3D metric depth network, we require ground truth depth maps. In real-world datasets, ground truth depth maps are acquired by LiDAR scanners, resulting in sparse virtual depth maps created by projection of 3D LiDAR points onto an image, where, in most cases, depth data is only available for ~5% of pixels. This script creates a filled-in LiDAR depth map based on the method [In Defense of Classical Image Processing: Fast Depth Completion on the CPU](https://arxiv.org/abs/1802.00036).

![Lidar Depth Fill](../../../Media/Lidar_Depth_Fill.jpg)

## Important Note
However, one modification was made to the above method, wherein extrapolation of data to the top of the image based on a fixed depth assumption was not applied, leaving the top section of the image empty where depth data was unavailable.
