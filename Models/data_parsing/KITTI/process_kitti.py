#! /usr/bin/env python3
# %%
import os
import pathlib
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_parsing.common.lidar_depth_fill import LidarDepthFill
from Models.data_utils.check_data import CheckData


def removeExtraSamples(image_folders):
    filtered_images_left = []
    filtered_images_right = []

    for i in range(0, len(image_folders)):
        images_left = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_02/*data/*.png")])
        images_right = sorted([f for f in pathlib.Path(str(image_folders[i])).glob("*image_03/*data/*.png")])
        for j in range(5, len(images_left) - 5):
            filtered_images_left.append(images_left[j])
            filtered_images_right.append(images_right[j])

    return filtered_images_left, filtered_images_right


def createDepthMap(depth_data):
    assert np.max(depth_data) > 255
    depth_map = depth_data.astype("float32") / 256.0
    return depth_map


def cropData(image_left, depth_map_fill_only, validity_mask):
    size = depth_map_fill_only.shape
    height = size[0]
    width = size[1]

    image_left = image_left.crop((256, 0, width - 100, height))
    depth_map_fill_only = depth_map_fill_only[:, 256 : width - 100]
    validity_mask = validity_mask[:, 256 : width - 100]

    return image_left, depth_map_fill_only, validity_mask


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        dest="root_data_path",
        required=True,
        help="path to root folder with input ground truth labels and images",
    )
    parser.add_argument(
        "-s",
        "--save",
        dest="root_save_path",
        required=True,
        help="path to folder where processed data will be saved",
    )
    args = parser.parse_args()

    root_data_path = args.root_data_path
    root_save_path = args.root_save_path

    depth_filepath = os.path.join(root_data_path, "train")
    images_filepath = os.path.join(root_data_path, "data")

    depth_maps = sorted(
        [f for f in pathlib.Path(depth_filepath).glob("*/proj_depth/*groundtruth/*image_02/*.png")]
    )
    image_folders = sorted([f for f in pathlib.Path(images_filepath).glob("*")])

    images_left, images_right = removeExtraSamples(image_folders)

    num_depth_maps = len(depth_maps)
    num_images_left = len(images_left)
    num_images_right = len(images_right)

    check_data_left = CheckData("KITTI (left)", num_images_left, num_depth_maps)
    check_data_right = CheckData("KITTI (right)", num_images_right, num_depth_maps)

    if check_data_left.getCheck() and check_data_right.getCheck():
        print("Beginning processing of data")
        os.makedirs(os.path.join(root_save_path, "image-full"), exist_ok=True)
        os.makedirs(os.path.join(root_save_path, "depth-full"), exist_ok=True)
        os.makedirs(os.path.join(root_save_path, "validity-full"), exist_ok=True)

        for index in range(15815, num_depth_maps):
            print(f"Processing image {index} of {num_depth_maps - 1}")

            image_left = Image.open(str(images_left[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            sparse_depth_map = createDepthMap(depth_data)
            lidarDepthFill = LidarDepthFill(sparse_depth_map)
            depth_map = lidarDepthFill.getDepthMap()

            validity_mask = np.zeros_like(depth_map)
            validity_mask[np.where(depth_map != 0)] = 1

            image_left, depth_map, validity_mask = cropData(image_left, depth_map, validity_mask)

            image_save_path = os.path.join(root_save_path, "image-full", f"{index}.jpg")
            image_left.save(image_save_path)

            depth_save_path = os.path.join(root_save_path, "depth-full", f"{index}.npy")
            np.save(depth_save_path, depth_map)

            validity_save_path = os.path.join(root_save_path, "validity-full", f"{index}.png")
            validity_mask = Image.fromarray(np.uint8(validity_mask * 255))
            validity_mask.save(validity_save_path, "PNG")

        print("----- Processing complete -----")


if __name__ == "__main__":
    main()
# %%
