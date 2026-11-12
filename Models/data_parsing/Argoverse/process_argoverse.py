#! /usr/bin/env python3
# %%
import json
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


def parseCalib(calib_files):
    calib_logs = []
    focal_lengths = []

    for i in range(0, len(calib_files)):
        calib_filepath = str(calib_files[i])
        calib_log = calib_filepath[56:-37]
        calib_logs.append(calib_log)

        with open(calib_filepath, "r") as file:
            data = json.load(file)

            for p in range(0, len(data["camera_data_"])):
                camera = data["camera_data_"][p]["key"]
                if camera == "image_raw_stereo_front_left_rect":
                    focal_length = data["camera_data_"][p]["value"]["focal_length_y_px_"]
                    focal_lengths.append(focal_length)

    return calib_logs, focal_lengths


def createDepthMap(depth_data, focal_length, baseline):
    assert np.max(depth_data) > 255
    depth_data = depth_data.astype("float32") / 256.0

    valid_pixels = depth_data > 0
    depth_map = np.float32((focal_length * baseline) / (depth_data + (1.0 - valid_pixels)))
    depth_map[depth_map > 200] = 0

    return depth_map


def cropData(image_left, depth_map, validity_mask):
    size = depth_map.shape
    width = size[1]

    image_left = image_left.crop((0, 518, width, 1750))
    depth_map = depth_map[518:1750, 0:width]
    validity_mask = validity_mask[518:1750, 0:width]

    return image_left, depth_map, validity_mask


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

    depth_filepath = os.path.join(root_data_path, "disparity_maps_v1.1")
    images_filepath = os.path.join(root_data_path, "rectified_stereo_images_v1.1", "train")

    depth_maps = sorted(
        [f for f in pathlib.Path(depth_filepath).glob("*/stereo_front_left_rect_disparity/*.png")]
    )
    images_left = sorted(
        [f for f in pathlib.Path(images_filepath).glob("*/stereo_front_left_rect/*.jpg")]
    )
    images_right = sorted(
        [f for f in pathlib.Path(images_filepath).glob("*/stereo_front_right_rect/*.jpg")]
    )
    calib_files = sorted([f for f in pathlib.Path(images_filepath).glob("*/*.json")])

    num_depth_maps = len(depth_maps)
    num_images_left = len(images_left)
    num_images_right = len(images_right)

    check_data_left = CheckData("Argoverse (left)", num_images_left, num_depth_maps)
    check_data_right = CheckData("Argoverse (right)", num_images_right, num_depth_maps)

    if check_data_left.getCheck() and check_data_right.getCheck():
        print("Beginning processing of data")
        os.makedirs(os.path.join(root_save_path, "image"), exist_ok=True)
        os.makedirs(os.path.join(root_save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(root_save_path, "validity"), exist_ok=True)

        calib_logs, focal_lengths = parseCalib(calib_files)
        baseline = 0.2986

        for index in range(0, num_depth_maps):
            print(f"Processing image {index} of {num_depth_maps - 1}")

            image_left = Image.open(str(images_left[index]))
            depth_data = np.array(Image.open(str(depth_maps[index])), dtype=int)

            data_log = str(str(images_left[index])[56:-69])
            log_index = calib_logs.index(data_log)
            focal_length = focal_lengths[log_index]

            sparse_depth_map = createDepthMap(depth_data, focal_length, baseline)
            lidar_depth_fill = LidarDepthFill(sparse_depth_map)
            depth_map_fill = lidar_depth_fill.getDepthMap()

            validity_mask = np.zeros_like(depth_map_fill)
            validity_mask[np.where(depth_map_fill != 0)] = 1

            image_left, depth_map_fill, validity_mask = cropData(
                image_left, depth_map_fill, validity_mask
            )

            image_save_path = os.path.join(root_save_path, "image", f"{index}.png")
            image_left.save(image_save_path, "PNG")

            depth_save_path = os.path.join(root_save_path, "depth", f"{index}.npy")
            np.save(depth_save_path, depth_map_fill)

            validity_save_path = os.path.join(root_save_path, "validity", f"{index}.png")
            validity_mask = Image.fromarray(np.uint8(validity_mask * 255))
            validity_mask.save(validity_save_path, "PNG")

        print("----- Processing complete -----")


if __name__ == "__main__":
    main()
# %%
