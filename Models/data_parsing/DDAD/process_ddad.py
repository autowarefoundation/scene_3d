# %%
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from dgp.datasets import SynchronizedSceneDataset
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_parsing.common.lidar_depth_fill import LidarDepthFill


def cropData(image, depth_map, validity_mask):
    # Cropping out those parts of data for which depth is unavailable
    image = image.crop((268, 200, 1668, 900))
    depth_map = depth_map[200:900, 268:1668]
    validity_mask = validity_mask[200:900, 268:1668]

    return image, depth_map, validity_mask


def processSample(sample):
    camera = sample[0][0]
    image = camera["rgb"]
    sparse_depth_map = camera["depth"]

    lidarDepthFill = LidarDepthFill(sparse_depth_map)
    depth_map = lidarDepthFill.getDepthMap()

    validity_mask = np.zeros_like(depth_map)
    validity_mask[np.where(depth_map != 0)] = 1

    image, depth_map, validity_mask = cropData(image, depth_map, validity_mask)
    return image, depth_map, validity_mask


def saveData(root_save_path, counter, image, depth_map, validity_mask):
    image_save_path = os.path.join(root_save_path, "image", f"{counter}.png")
    image.save(image_save_path, "PNG")

    depth_save_path = os.path.join(root_save_path, "depth", f"{counter}.npy")
    np.save(depth_save_path, depth_map)

    validity_save_path = os.path.join(root_save_path, "validity", f"{counter}.png")
    validity_mask = Image.fromarray(np.uint8(validity_mask * 255))
    validity_mask.save(validity_save_path, "PNG")


def main():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="ddad_json_path", required=True, help="path to ddad.json file")
    parser.add_argument(
        "-s",
        "--save",
        dest="root_save_path",
        required=True,
        help="path to folder where processed data will be saved",
    )
    args = parser.parse_args()

    ddad_json_path = args.ddad_json_path
    root_save_path = args.root_save_path

    os.makedirs(os.path.join(root_save_path, "image"), exist_ok=True)
    os.makedirs(os.path.join(root_save_path, "depth"), exist_ok=True)
    os.makedirs(os.path.join(root_save_path, "validity"), exist_ok=True)

    dataset_train_front = SynchronizedSceneDataset(
        ddad_json_path,
        datum_names=("lidar", "CAMERA_01"),
        generate_depth_from_datum="lidar",
        split="train",
    )
    dataset_val_front = SynchronizedSceneDataset(
        ddad_json_path,
        datum_names=("lidar", "CAMERA_01"),
        generate_depth_from_datum="lidar",
        split="val",
    )
    dataset_train_rear = SynchronizedSceneDataset(
        ddad_json_path,
        datum_names=("lidar", "CAMERA_09"),
        generate_depth_from_datum="lidar",
        split="train",
    )
    dataset_val_rear = SynchronizedSceneDataset(
        ddad_json_path,
        datum_names=("lidar", "CAMERA_09"),
        generate_depth_from_datum="lidar",
        split="val",
    )

    num_train_front_samples = len(dataset_train_front)
    num_val_front_samples = len(dataset_val_front)
    num_train_rear_samples = len(dataset_train_rear)
    num_val_rear_samples = len(dataset_val_rear)
    total_data = (
        num_train_front_samples
        + num_val_front_samples
        + num_train_rear_samples
        + num_val_rear_samples
    )

    print("Total samples", total_data)
    counter = 0

    for index in range(0, num_train_rear_samples, 2):
        sample = dataset_train_rear[index]
        image, depth_map, validity_mask = processSample(sample)
        saveData(root_save_path, counter, image, depth_map, validity_mask)
        print("Processing image ", counter, " of ", round(total_data / 2) - 1)
        counter += 1

    for index in range(0, num_train_front_samples, 2):
        sample = dataset_train_front[index]
        image, depth_map, validity_mask = processSample(sample)
        saveData(root_save_path, counter, image, depth_map, validity_mask)
        print("Processing image ", counter, " of ", round(total_data / 2) - 1)
        counter += 1

    for index in range(0, num_val_rear_samples, 2):
        sample = dataset_val_rear[index]
        image, depth_map, validity_mask = processSample(sample)
        saveData(root_save_path, counter, image, depth_map, validity_mask)
        print("Processing image ", counter, " of ", round(total_data / 2) - 1)
        counter += 1

    for index in range(0, num_val_front_samples, 2):
        sample = dataset_val_front[index]
        image, depth_map, validity_mask = processSample(sample)
        saveData(root_save_path, counter, image, depth_map, validity_mask)
        print("Processing image ", counter, " of ", round(total_data / 2) - 1)
        counter += 1

    print("--- Processing Complete ----")


if __name__ == "__main__":
    main()
# %%
