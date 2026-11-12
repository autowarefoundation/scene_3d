# %%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import pathlib
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_scene_3d import LoadDataScene3D
from Models.training.scene_3d_trainer import Scene3DTrainer


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--model_save_root_path",
        dest="model_save_root_path",
        required=True,
        help="root path where pytorch checkpoint file should be saved",
    )
    parser.add_argument(
        "-m",
        "--pretrained_checkpoint_path",
        dest="pretrained_checkpoint_path",
        default="",
        help="path to SceneSeg weights file for pre-trained backbone",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        dest="checkpoint_path",
        default="",
        help="path to Scene3D weights file for training from saved checkpoint",
    )
    parser.add_argument(
        "-r",
        "--root",
        dest="root",
        required=True,
        help="root path to folder where training data is stored",
    )
    parser.add_argument(
        "-t",
        "--test_images_save_root_path",
        dest="test_images_save_root_path",
        required=True,
        help="root path where test images are saved",
    )
    parser.add_argument(
        "-l",
        "--load_from_save",
        action="store_true",
        help="flag for whether model is being loaded from a Scene3D checkpoint file",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=2,
        help="starting epoch index (default: 2, matching Stage-2 fine-tuning)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="total epochs upper bound (default: 5)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=92000,
        help="step interval for checkpoint saving and full validation (default: 92000)",
    )
    args = parser.parse_args()

    root = args.root
    model_save_root_path = args.model_save_root_path
    test_images_save_root_path = args.test_images_save_root_path

    os.makedirs(model_save_root_path, exist_ok=True)
    os.makedirs(test_images_save_root_path, exist_ok=True)

    test_images_filepath = os.path.join(root, "Test")
    test_images = (
        sorted([f for f in pathlib.Path(test_images_filepath).glob("*")])
        if os.path.isdir(test_images_filepath)
        else []
    )
    num_test_images = len(test_images)

    load_from_checkpoint = bool(args.load_from_save)

    diverse_labels_filepath = os.path.join(root, "Diverse", "relative-depth")
    diverse_images_filepath = os.path.join(root, "Diverse", "image")

    Dataset = LoadDataScene3D(diverse_labels_filepath, diverse_images_filepath)
    total_train_samples, total_val_samples = Dataset.getItemCount()

    print(total_train_samples, ": Total training samples")
    print(total_val_samples, ": Total validation samples")
    print(num_test_images, ": Total samples for visual testing")

    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    checkpoint_path = args.checkpoint_path

    if not load_from_checkpoint:
        trainer = Scene3DTrainer(
            pretrained_checkpoint_path=pretrained_checkpoint_path
        )
    else:
        trainer = Scene3DTrainer(
            checkpoint_path=checkpoint_path, is_pretrained=True
        )

    trainer.zero_grad()

    num_epochs = args.num_epochs
    batch_size = 24

    for epoch in range(args.start_epoch, num_epochs):
        print("Epoch: ", epoch + 1)
        randomlist_train_data = random.sample(
            range(0, total_train_samples), total_train_samples
        )

        # Learning Rate schedule
        if epoch >= 2:
            trainer.set_learning_rate(0.0000125)
            batch_size = 3
        else:
            trainer.set_learning_rate(0.0001)
            batch_size = 24

        for count in range(0, total_train_samples):
            log_count = count + total_train_samples * epoch

            image, gt = Dataset.getItemTrain(randomlist_train_data[count])
            trainer.set_data(image, gt)
            trainer.apply_augmentations(is_train=True)
            trainer.load_data()
            trainer.run_model()
            trainer.loss_backward()

            if (count + 1) % batch_size == 0:
                trainer.run_optimizer()

            if (count + 1) % 250 == 0:
                trainer.log_loss(log_count)

            if (count + 1) % 1000 == 0:
                trainer.save_visualization(log_count)

            if (log_count + 1) % args.val_interval == 0:
                print("Running Validation - Step:", str(log_count))
                model_save_path = os.path.join(
                    model_save_root_path,
                    f"iter_{log_count}_epoch_{epoch}_step_{count}.pth",
                )
                trainer.save_model(model_save_path)
                trainer.set_eval_mode()

                mAE = 0
                mEL = 0

                with torch.no_grad():
                    for test_count in range(0, num_test_images):
                        test_image = test_images[test_count]
                        test_save_path = os.path.join(
                            test_images_save_root_path,
                            f"iter_{log_count}_epoch_{epoch}_step_{count}_img_{test_count}.png",
                        )
                        trainer.test(test_image, test_save_path)

                    for val_count in range(0, total_val_samples):
                        image_val, gt_val = Dataset.getItemVal(val_count)
                        mAE_item, mEL_item = trainer.validate(image_val, gt_val)
                        mAE += mAE_item
                        mEL += mEL_item

                    avg_mEL = mEL / total_val_samples
                    avg_mAE = mAE / total_val_samples
                    avg_overall = avg_mEL + avg_mAE
                    trainer.log_val_loss(avg_overall, avg_mAE, avg_mEL, log_count)

                trainer.set_train_mode()

    trainer.cleanup()


if __name__ == "__main__":
    main()
# %%
