#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import cv2
import sys
import numpy as np
from argparse import ArgumentParser
import cmapy
from PIL import Image
from Models.inference.scene_3d_infer import Scene3DNetworkInfer


def main(): 

    parser = ArgumentParser()
    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path", 
        help = "Path to pytorch checkpoint file to load model dict."
    )
    parser.add_argument(
        "-i", 
        "--input_image_dirpath", 
        dest = "input_image_dirpath", 
        help = "Path to input image which will be processed by SceneSeg."
    )
    parser.add_argument(
        "-o", 
        "--output_image_dirpath", 
        dest = "output_image_dirpath", 
        help = "Path to output image visualization directory, containing all results."
    )

    args = parser.parse_args()
    # Arranging I/O dirs
    input_image_dirpath = args.input_image_dirpath
    output_image_dirpath = args.output_image_dirpath
    if (not os.path.exists(output_image_dirpath)):
        os.makedirs(output_image_dirpath)

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = Scene3DNetworkInfer(checkpoint_path = model_checkpoint_path)

    # Transparency factor
    alpha = 0.97
  
    # # Reading input image
    # input_image_filepath = args.input_image_filepath
    # frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image_pil = Image.fromarray(image)
    # image_pil = image_pil.resize((640, 320))

    # # Run inference
    # prediction = model.inference(image_pil)
    # prediction = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))

    # # Transparency factor
    # alpha = 0.97

    # # Create visualization
    # prediction_image = 255.0*((prediction - np.min(prediction))/ (np.max(prediction) - np.min(prediction)))
    # prediction_image = prediction_image.astype(np.uint8)
    # prediction_image = cv2.applyColorMap(prediction_image, cmapy.cmap('viridis'))
    # image_vis_obj = cv2.addWeighted(prediction_image, alpha, frame, 1 - alpha, 0)

    # # Display depth map
    # window_name = 'depth'
    # cv2.imshow(window_name, image_vis_obj)
    # cv2.waitKey(0)

    # Process through input image dir
    for filename in sorted(os.listdir(input_image_dirpath)):
        if (filename.endswith((".png", ".jpg", ".jpeg"))):

            # Fetch image
            input_image_filepath = os.path.join(
                input_image_dirpath, filename
            )
            img_id = filename.split(".")[0].zfill(3)

            print(f"Reading Image: {input_image_filepath}")
            frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((640, 320))

            # Inference
            prediction = model.inference(image_pil)
            prediction = cv2.resize(
                prediction, 
                (frame.shape[1], frame.shape[0])
            )

            # Visualization
            prediction_image = 255.0 * (
                (prediction - np.min(prediction)) / 
                (np.max(prediction) - np.min(prediction))
            )
            prediction_image = prediction_image.astype(np.uint8)
            prediction_image = cv2.applyColorMap(
                prediction_image, 
                cmapy.cmap("viridis")
            )
            image_vis_obj = cv2.addWeighted(
                prediction_image, alpha, 
                frame, 1 - alpha, 
                0
            )
            
            output_image_filepath = os.path.join(
                output_image_dirpath,
                f"{img_id}_result.png"
            )
            cv2.imwrite(output_image_filepath, image_vis_obj)

        else:
            print(f"Skipping non-image file: {filename}")
            continue

if __name__ == "__main__":
    main()
# %%