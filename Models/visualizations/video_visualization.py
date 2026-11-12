#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import os
import cv2
import sys
import numpy as np
from PIL import Image
import cmapy
from argparse import ArgumentParser
from Models.inference.scene_3d_infer import Scene3DNetworkInfer


def main(): 

    parser = ArgumentParser()
    parser.add_argument(
        "-p", 
        "--model_checkpoint_path", 
        dest = "model_checkpoint_path", 
        help = "Path to Pytorch checkpoint file to load model dict."
    )
    parser.add_argument(
        "-i", 
        "--input_video_filepath", 
        dest = "input_video_filepath", 
        help = "Path to input video which will be processed by Scene3D."
    )
    parser.add_argument(
        "-o", 
        "--output_video_filepath", 
        dest = "output_video_filepath", 
        help = "Path to output video visualization file, must include output file name."
    )
    parser.add_argument(
        "-v", 
        "--vis", 
        action = "store_true", 
        help = "Flag for whether to show frame by frame visualization while processing is occuring."
    )

    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = Scene3DNetworkInfer(checkpoint_path = model_checkpoint_path)
    print("Scene3D model loaded.")
    
    # Create a VideoCapture object and read from input file
    # If the input is taken from the camera, pass 0 instead of the video file name.
    video_filepath = args.input_video_filepath
    cap = cv2.VideoCapture(video_filepath)

    # Output filepath
    output_filepath_obj = args.output_video_filepath
    if (not output_filepath_obj.lower().endswith(".avi")):
        output_filepath_obj = output_filepath_obj + ".avi"

    # Video writer object
    writer_obj = cv2.VideoWriter(
        output_filepath_obj,
        cv2.VideoWriter_fourcc(*"MJPG"), 
        25,
        (1280,720)
    )

    # Check if video catpure opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video stream or file.")
    else:
        print("Reading video frames...")

    # Transparency factor
    alpha = 0.97
 
    # Read until video is completed
    print("Processing started.")
    while(cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # Display the resulting frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize((640, 320))
            
            # Run inference
            prediction = model.inference(image_pil)
            prediction = cv2.resize(
                prediction, 
                (frame.shape[1], frame.shape[0])
            )
    
            # Create visualization
            prediction_image = 255.0 * (
                (prediction - np.min(prediction)) / 
                (np.max(prediction) - np.min(prediction))
            )
            prediction_image = prediction_image.astype(np.uint8)
            vis_obj = cv2.applyColorMap(prediction_image, cmapy.cmap("viridis"))

            # Resizing to match the size of the output video
            # which is set to standard HD resolution
            frame = cv2.resize(
                frame, 
                (1280, 720)
            )
            vis_obj = cv2.resize(
                vis_obj, 
                (1280, 720)
            )

            # Create the composite visualization
            image_vis_obj = cv2.addWeighted(
                vis_obj, alpha, 
                frame, 1 - alpha, 
                0
            )

            if (args.vis):
                cv2.imshow("Depth", image_vis_obj)
                cv2.waitKey(10)

            # Writing to video frame
            writer_obj.write(image_vis_obj)
        else:
            print("Frame not read - ending processing...")
            break

    # When everything done, release the video capture and writer objects
    cap.release()
    writer_obj.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print("Completed.")
    
if __name__ == "__main__":
    main()
    # %%