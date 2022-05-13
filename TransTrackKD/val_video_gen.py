import argparse
import numpy as np
import torch
import torchvision.transforms.functional as F
import glob as gb
import os
import cv2

#Giro made
print("Starting img2video")
demo_images_path = "./visual_val_predict"
img_paths = gb.glob(os.path.join(demo_images_path, "*.png"))
#size = (video_width, video_height) 
demo_output="."

#framerate varies between videos but will use 30
frame_rate=30
size=((1920,1080))
videowriter = cv2.VideoWriter(os.path.join(demo_output, "val_video.avi"), cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, size)

for img_path in sorted(img_paths):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    videowriter.write(img)

videowriter.release()
print("img2video is done")  