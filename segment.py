# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import torch
import numpy as np
import cv2
import random
import time
import os
import main as RCNN  # Assure-toi que ce module est correctement installé et importable
import sys

if __name__ == "__main__":
    data = f"{os.getcwd()}/{sys.argv[1]}"
    model = torch.load(f"{os.getcwd()}/model/mask_rcnn_model.pth")
    RCNN.instance_segmentation_api(data, 0.5,text_size=1.5, text_th=1, rect_th=1)