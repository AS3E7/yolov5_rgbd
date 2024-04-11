# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:52:08 2020

@author: huqiugen
"""
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from .utils.torch_utils import select_device, load_classifier, time_synchronized
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='demo_jc/weights/best.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
opt = parser.parse_args()

weights = opt.weights
imgsz = opt.img_size
# Initialize
device = select_device(opt.device)
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
# Eval模式
model.to(device).eval()
def detect(img0):

    # Set Dataloader
    dataset = LoadImages(img0, img_size=imgsz, half=False)

    img = torch.from_numpy(dataset).to(device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            return det
if __name__ == '__main__':

    with torch.no_grad():
        path = './data/samples'
        path = str(Path(path))
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        for i in files:
            print('i:',i)
            img0 = cv2.imread(i)
            t1 = time.time()
            y = detect(img0)
            t2 = time.time()
            print(y)
            print('t3:',t2-t1)
