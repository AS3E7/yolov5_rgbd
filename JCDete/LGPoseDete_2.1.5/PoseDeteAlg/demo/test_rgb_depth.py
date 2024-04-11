# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:04:31 2020

@author: Administrator
"""

import numpy as np
import os
import cv2

from detect_bbox.Bbox_Detecter import Bbox_Detecter

if __name__ =='__main__':
    print('Start.')
    
    TestCase = 1 # Test RGB image data
    
    if TestCase == 1: # Test RGB image data
        CurRGBImageFolderName = r'C:\software\yolov3-dome\demo\test_images\online_rgb_depth\rgb'
        CurRGBImageFileName = 'temp_img_trans_1.png'
        CurRGBImageFullName = os.path.join(CurRGBImageFolderName, CurRGBImageFileName)
        # read image
        ImageData = cv2.imread(CurRGBImageFullName)
        # detect
        pred_bbox, pred_labels, pred_scores = Bbox_Detecter(ImageData)
        print('cur image result = ', pred_bbox, pred_labels, pred_scores)
    
    
    
    print('End.')
    
    
    