# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:13:48 2020

@author: HYD
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from chainercv.visualizations import vis_bbox

from util import SaveResultInfo
from detect_bbox.YOLO_Detect.YOLO_Detecter import LoadYOLODeteParam

def PlotDeteResult(Image, Bbox, Labels, Scores, SaveFileName = ''):
    """
    功能：显示检测的结果
    """
#    print(Image.shape)
    PlotResultMethodFlag = 2 # PlotResultMethodFlag ==1，使用 chainercv.visualizations
                             # PlotResultMethodFlag ==2，使用 SaveResultInfo.draw_bbox
    
    if PlotResultMethodFlag == 2: # 使用 SaveResultInfo.draw_bbox
        # 转换原始数据信息
        # ori_img = np.transpose(np.transpose(Image,(1,0,2)),(0,2,1)) # (3, 424, 512) --> (424, 512, 3)
        # # draw_bbox
        # pose_bbox_label_names, _, _ = LoadYOLODeteParam()
        # ori_image = SaveResultInfo.draw_bbox(ori_img, Bbox, Labels, Scores, pose_bbox_label_names)
        # # saveFile
        # cv2.imwrite(SaveFileName, ori_image)


        # 转换原始数据信息
        ori_img = np.transpose(np.transpose(Image,(1,0,2)),(0,2,1)) # (3, 424, 512) --> (424, 512, 3)
        # draw_bbox
        # _, pose_bbox_label_names, _ = LoadSSDDeteParam()
        pose_bbox_label_names, _, _ = LoadYOLODeteParam()
        
        Image = SaveResultInfo.draw_bbox(ori_img, Bbox, Labels, Scores, pose_bbox_label_names)
        # save
        fig = plt.figure()
        plt.clf()
        plt.imshow(Image)
        plt.tight_layout()
        plt.savefig(SaveFileName)
        # plt.pause(0.1)
        plt.close()
        
        
    return 0
    
    