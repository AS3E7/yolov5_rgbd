# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:13:48 2020

@author: HYD
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from chainercv.visualizations import vis_bbox

from detect_bbox.SSD_Detect import SaveResultInfo
from detect_bbox.SSD_Detect.SSD_Detecter import LoadSSDDeteParam

def PlotDeteResult(Image, Bbox, Labels, Scores, SaveFileName = ''):
    """
    功能：显示检测的结果
    """
#    print(Image.shape)
    PlotResultMethodFlag = 2 # PlotResultMethodFlag ==1，使用 chainercv.visualizations
                             # PlotResultMethodFlag ==2，使用 SaveResultInfo.draw_bbox
    
    if PlotResultMethodFlag == 1: # 使用 chainercv.visualizations
        _, pose_bbox_label_names, _ = LoadSSDDeteParam()
#        fig = plt.figure()
#        plt.clf()
#        ax1 = fig.add_subplot(1, 1, 1)
#        vis_bbox(Image, Bbox, Labels, Scores, label_names=pose_bbox_label_names, ax=ax1) # (3, height, width)
#        plt.tight_layout()
#        plt.savefig(SaveFileName)
#        plt.close()
    elif PlotResultMethodFlag == 2: # 使用 SaveResultInfo.draw_bbox
#        # 转换原始数据信息
#        ori_img = np.transpose(np.transpose(Image,(1,0,2)),(0,2,1)) # (3, 424, 512) --> (424, 512, 3)
#        # draw_bbox
#        _, pose_bbox_label_names, _ = LoadSSDDeteParam()
#        ori_image = SaveResultInfo.draw_bbox(ori_img, Bbox, Labels, Scores, pose_bbox_label_names)
#        # saveFile
#        cv2.imwrite(SaveFileName, ori_image)


        # 转换原始数据信息
        ori_img = np.transpose(np.transpose(Image,(1,0,2)),(0,2,1)) # (3, 424, 512) --> (424, 512, 3)
        # draw_bbox
        _, pose_bbox_label_names, _ = LoadSSDDeteParam()
        Image = SaveResultInfo.draw_bbox(ori_img, Bbox, Labels, Scores, pose_bbox_label_names)
        # save
        fig = plt.figure()
        plt.clf()
        plt.imshow(Image)
        plt.show()
        plt.tight_layout()
        plt.savefig(SaveFileName)
        plt.close()
    
    return 0
    
    