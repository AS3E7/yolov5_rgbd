# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:30:47 2020

@author: Administrator
"""

import numpy as np
from detect_bbox.SSD_Detect.SSD_Detecter import SSD_Detecter, LoadSSDDeteParam
from detect_bbox.YOLO_Detect.YOLO_Detecter import YOLO_Detecter, LoadYOLODeteParam

from config import one_sensor_params
DetectBboxMethodName = one_sensor_params['detect_bbox_method_name']

def Bbox_Detecter(ImageData, mode = 'Colormap'):
    """
    功能：使用ssd 方法检测目标
    数据模式：   
        mode：‘Colormap’/'RGB'
    """
    if DetectBboxMethodName == 'ssd':
        # 由于torch版本升级至1.3.0，暂时使用屏蔽 warnings 【20200715】
        import warnings
        warnings.filterwarnings("ignore")
        
        # 检测一帧数据
        pred_bbox, pred_labels, pred_scores = SSD_Detecter(ImageData)
    elif DetectBboxMethodName == 'yolov3':
        # 检测一帧数据
#        print('DetectBboxMethodName = {}'.format(DetectBboxMethodName))
        pred_bbox, pred_labels, pred_scores = YOLO_Detecter(ImageData, mode = mode)
    else:
        print('Load bbox detect method name error!')
        
#    print('Bbox_Detecter = ', pred_bbox)
    
    return pred_bbox, pred_labels, pred_scores
    
def LoadDeteParam():
    """
    功能：获取目标检测的参数信息
    """
    BboxLabelNames = ['person']
    if DetectBboxMethodName == 'ssd':
        _, BboxLabelNames, _ = LoadSSDDeteParam()
    elif DetectBboxMethodName == 'yolov3':
        BboxLabelNames, _, _ = LoadYOLODeteParam()
    else:
        print('Load bbox detect method name error!')
    return BboxLabelNames
    

if __name__ == '__main__':
    print('BboxDetect.')
