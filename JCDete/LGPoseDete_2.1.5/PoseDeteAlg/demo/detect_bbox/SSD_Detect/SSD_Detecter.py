# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:14:03 2020

@author: HYD
"""

import numpy as np
import os
import argparse
#import logging as lgmsg

import torch
from torch.autograd import Variable

from .data import AnnotationTransform, POSEDetection, BaseTransform, POSE_CLASSES
from .ssd import build_ssd
from config import online_offline_type, one_sensor_params

# 模型名称设置
PoseClassNum = len(POSE_CLASSES) # 检测目标类别
SelectBboxDeteModelName = one_sensor_params['detect_bbox_model_name']
DetectBboxMethodName = one_sensor_params['detect_bbox_method_name']
# TrainPthModelName
if type(SelectBboxDeteModelName) == str:
    TrainPthModelName = SelectBboxDeteModelName + '.pth'
elif type(SelectBboxDeteModelName) == list:
    SelectBboxDeteModelNameNum = len(SelectBboxDeteModelName)
    if SelectBboxDeteModelNameNum == 1:
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pth'
    elif SelectBboxDeteModelNameNum == 2: # 默认只有两个网络模型,[colormap, rgb]
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pth'
        TrainPthModelName_2 = SelectBboxDeteModelName[1] + '.pth'
    else:
        print('SelectBboxDeteModelName number error.')
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pth'
else:
    print('SelectBboxDeteModelName type error.')
    TrainPthModelName = SelectBboxDeteModelName + '.pth'
    
# 在线/离线文件地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SelectModelFileName = 'demo/detect_bbox/SSD_Detect/weights'
else:
    SelectModelFileName = 'detect_bbox/SSD_Detect/weights'

# 导入模型
# TrainPthModelName
ModelFileName = os.path.join(SelectModelFileName, TrainPthModelName) # 模型地址
num_classes = len(POSE_CLASSES) + 1 # +1 background
cuda_flag = False
if DetectBboxMethodName == 'ssd':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        net = build_ssd('test', 300, num_classes)
        net.load_state_dict(torch.load(ModelFileName))
        net = net.cuda()
        cuda_flag = True
#        lgmsg.info('Finished loading gpu-model : {} !'.format(TrainPthModelName))
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        net = build_ssd('test', 300, num_classes)
        net.load_state_dict(torch.load(ModelFileName,  map_location='cpu')) # use only cpus
        print('Finished loading cpu-model : {} !'.format(TrainPthModelName))
#        lgmsg.info('Finished loading cpu-model : {} !'.format(TrainPthModelName))
else:
    net = dict()
    
# TrainPthModelName_2
try:
    ModelFileName_2 = os.path.join(SelectModelFileName, TrainPthModelName_2) # 模型地址
    num_classes = len(POSE_CLASSES) + 1 # +1 background
    cuda_flag = False
    if DetectBboxMethodName == 'ssd':
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            net_2 = build_ssd('test', 300, num_classes)
            net_2.load_state_dict(torch.load(ModelFileName_2))
            net_2 = net_2.cuda()
            cuda_flag = True
            print('Finished loading gpu-model_2 : {} !'.format(TrainPthModelName_2))
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            net_2 = build_ssd('test', 300, num_classes)
            net_2.load_state_dict(torch.load(ModelFileName_2,  map_location='cpu')) # use only cpus
            print('Finished loading cpu-model_2 : {} !'.format(TrainPthModelName_2))
    else:
        net_2 = dict()
except:
    print('ModelFileName_2 not exist.')

    
def LoadSSDDeteParam():
    """
    导入网络其他参数
    """

    # images mean value
    means = (104, 117, 123) # means = (122, 156, 98)
    
    # pose pattern
    pose_bbox_label_names = POSE_CLASSES

    # visual_threshold
    visual_threshold = one_sensor_params['detect_bbox_score_thod']
    
    return means, pose_bbox_label_names, visual_threshold

def SSD_Detecter(ImageData):
    """
    功能：使用ssd 方法检测目标
    """
    # 获取其他参数
    means, pose_bbox_label_names, visual_threshold = LoadSSDDeteParam()
    # 检测一帧数据
    pred_bbox, pred_labels, pred_scores = test_one_depth_image(net, cuda_flag, ImageData, BaseTransform(net.size, means), visual_threshold)
    
    return pred_bbox, pred_labels, pred_scores

    
def test_one_depth_image(net_dete, cuda, testset, transform, thresh):
    # inputs: 
    #       save_folder: save folder
    #       net: detect network
    #       cuda: ture or false
    #       testset: test image
    #       transform: depth image preprocess
    #       thresh: true object thresh
    # outputs:
    #       pred_bbox: bbox
    #       pred_labels: label
    #       pred_scores: score
    

    # predict thod
#    pred_thod = 0.5 # init: 0.5
    pred_thod = thresh
#    print('pred_thod = {}'.format(pred_thod))
    
    # test one image
    image0 = testset
    img = image0
    img = np.transpose(img,(1,2,0)) 
    
    # net
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1) # image change to [threechannle, height, width]
    x = Variable(x.unsqueeze(0))
    if cuda:
        x = x.cuda()
    y = net_dete(x)      # forward pass
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])

    # predict
    pred_bbox = []
    pred_labels = []
    pred_scores = []
    pred_num = 0
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= pred_thod: 
            score = detections[0, i, j, 0]
            # label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[1], pt[0], pt[3], pt[2]) # correspnd to [424 512]
            
            pred_num += 1
            j += 1
            # pred_bbox
            pred_bbox.append(coords)
            pred_labels.append(i-1)
            pred_scores.append(score)
    
    return pred_bbox, pred_labels, pred_scores
    
if __name__ == '__main__':
    print('Start.')
    

    print('End.')    
