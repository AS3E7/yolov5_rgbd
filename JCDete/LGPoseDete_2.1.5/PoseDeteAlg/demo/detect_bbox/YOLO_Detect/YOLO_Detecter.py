# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 19:16:57 2020

@author: Administrator
"""
from config import online_offline_type
import sys
if online_offline_type == 'OnLine':
    sys.path.append('demo/detect_bbox/YOLO_Detect')
else:
    sys.path.append('detect_bbox/YOLO_Detect')

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
import numpy as np
from numpy import random
#import logging as lgmsg

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImage_set
from .utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from .utils.torch_utils import select_device, load_classifier, time_synchronized

from config import one_sensor_params

# 模型名称设置
SelectBboxDeteModelName = one_sensor_params['detect_bbox_model_name']
DetectBboxInputImageSize = one_sensor_params['detect_bbox_input_image_size']
DetectBboxMethodName = one_sensor_params['detect_bbox_method_name']
DeteBboxNmsThod = one_sensor_params['detect_bbox_nms_thod']# 目标检测目标框 nms 阈值
DeteBboxScoreThod = one_sensor_params['detect_bbox_score_thod']

# 选择目标检测的类别信息
PoseSelectType = one_sensor_params['detect_bbox_pose_type'] # PoseSelectType == 1, 只检测一类目标： ['person']
                   # PoseSelectType == 2, 检测两类类目标： ('nolying', 'lying')
                   # PoseSelectType == 3, 检测三类目标：['sitting','standing','lying'] 
PoseSelectIndex = one_sensor_params['detect_bbox_pose_index'] # 多类检测序号变化
POSE_CLASSES_INDEX = None # 变换序号默认值
if PoseSelectType == 1: # 只检测一类目标： ['person']
    POSE_CLASSES = ['person'] 
elif PoseSelectType == 2: # 检测三类目标：['nolying','lying']
    POSE_CLASSES = ('nolying', 'lying')
elif PoseSelectType == 3: # 检测三类目标：['sitting','standing','lying']
    POSE_CLASSES = ('sitting', 'standing', 'lying')
    POSE_CLASSES_INDEX = PoseSelectIndex
else:
    POSE_CLASSES = ['person']

# TrainPthModelName
if type(SelectBboxDeteModelName) == str:
    TrainPthModelName = SelectBboxDeteModelName + '.pt'
elif type(SelectBboxDeteModelName) == list:
    SelectBboxDeteModelNameNum = len(SelectBboxDeteModelName)
    if SelectBboxDeteModelNameNum == 1:
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pt'
    elif SelectBboxDeteModelNameNum == 2: # 默认只有两个网络模型,[colormap, rgb]
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pt'
        TrainPthModelName_2 = SelectBboxDeteModelName[1] + '.pt'
    else:
        print('SelectBboxDeteModelName number error.')
        TrainPthModelName = SelectBboxDeteModelName[0] + '.pt'
else:
    print('SelectBboxDeteModelName type error.')
    TrainPthModelName = SelectBboxDeteModelName + '.pt'

# 在线/离线文件地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SelectModelFileName = 'demo/detect_bbox/YOLO_Detect/weights'
else:
    SelectModelFileName = 'detect_bbox/YOLO_Detect/weights'

# 导入模型
ModelFileName = os.path.join(SelectModelFileName, TrainPthModelName) # 模型地址
cuda_flag = False
if DetectBboxMethodName == 'yolov3':
    if torch.cuda.is_available(): # gpu
        opt_img_size = DetectBboxInputImageSize
        imgsz = opt_img_size
        # 初始化
        opt_device = '0'
        device = select_device(opt_device)
        # 加载权重
        model = attempt_load(ModelFileName, map_location=device)
        imgsz = check_img_size(imgsz, s=model.stride.max())
        # Eval模式
        model.to(device).eval()
        cuda_flag = True
#        lgmsg.info('Finished yolo loading gpu-model : {} !'.format(TrainPthModelName))
    else:
        opt_img_size = DetectBboxInputImageSize
        imgsz = opt_img_size
        # 初始化
        device = select_device(device='cpu')
        # 加载权重
        model = attempt_load(ModelFileName, map_location=device)
        imgsz = check_img_size(imgsz, s=model.stride.max())
        # Eval模式
        model.to(device).eval()
        print('Finished yolo loading cpu-model : {} !'.format(TrainPthModelName))
#        lgmsg.info('Finished yolo loading cpu-model : {} !'.format(TrainPthModelName))
else:
    model = dict()
    
# TrainPthModelName_2
try:
    ModelFileName_2 = os.path.join(SelectModelFileName, TrainPthModelName_2) # 模型地址
    cuda_flag = False
    if DetectBboxMethodName == 'yolov3':
        if torch.cuda.is_available(): # gpu
            opt_img_size = DetectBboxInputImageSize
            imgsz = opt_img_size
            # 初始化
            opt_device = '0'
            device = select_device(opt_device)
            # 加载权重
            model_2 = attempt_load(ModelFileName_2, map_location=device)
            imgsz = check_img_size(imgsz, s=model_2.stride.max())
            # Eval模式
            model_2.to(device).eval()
            cuda_flag = True
            print('Finished yolo loading gpu-model_2 : {} !'.format(TrainPthModelName_2))
        else:
            opt_img_size = DetectBboxInputImageSize
            imgsz = opt_img_size
            # 初始化
            device = select_device(device='cpu')
            # 加载权重
            model_2 = attempt_load(ModelFileName_2, map_location=device)
            imgsz = check_img_size(imgsz, s=model_2.stride.max())
            # Eval模式
            model_2.to(device).eval()
            print('Finished yolo loading cpu-model_2 : {} !'.format(TrainPthModelName_2))
    else:
        model_2 = dict()
except:
    print('ModelFileName_2 not exist.')
    

def LoadYOLODeteParam():
    """
    导入网络其他参数
    """

    # pose pattern
    pose_bbox_label_names = POSE_CLASSES
    # visual_threshold
    conf_thres = DeteBboxScoreThod
    # ious
    iou_thres = DeteBboxNmsThod
    
    return pose_bbox_label_names, conf_thres, iou_thres

def YOLO_Detecter(ImageData, mode = 'Colormap'):
    """
    功能：使用yolo 方法检测目标
    """
    # 获取其他参数
    pose_bbox_label_names, conf_thres, iou_thres = LoadYOLODeteParam()
    # 检测一帧数据
    if mode == 'Colormap':
        img_trans = np.transpose(ImageData,(1,2,0)) # (424, 512, 3),[RGR]
        img_trans = img_trans[:,:,[2,1,0]] # (424, 512, 3),[RGB]
        pred_bbox, pred_labels, pred_scores = test_one_depth_image(model, img_trans, imgsz, conf_thres, iou_thres)
    elif mode == 'RGB':
        img_trans = np.transpose(ImageData,(1,2,0)) # (424, 512, 3),[RGR]
        pred_bbox, pred_labels, pred_scores = test_one_depth_image(model_2, img_trans, imgsz, conf_thres, iou_thres)
    elif mode == 'RGBD': 
        pred_bbox, pred_labels, pred_scores = test_one_depth_image(model, ImageData, imgsz, conf_thres, iou_thres) # RGBD model
    else:
        img_trans = np.transpose(ImageData,(1,2,0)) # (424, 512, 3),[RGR]
        img_trans = img_trans[:,:,[2,1,0]] # (424, 512, 3),[RGB]
        pred_bbox, pred_labels, pred_scores = test_one_depth_image(model, img_trans, imgsz, conf_thres, iou_thres)
        
    return pred_bbox, pred_labels, pred_scores


def test_one_depth_image(model_test, testset, img_size, opt_conf_thres, opt_iou_thres):
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
    
    
    # if testset.shape[2] == 3:
    #     cv2.imwrite('temp_img_trans_depth.png', testset[:,:,:3])
    # else:
    #     cv2.imwrite('temp_img_trans_depth.png', testset[:,:,:3])
    #     cv2.imwrite('temp_img_trans_rgb.png', testset[:,:,3:])
    
    
    img_trans = testset
    # LoadImages
    dataset = LoadImage_set(img_trans, img_size=img_size)
    # 运行推测,得到检测
    img = torch.from_numpy(dataset).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  
    # Inference
    pred, train_out = model_test(img, augment=False)

    if len(train_out)==3:
        pred = pred
    elif len(train_out)==5:
        # multi output
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
        # src rgbd
        pred1 = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
        pred2 = pred[:, inf_out_num_bbox_type1:inf_out_num_bbox_type1+inf_out_num_bbox_type2, :] # 416: [bs, 429, 6]
        pred3 = pred[:, -inf_out_num_bbox_type3:, :] # 416: [bs, 429, 6]
        pred = pred1
    
    elif len(train_out)==9:
        # multi output
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2_1 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type2_2 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3]
        inf_out_num_bbox_type2_3 = train_out[5].shape[1] * train_out[5].shape[2] * train_out[5].shape[3]
        inf_out_num_bbox_type2 = inf_out_num_bbox_type2_1 + inf_out_num_bbox_type2_2 + inf_out_num_bbox_type2_3
        inf_out_num_bbox_type3_1 = train_out[6].shape[1] * train_out[6].shape[2] * train_out[6].shape[3] # output3
        inf_out_num_bbox_type3_2 = train_out[7].shape[1] * train_out[7].shape[2] * train_out[7].shape[3]
        inf_out_num_bbox_type3_3 = train_out[8].shape[1] * train_out[8].shape[2] * train_out[8].shape[3]
        inf_out_num_bbox_type3 = inf_out_num_bbox_type3_1 + inf_out_num_bbox_type3_2 + inf_out_num_bbox_type3_3
        # src rgbd
        pred1 = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
        pred2 = pred[:, inf_out_num_bbox_type1:inf_out_num_bbox_type1+inf_out_num_bbox_type2, :] # 416: [bs, 9009, 6]
        pred3 = pred[:, -inf_out_num_bbox_type3:, :] # 416: [bs, 9009, 6]
        pred = pred1
        
    else:
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
        pred = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
    
    # NMS应用
    pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=None, agnostic=True)
    # predict
    pred_bbox = []
    pred_labels = []
    pred_scores = []
    for i, det in enumerate(pred):  # 检测每一个图像
        if det is not None and len(det): # det: [[x1, y1, x2, y2, score, label],..]
            det = scale_coords(img.shape[2:], det, img_trans.shape)# .round()
            result = det.cpu().detach().numpy()
            # pred info
            pred_bbox = result[:,:4].round() 
            pred_bbox = pred_bbox[:,[1,0,3,2]]# [y1,x1,y2,x2]
            
            pred_scores = result[:,4]
            for i_obj in range(result.shape[0]):
                pred_labels.append(int(result[i_obj,5]))
                
    # trans label
    if POSE_CLASSES_INDEX is not None:
        pred_labels_trans = []
        for i_obj in range(len(pred_scores)):
            pred_labels_trans.append(POSE_CLASSES_INDEX[pred_labels[i_obj]])
        pred_labels = pred_labels_trans
    
    return pred_bbox, pred_labels, pred_scores


if __name__ == '__main__':
    print('YOLO_Detecter.')
    
#    TestYOLO_DetecterFlag = 1 # 测试 yolo 检测效果
#    
#    if TestYOLO_DetecterFlag == 1:
#        SrcTestImageFolderName = r'C:\software\yolov3-dome\demo\temp'
#        DestSaveImageFolderName = r'C:\software\LGDete\demo\detect_bbox\YOLO_Detect\temp'
#        
#        input_img = 416
#        save_name = DestSaveImageFolderName
#        if not os.path.exists(save_name):
#            os.makedirs(save_name)
#        with torch.no_grad():
#            path = str(Path(SrcTestImageFolderName))
#            files = []
#            if os.path.isdir(path):
#                files = sorted(glob.glob(os.path.join(path, '*.*')))
#            elif os.path.isfile(path):
#                files = [path]
#            for i in files:
#                filepath, tempfilename = os.path.split(i)
#                img0 = cv2.imread(i)
#                t1 = time.time()
#                pred_bbox, pred_labels, pred_scores = YOLO_Detecter(img0)
#                # imshow
                

    
    
    
    
    
    