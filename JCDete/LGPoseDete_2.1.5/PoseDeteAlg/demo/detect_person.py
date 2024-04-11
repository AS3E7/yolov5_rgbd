# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:16:24 2020

@author: HYD
"""
import logging as lgmsg
import time
import os
import cv2
import numpy as np

from util.DataPreprocess import DepthPreproces
from util.ReadConfig import ReadIniConfig, TransformConfigInfo
from detect_bbox.Bbox_Detecter import Bbox_Detecter
from detect_pose.Pose_Detecter import Pose_Detecter
from detect_alarm.AlarmDete import AlarmDete
from util.RGB2Depth import TransformRGBBBoxInfoToDepth
from util.SaveData import LogSaveCategoryIndexFun
from detect_bbox.YOLO_Detect.utils.datasets import trans_rgb2depth
from config import online_offline_type, one_sensor_params, log_info


# 选择目标检测方法
SelectDeteMethodFlag = 1 # 使用 bbox 目标框检测方法
                         # 使用 pose-detect 检测方法
                         
# 图像转换方法
ImageTransMethod = 2 # ImageTransMethod == 0, 输入深度，不进行变换
                     # ImageTransMethod == 1, 按传感器距离值，进行伪彩色配色
                     # ImageTransMethod == 2, 深度数据配准，去数据Z方向高度值，进行伪彩色配色
                         
# RGB坐标转换Depth坐标
OnOffLineType = online_offline_type # 'OnLine','OffLine'
RGB2DepthFolderNameSrc = one_sensor_params['rgb2depth_foldername']
if OnOffLineType == 'OnLine':
    RotParamFileName = 'demo/log/rgb2depth_param.txt'
    RGB2DepthFolderName = os.path.join('demo', RGB2DepthFolderNameSrc) # 新添加的 转换关系
else:
    RotParamFileName = 'log/rgb2depth_param.txt'
    RGB2DepthFolderName = RGB2DepthFolderNameSrc # 新添加的 转换关系
    
# log 信息
ConfigDebugInfo = log_info['debug_info']
ConfigDebugTypeIndex = LogSaveCategoryIndexFun()

def DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = 0, PtsSubBG=None):
    """
    功能：使用深度数据检测目标
    输入：
        DepthData：Depth数据，一维数据
		DepthWidth：Depth图像宽度，如：Kinect2.0的Depth图像宽度为512；
		DepthHeight：Depth图像高度，如：Kinect2.0的Depth图像高度为424；
		Pts：点云数据，二维数据，如：Kinect2.0的点云数据尺寸为[3 x 217088]；
    """
    if DebugFlag > 0: 
        lgmsg.debug('  DetectDepth Function, DebugFlag = {}'.format(DebugFlag))
            
    # 数据预处理
    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
    CurH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H'] # H
    # 目标bbox检测使用的点云数据
    ProcessDepth, _, _ = DepthPreproces(DepthData, PtsDeteBbox, DepthWidth, DepthHeight, CurH, ImageTransMethod)
    # 纯点云计算使用的数据
    _, ProcessPts, DepthToCloudPointIdx = DepthPreproces(DepthData, Pts, DepthWidth, DepthHeight, CurH, ImageTransMethod, OnlyProcessPts=True)
#    print('ProcessDepth size = ', ProcessDepth.shape)
    
    # 图像数据检测
    CurImgDeteResult = dict()
    if SelectDeteMethodFlag == 1:
        PredBbox, PredLabels, PredScores = Bbox_Detecter(ProcessDepth)
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    elif SelectDeteMethodFlag == 2:
        PredBbox, PredLabels, PredScores = Pose_Detecter(ProcessDepth)
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    CurImgDeteResult['Data'] = ProcessDepth # 保存结果为： (6, 424, 512)
    
    # print result
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.DeteObjInfo]) > 0:
        lgmsg.debug('PredBbox = {}, PredScores = {}'.format(PredBbox, PredScores))
    
    # 功能检测
    AlarmResult = AlarmDete(PredBbox, PredLabels, PredScores, ProcessPts, DepthWidth, DepthHeight, DepthToCloudPointIdx, WorldTime, ConfigInfo, PreDete, DebugFlag, SrcPts = PtsDeteBbox, PtsSubBG=PtsSubBG)

    return CurImgDeteResult, AlarmResult
    
def DetectRGBDepth(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = 0, PtsSubBG=None):
    """
    功能：使用RGB数据检测目标，对应深度数据位置和点云空间位置
    """
    if DebugFlag > 0: 
        lgmsg.debug('  DetectDepth Function, DebugFlag = {}'.format(DebugFlag))
            
    # 数据预处理
    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
    CurH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H'] # H
    # 目标bbox检测使用的点云数据
    ProcessDepth, _, _ = DepthPreproces(DepthData, PtsDeteBbox, DepthWidth, DepthHeight, CurH, ImageTransMethod)
    # 纯点云计算使用的数据
    _, ProcessPts, DepthToCloudPointIdx = DepthPreproces(DepthData, Pts, DepthWidth, DepthHeight, CurH, ImageTransMethod, OnlyProcessPts=True)
    
    # 图像数据检测
    CurImgDeteResult = dict()
    if SelectDeteMethodFlag == 1:
        # 转换输入的 RGB 数据格式
        RGBImageData = np.transpose(RGBImageData,(2,0,1))
        RGBImageData = RGBImageData[[2,1,0],:,:]
        # RGBImage Bbox_Detecter
        PredBbox_RGB, PredLabels_RGB, PredScores_RGB = Bbox_Detecter(RGBImageData, mode = 'RGB')
        # DepthImage detect info
        PredBbox, PredScores, PredLabels = TransformRGBBBoxInfoToDepth(PredBbox_RGB, PredScores_RGB, PredLabels_RGB, RotParamFileName, DepthWidth, DepthHeight)

        # 返回检测目标结果
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    elif SelectDeteMethodFlag == 2:
        PredBbox, PredLabels, PredScores = Pose_Detecter(ProcessDepth)
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    CurImgDeteResult['Data'] = ProcessDepth
    
    # print result
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.DeteObjInfo]) > 0:
        lgmsg.debug('PredBbox = {}, PredScores = {}'.format(PredBbox, PredScores))
    
    # 功能检测
    AlarmResult = AlarmDete(PredBbox, PredLabels, PredScores, ProcessPts, DepthWidth, DepthHeight, DepthToCloudPointIdx, WorldTime, ConfigInfo, PreDete, DebugFlag, SrcPts = PtsDeteBbox, PtsSubBG=PtsSubBG)

    return CurImgDeteResult, AlarmResult

def DetectRGBD(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = 0, PtsSubBG=None):
    """
    功能：使用RGBD数据检测目标
    输入：
        RGBImageData: RGB数据，[540x960x3]
        RGBWidth: 图像宽
        RGBHeight: 图像高
        DepthData：Depth数据，一维数据
		DepthWidth：Depth图像宽度，如：Kinect2.0的Depth图像宽度为512；
		DepthHeight：Depth图像高度，如：Kinect2.0的Depth图像高度为424；
		Pts：点云数据，二维数据，如：Kinect2.0的点云数据尺寸为[3 x 217088]；
        PtsDeteBbox：目标检测点云数据
    """
    if DebugFlag > 0: 
        lgmsg.debug('  DetectDepth Function, DebugFlag = {}'.format(DebugFlag))
            
    # 数据预处理
    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
    CurH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H'] # H
    # 目标bbox检测使用的点云数据
    ProcessDepth, _, _ = DepthPreproces(DepthData, PtsDeteBbox, DepthWidth, DepthHeight, CurH, ImageTransMethod)
    
    # 纯点云计算使用的数据
    _, ProcessPts, DepthToCloudPointIdx = DepthPreproces(DepthData, Pts, DepthWidth, DepthHeight, CurH, ImageTransMethod, OnlyProcessPts=True)
    
    # 图像数据检测
    CurImgDeteResult = dict()
    if SelectDeteMethodFlag == 1:
        # 输入RGBD检测图像数据
        #    当前传感器对应的 rgb2depth 文件
        RGB2DepthFileName = os.path.join(RGB2DepthFolderName, str(ConfigInfo['ROOM_INFO']['roomid'])+'_'+str(ConfigInfo['SENSORS_INFO']['currsensorid']), 'rot_param.txt')
        RGBImageData = trans_rgb2depth(RGBImageData, RGB2DepthFileName, DepthWidth, DepthHeight)# 转换RGB对应Depth关系
        #    Depth + RGB
        DepthImageTrans = np.transpose(ProcessDepth,(1,2,0)) # (424, 512, 3),[RGR]
        DepthImageTrans = DepthImageTrans[:,:,[2,1,0]] # (424, 512, 3),[RGB]
        ProcessDepth = np.concatenate((DepthImageTrans, RGBImageData),2) # (424, 512, 6)
        # Bbox_Detecter
        PredBbox, PredLabels, PredScores = Bbox_Detecter(ProcessDepth, mode = 'RGBD')
        
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    elif SelectDeteMethodFlag == 2:
        PredBbox, PredLabels, PredScores = Pose_Detecter(ProcessDepth)
        CurImgDeteResult['Bbox'] = PredBbox
        CurImgDeteResult['Label'] = PredLabels
        CurImgDeteResult['Score'] = PredScores
    CurImgDeteResult['Data'] = np.transpose(ProcessDepth, (2,0,1))[::-1,:,:] # 保存结果为： (6, 424, 512)
    
    # print result
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.DeteObjInfo]) > 0:
        lgmsg.debug('PredBbox = {}, PredScores = {}'.format(PredBbox, PredScores))
    
    # 功能检测
    AlarmResult = AlarmDete(PredBbox, PredLabels, PredScores, ProcessPts, DepthWidth, DepthHeight, DepthToCloudPointIdx, WorldTime, ConfigInfo, PreDete, DebugFlag, SrcPts = PtsDeteBbox, PtsSubBG=PtsSubBG)

    return CurImgDeteResult, AlarmResult
    
    