# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:07:28 2021

@author: Administrator
"""

import cv2
import os
import time
import numpy as np
from shutil import copyfile
import glob

from DatasetTrainValSetting import ReSetDatasetTrainvalInfo
from ImageProcessFuns import TwoModeDataTransform
from RGB2Depth import ReadTransformFile
from HSFFuns import ReadOneFrameDepthFile
from ColormapProcessFuns import depthToColormap


# 手动设置一个不挑选的传感器，时间段, [20210119,现只在 class.GenAllNoAnnoDataset 中使用]
SetInValidSensorAndTimeFlag = True
if SetInValidSensorAndTimeFlag == True:
    SetInValidSensorName = ['10001', '10002', '10003', '10004']    # 针对EK数据，设置外仓洗漱时间数据，暂不获取
    SetInValidPeriodTime = [16*60+00, 18*60+30] # 时间段=[16:00, 18:30]
    

class GenAllNoAnnoDataset():
    """
    功能：根据采集到的所有告警数据，生成测试数据集
    """
    def __init__(self, SrcDataFolderName, DestDataFolderName, Opt):
        self.SrcRGBImageFolderName = SrcDataFolderName['RGB']
        self.SrcDepthFolderName = SrcDataFolderName['Depth']
        self.SrcColormapImageFolderName = SrcDataFolderName['Colormap']
        self.RotParamFileName = SrcDataFolderName['RotParamFileName']
        
        self.DestRGBImageFolderName = DestDataFolderName['RGB']
        self.DestDepthFolderName = DestDataFolderName['Depth']
        self.DestColormapImageFolderName = DestDataFolderName['Colormap']
        self.DestTrainvalFolderName = DestDataFolderName['Dataset']
        
        self.OptSelectRoom = Opt['SelectRoom']
        self.OptSelectDate = Opt['SelectDate']
        self.DepthWidth = 512
        self.DepthHeight = 424
        self.DestRGBWidth = 960
        self.DestRGBHeight = 540
        
        if not os.path.exists(self.DestRGBImageFolderName[0]):
            os.makedirs(self.DestRGBImageFolderName[0])
        if not os.path.exists(self.DestDepthFolderName[0]):
            os.makedirs(self.DestDepthFolderName[0])
        if not os.path.exists(self.DestColormapImageFolderName[0]):
            os.makedirs(self.DestColormapImageFolderName[0])
        if not os.path.exists(self.DestTrainvalFolderName[0]):
            os.makedirs(self.DestTrainvalFolderName[0])

    def GenTestDataset(self):
        """
        功能：生成测试数据，未标注测试数据
        """
        DepthWidth = self.DepthWidth
        DepthHeight = self.DepthHeight
        DestRGBWidth = self.DestRGBWidth
        DestRGBHeight = self.DestRGBHeight
        
        # RGB2DepthParamsFileGroup
        RGB2DepthParamsFileGroup = dict()
        RGB2DepthParamsFileGroup['SensorName'] = []
        RGB2DepthParamsFileGroup['RotParamFileName'] = []
        RGB2DepthParamsFileGroup['RotParamFileNameDefault'] = self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0]
        for i_RGB2DepthParamsFolderName in self.RotParamFileName['RGB2DepthParamsFolderName']:
            CurRGB2DepthParamsFolderName = i_RGB2DepthParamsFolderName
            CurFolderSensorName = os.listdir(CurRGB2DepthParamsFolderName)
            for i_SensorName in CurFolderSensorName:
                CurFolderSensorCalibName = os.path.join(CurRGB2DepthParamsFolderName, i_SensorName, 'rot_param.txt')
                RGB2DepthParamsFileGroup['SensorName'].append(i_SensorName)
                RGB2DepthParamsFileGroup['RotParamFileName'].append(CurFolderSensorCalibName)
        # 读取转换矩阵文件
        # RGB2DepthTransformMatrix = ReadTransformFile(self.RotParamFileName)
        RGB2DepthTransformMatrixDefault = ReadTransformFile(self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0])
        
        # dest image folder
        DestRGBImageFolderName = self.DestRGBImageFolderName[0]
        DestColormapImageFolderName = self.DestColormapImageFolderName[0]
        DestDepthFolderName = self.DestDepthFolderName[0]
        DestTrainvalFolderName = self.DestTrainvalFolderName[0]
        DestDatasetTrainvalFolderName = os.path.join(self.DestTrainvalFolderName[0], 'coco', 'images', 'val2014')
        if not os.path.exists(DestDatasetTrainvalFolderName):
            os.makedirs(DestDatasetTrainvalFolderName)
            
        
        # SrcRGBImageFolderName/SrcColormapImageFolderName
        SrcDepthFolderName = self.SrcDepthFolderName[0]
        SrcRGBImageFileNameGroup = []
        SrcDataFolderDateGroup = os.listdir(SrcDepthFolderName) # 所有日期名称
        for i_SelectDate in SrcDataFolderDateGroup: # 遍历日期名称
            if not i_SelectDate in self.OptSelectDate: # 挑选部分日期数据
                continue
            print('Date: ', i_SelectDate)
            CurDateFileName = os.path.join(SrcDepthFolderName, i_SelectDate)
            CurDateSelectRoomNameGroup = os.listdir(CurDateFileName) # 当前日期所有房间名称
            for i_SlectRoom in CurDateSelectRoomNameGroup: # 遍历房间名称
                if not i_SlectRoom in self.OptSelectRoom: # 挑选部分房间数据
                    continue
                print('  Room: ', i_SlectRoom)
                # 遍历当前目录文件名
                CurDateSelectRoomFileName = os.path.join(SrcDepthFolderName, i_SelectDate, i_SlectRoom) # 当前房间数据地址
                CurDateSelectRoomDeothFileGlob = glob.glob(os.path.join(CurDateSelectRoomFileName, '*.dep'))
                for i_DepthFileFullName in CurDateSelectRoomDeothFileGlob:
                    i_DepthFile = os.path.basename(i_DepthFileFullName)
                    # 原始文件，三类数据文件
                    # CurDepthFileName = os.path.join(CurDateSelectRoomFileName, i_DepthFile)
                    CurRGBFileName = os.path.join(CurDateSelectRoomFileName, i_DepthFile.replace('.dep', '').replace('Depth', 'Color'))
                    CurColormapFileName = os.path.join(CurDateSelectRoomFileName, i_DepthFile.replace('.dep', ''))
                    if (not os.path.exists(CurRGBFileName)) or (not os.path.exists(CurColormapFileName)): # 判断对应的RGB/Colormap数据是否存在
                        continue
                    print('    ', i_DepthFile)
                    # 读取转换矩阵文件
                    CurRoomName = i_SlectRoom
                    CurSensorName = i_DepthFile.split('.')[0].split('_')[-1]
                    # 当前文件名日期
                    CurFileDateHour = int(i_DepthFile.split('-')[-1].split('_')[0][:2])     # 小时
                    CurFileDateMinute = int(i_DepthFile.split('-')[-1].split('_')[0][2:4])   # 分钟
                    CurFileDate24HourMinute = CurFileDateHour*60 + CurFileDateMinute    # 一天的分钟数
                    if SetInValidSensorAndTimeFlag == True:     # 针对EK数据，设置外仓洗漱时间数据，暂不获取
                        if (CurSensorName in SetInValidSensorName) and (CurFileDate24HourMinute > SetInValidPeriodTime[0]) and (CurFileDate24HourMinute < SetInValidPeriodTime[1]):
                            continue
                    # RGB2DepthTransformMatrix
                    CurRoomSensorName = CurRoomName + '_' + CurSensorName
                    if CurRoomSensorName in RGB2DepthParamsFileGroup['SensorName']:
                        CurSensorNameIdx = RGB2DepthParamsFileGroup['SensorName'].index(CurRoomSensorName)
                        RGB2DepthTransformMatrix = ReadTransformFile(RGB2DepthParamsFileGroup['RotParamFileName'][CurSensorNameIdx])
                        RGB2DepthTransformMatrix[2,0] = 0
                        RGB2DepthTransformMatrix[2,1] = 0
                    else:
                        RGB2DepthTransformMatrix = RGB2DepthTransformMatrixDefault
                        RGB2DepthTransformMatrix[2,0] = 0
                        RGB2DepthTransformMatrix[2,1] = 0
                    
                    # find file name
                    file = i_DepthFile.replace('.dep', '')
                    cur_rgb_file_name = file.replace('Depth', 'Color')
                    cur_depth_file_name = file.replace('.jpg', '.jpg.dep')
                    dest_rgb_file_name = CurRoomName + '_' + cur_rgb_file_name.replace('.jpg', '.png')
                    dest_depth_file_name = CurRoomName + '_' + cur_depth_file_name
                    dest_colormap_file_name = CurRoomName + '_' + file.replace('.jpg', '.png')
                    # SrcRGBImageFileNameGroup
                    SrcRGBImageFileNameGroup.append(dest_colormap_file_name)
                    # copy file
                    cur_rgb_full_file_name = os.path.join(CurDateSelectRoomFileName, cur_rgb_file_name)
                    cur_depth_full_file_name = os.path.join(CurDateSelectRoomFileName, cur_depth_file_name)
                    dest_rgb_full_file_name = os.path.join(DestRGBImageFolderName, dest_rgb_file_name)
                    dest_depth_full_file_name = os.path.join(DestDepthFolderName, dest_depth_file_name)
                    copyfile(cur_rgb_full_file_name, dest_rgb_full_file_name)
                    copyfile(cur_depth_full_file_name, dest_depth_full_file_name)
                    # colormap file
                    dest_colormap_full_file_name = os.path.join(DestColormapImageFolderName, dest_colormap_file_name)
                    CurFrameDepth = ReadOneFrameDepthFile(cur_depth_full_file_name, DepthWidth, DepthHeight)
                    depthToColormap(CurFrameDepth, dest_colormap_full_file_name) # generate and save colormap
                    
                    # dataset copy image file
                    DestCurDatasetColormapImageFileName = os.path.join(DestDatasetTrainvalFolderName, dest_colormap_file_name)
                    copyfile(dest_colormap_full_file_name, DestCurDatasetColormapImageFileName)
                    DestCurDatasetRGBImageFileName = os.path.join(DestDatasetTrainvalFolderName, dest_colormap_file_name.replace('.png', '_RGB.png'))
                    #     resize color image
                    CurColormapImageData = cv2.imread(dest_colormap_full_file_name) # [424, 512, 3]
                    DestImgWidth, DestImgHeight = CurColormapImageData.shape[1], CurColormapImageData.shape[0]
                    CurRGBImageData = cv2.imread(dest_rgb_full_file_name) # [540, 960, 3]
                    if not CurRGBImageData.shape[0] == DestRGBHeight:
                        CurRGBImageData = cv2.resize(CurRGBImageData, (DestRGBWidth, DestRGBHeight))
                    #     AnnoImageTransform
                    CurDestImageData = TwoModeDataTransform.AnnoImageTransform(CurRGBImageData, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
                    #     保存Color图像
                    cv2.imwrite(DestCurDatasetRGBImageFileName, CurDestImageData)


                    # debug
            #         break
            #     break
            # break

                    
        # save select image file
        TrainValTxtPrefixName = './coco/images/'
        ValTxtFileName = 'val2014_' + str(round(time.time())) + '.txt'
        DestDatasetTrainvalTxtFileName = os.path.join(DestTrainvalFolderName, ValTxtFileName)
        fp = open(DestDatasetTrainvalTxtFileName, 'w')
        for i_trainval in SrcRGBImageFileNameGroup: # [0, trainval]
            CurImageName = TrainValTxtPrefixName + 'val2014/' + i_trainval
            fp.writelines(CurImageName + '\n')
        fp.close()
        
        return SrcRGBImageFileNameGroup



class GenTestErrorDataset():
    """
    功能：根据检测错误的数据，生成测试数据集
    """
    def __init__(self, SrcDataFolderName, DestDataFolderName, CompareDataFolderName):
        self.SrcRGBImageFolderName = SrcDataFolderName['RGB']
        self.SrcDepthFolderName = SrcDataFolderName['Depth']
        self.SrcColormapImageFolderName = SrcDataFolderName['Colormap']
        self.DestRGBImageFolderName = DestDataFolderName['RGB']
        self.DestDepthFolderName = DestDataFolderName['Depth']
        self.DestColormapImageFolderName = DestDataFolderName['Colormap']
        self.DestTrainvalFolderName = DestDataFolderName['Dataset']
        self.CompareColormapFolderName = CompareDataFolderName['Colormap']
        self.RotParamFileName = SrcDataFolderName['RotParamFileName']
        self.DepthWidth = 512
        self.DepthHeight = 424
        
        if not os.path.exists(self.DestRGBImageFolderName[0]):
            os.makedirs(self.DestRGBImageFolderName[0])
        if not os.path.exists(self.DestDepthFolderName[0]):
            os.makedirs(self.DestDepthFolderName[0])
        if not os.path.exists(self.DestColormapImageFolderName[0]):
            os.makedirs(self.DestColormapImageFolderName[0])
        if not os.path.exists(self.DestTrainvalFolderName[0]):
            os.makedirs(self.DestTrainvalFolderName[0])

    def GenTestDataset(self):
        """
        功能：生成测试数据，未标注测试数据
        """
        DepthWidth = self.DepthWidth
        DepthHeight = self.DepthHeight
        
        # RGB2DepthParamsFileGroup
        RGB2DepthParamsFileGroup = dict()
        RGB2DepthParamsFileGroup['SensorName'] = []
        RGB2DepthParamsFileGroup['RotParamFileName'] = []
        RGB2DepthParamsFileGroup['RotParamFileNameDefault'] = self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0]
        for i_RGB2DepthParamsFolderName in self.RotParamFileName['RGB2DepthParamsFolderName']:
            CurRGB2DepthParamsFolderName = i_RGB2DepthParamsFolderName
            CurFolderSensorName = os.listdir(CurRGB2DepthParamsFolderName)
            for i_SensorName in CurFolderSensorName:
                CurFolderSensorCalibName = os.path.join(CurRGB2DepthParamsFolderName, i_SensorName, 'rot_param.txt')
                RGB2DepthParamsFileGroup['SensorName'].append(i_SensorName)
                RGB2DepthParamsFileGroup['RotParamFileName'].append(CurFolderSensorCalibName)
        # 读取转换矩阵文件
        # RGB2DepthTransformMatrix = ReadTransformFile(self.RotParamFileName)
        RGB2DepthTransformMatrixDefault = ReadTransformFile(self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0])
        
        # dest image folder
        CompareColormapFolderName = self.CompareColormapFolderName[0]
        DestRGBImageFolderName = self.DestRGBImageFolderName[0]
        DestColormapImageFolderName = self.DestColormapImageFolderName[0]
        DestDepthFolderName = self.DestDepthFolderName[0]
        DestTrainvalFolderName = self.DestTrainvalFolderName[0]
        DestDatasetTrainvalFolderName = os.path.join(self.DestTrainvalFolderName[0], 'coco', 'images', 'val2014')
        if not os.path.exists(DestDatasetTrainvalFolderName):
            os.makedirs(DestDatasetTrainvalFolderName)
        # CompareColormapFolderName
        CompareImageFileNameGroup = []
        CompareImageFileNameGroupSrc = glob.glob(CompareColormapFolderName+'\\*.jpg')
        for i_CompareImageFileNameGroupSrc in CompareImageFileNameGroupSrc:
            CompareImageFileNameGroup.append(os.path.basename(i_CompareImageFileNameGroupSrc).replace('ctdet.jpg', '.jpg'))
        # SrcRGBImageFolderName/SrcColormapImageFolderName
        SrcDepthFolderName = self.SrcDepthFolderName[0]
        CurRGBPostfixName = '.jpg'
        SrcRGBImageFileNameGroup = []
        for i_SelectImage in CompareImageFileNameGroup:
            file = i_SelectImage
            if file.endswith(CurRGBPostfixName) and file.find('Depth')>-1:
                CurFileNameIdx1 = file.find(CurRGBPostfixName) # '.jpg'
                CurFileNameIdx2 = file.find('_')
                CurFileNameIdx3Str = file.split('-') # 
                CurSensorName = file[CurFileNameIdx2+1:CurFileNameIdx1]
                CurDateName = CurFileNameIdx3Str[1] + CurFileNameIdx3Str[2] + CurFileNameIdx3Str[3]
                # find file
                CurRoomName = ''
                CurDataFullFolderName = os.path.join(SrcDepthFolderName, CurDateName)
                CurDataFullFolderNameGroup = os.listdir(CurDataFullFolderName)
                CurFileExistFlag = False
                for CurOneDateFolderName in CurDataFullFolderNameGroup:
                    CurFindImageFullFileName = os.path.join(CurDataFullFolderName, CurOneDateFolderName, file)
                    if os.path.exists(CurFindImageFullFileName):
                        CurFileExistFlag = True
                        CurRoomName = CurOneDateFolderName
                        break
                    
                # sensor name
                CurRoomSensorName = CurRoomName + '_' + CurSensorName
                # 读取转换矩阵文件
                if CurRoomSensorName in RGB2DepthParamsFileGroup['SensorName']:
                    CurSensorNameIdx = RGB2DepthParamsFileGroup['SensorName'].index(CurRoomSensorName)
                    RGB2DepthTransformMatrix = ReadTransformFile(RGB2DepthParamsFileGroup['RotParamFileName'][CurSensorNameIdx])
                    RGB2DepthTransformMatrix[2,0] = 0
                    RGB2DepthTransformMatrix[2,1] = 0
                else:
                    RGB2DepthTransformMatrix = RGB2DepthTransformMatrixDefault
                    RGB2DepthTransformMatrix[2,0] = 0
                    RGB2DepthTransformMatrix[2,1] = 0
                    
                # find file name
                if CurFileExistFlag == True:
                    cur_rgb_file_name = file.replace('Depth', 'Color')
                    cur_depth_file_name = file.replace('.jpg', '.jpg.dep')
                    dest_rgb_file_name = CurRoomName + '_' + cur_rgb_file_name.replace('.jpg', '.png')
                    dest_depth_file_name = CurRoomName + '_' + cur_depth_file_name
                    dest_colormap_file_name = CurRoomName + '_' + file.replace('.jpg', '.png')
                    # SrcRGBImageFileNameGroup
                    SrcRGBImageFileNameGroup.append(dest_colormap_file_name)
                    # copy file
                    cur_rgb_full_file_name = os.path.join(CurDataFullFolderName, CurOneDateFolderName, cur_rgb_file_name)
                    cur_depth_full_file_name = os.path.join(CurDataFullFolderName, CurOneDateFolderName, cur_depth_file_name)
                    dest_rgb_full_file_name = os.path.join(DestRGBImageFolderName, dest_rgb_file_name)
                    dest_depth_full_file_name = os.path.join(DestDepthFolderName, dest_depth_file_name)
                    copyfile(cur_rgb_full_file_name, dest_rgb_full_file_name)
                    copyfile(cur_depth_full_file_name, dest_depth_full_file_name)
                    # colormap file
                    dest_colormap_full_file_name = os.path.join(DestColormapImageFolderName, dest_colormap_file_name)
                    CurFrameDepth = ReadOneFrameDepthFile(cur_depth_full_file_name, DepthWidth, DepthHeight)
                    depthToColormap(CurFrameDepth, dest_colormap_full_file_name) # generate and save colormap
                    
                    # dataset copy image file
                    DestCurDatasetColormapImageFileName = os.path.join(DestDatasetTrainvalFolderName, dest_colormap_file_name)
                    copyfile(dest_colormap_full_file_name, DestCurDatasetColormapImageFileName)
                    DestCurDatasetRGBImageFileName = os.path.join(DestDatasetTrainvalFolderName, dest_colormap_file_name.replace('.png', '_RGB.png'))
                    #     resize color image
                    CurColormapImageData = cv2.imread(dest_colormap_full_file_name) # [424, 512, 3]
                    DestImgWidth, DestImgHeight = CurColormapImageData.shape[1], CurColormapImageData.shape[0]
                    CurRGBImageData = cv2.imread(dest_rgb_full_file_name) # [540, 960, 3]
                    if not CurRGBImageData.shape[0] == 540:
                        CurRGBImageData = cv2.resize(CurRGBImageData, (960, 540))
                    #     AnnoImageTransform
                    CurDestImageData = TwoModeDataTransform.AnnoImageTransform(CurRGBImageData, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
                    #     保存Color图像
                    cv2.imwrite(DestCurDatasetRGBImageFileName, CurDestImageData)
                else:
                    print('not exist file = ', file)
                    
        # save select image file
        TrainValTxtPrefixName = './coco/images/'
        DestDatasetTrainvalTxtFileName = os.path.join(DestTrainvalFolderName, 'val2014.txt')
        fp = open(DestDatasetTrainvalTxtFileName, 'w')
        for i_trainval in SrcRGBImageFileNameGroup: # [0, trainval]
            CurImageName = TrainValTxtPrefixName + 'val2014/' + i_trainval
            fp.writelines(CurImageName + '\n')
        fp.close()
        
        return SrcRGBImageFileNameGroup


class GenNoAnnoDataset():
    """
    功能：根据挑选的难度大的数据，生成未标注的测试数据集
    """
    def __init__(self, SrcDataFolderName, DestDataFolderName, CompareDataFolderName):
        self.SrcRGBImageFolderName = SrcDataFolderName['RGB']
        self.SrcDepthFolderName = SrcDataFolderName['Depth']
        self.SrcColormapImageFolderName = SrcDataFolderName['Colormap']
        self.DestRGBImageFolderName = DestDataFolderName['RGB']
        self.DestDepthFolderName = DestDataFolderName['Depth']
        self.DestColormapImageFolderName = DestDataFolderName['Colormap']
        self.DestTrainvalFolderName = DestDataFolderName['Dataset']
        self.CompareTrainvalFolderName = CompareDataFolderName['TrainvalFile']
        self.RotParamFileName = SrcDataFolderName['RotParamFileName']
        
        if not os.path.exists(self.DestRGBImageFolderName[0]):
            os.makedirs(self.DestRGBImageFolderName[0])
        if not os.path.exists(self.DestDepthFolderName[0]):
            os.makedirs(self.DestDepthFolderName[0])
        if not os.path.exists(self.DestColormapImageFolderName[0]):
            os.makedirs(self.DestColormapImageFolderName[0])
        if not os.path.exists(self.DestTrainvalFolderName[0]):
            os.makedirs(self.DestTrainvalFolderName[0])
        
    def GenTestDataset(self):
        """
        功能：生成测试数据，未标注测试数据
        """
        # RGB2DepthParamsFileGroup
        RGB2DepthParamsFileGroup = dict()
        RGB2DepthParamsFileGroup['SensorName'] = []
        RGB2DepthParamsFileGroup['RotParamFileName'] = []
        RGB2DepthParamsFileGroup['RotParamFileNameDefault'] = self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0]
        for i_RGB2DepthParamsFolderName in self.RotParamFileName['RGB2DepthParamsFolderName']:
            CurRGB2DepthParamsFolderName = i_RGB2DepthParamsFolderName
            CurFolderSensorName = os.listdir(CurRGB2DepthParamsFolderName)
            for i_SensorName in CurFolderSensorName:
                CurFolderSensorCalibName = os.path.join(CurRGB2DepthParamsFolderName, i_SensorName, 'rot_param.txt')
                RGB2DepthParamsFileGroup['SensorName'].append(i_SensorName)
                RGB2DepthParamsFileGroup['RotParamFileName'].append(CurFolderSensorCalibName)   
        # 读取转换矩阵文件
        # RGB2DepthTransformMatrix = ReadTransformFile(self.RotParamFileName)
        RGB2DepthTransformMatrixDefault = ReadTransformFile(self.RotParamFileName['RGB2DepthParamsFolderNameDefault'][0])
        
        # dest image folder
        DestRGBImageFolderName = self.DestRGBImageFolderName[0]
        DestColormapImageFolderName = self.DestColormapImageFolderName[0]
        DestDepthFolderName = self.DestDepthFolderName[0]
        DestTrainvalFolderName = self.DestTrainvalFolderName[0]
        DestDatasetTrainvalFolderName = os.path.join(self.DestTrainvalFolderName[0], 'coco', 'images', 'val2014')
        if not os.path.exists(DestDatasetTrainvalFolderName):
            os.makedirs(DestDatasetTrainvalFolderName)
        # CompareTrainvalFolderName
        CompareTrainvalFileNameGroup = []
        for i_CompareTrainvalFolderName in self.CompareTrainvalFolderName:
            CurDataTrainvalTxtFileGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(i_CompareTrainvalFolderName) 
            CompareTrainvalFileNameGroup = CompareTrainvalFileNameGroup + CurDataTrainvalTxtFileGroup
        CompareTrainvalFileNameGroup = np.unique(CompareTrainvalFileNameGroup).tolist()
        # SrcRGBImageFolderName/SrcColormapImageFolderName
        SrcRGBImageFileNameGroup = []
        FilePostfixName = ['.png']
        ExcludePostfixName = ['_Depth.png', '_RGB.png']
        for i_SrcRGBImageFolderName, i_SrcColormapImageFolderName, i_SrcDepthFolderName in zip(self.SrcRGBImageFolderName, self.SrcColormapImageFolderName, self.SrcDepthFolderName):
            print(i_SrcRGBImageFolderName, i_SrcColormapImageFolderName, i_SrcDepthFolderName)
            CurRGBImageFileNameGroup = ReSetDatasetTrainvalInfo.GenFoderFileNameGroup(i_SrcRGBImageFolderName, FilePostfixName=FilePostfixName, ExcludePostfixName=ExcludePostfixName)
            CurColormapImageFileNameGroup = ReSetDatasetTrainvalInfo.GenFoderFileNameGroup(i_SrcColormapImageFolderName, FilePostfixName=FilePostfixName, ExcludePostfixName=ExcludePostfixName)
            # CurDepthFileNameGroup = ReSetDatasetTrainvalInfo.GenFoderFileNameGroup(i_SrcDepthFolderName, FilePostfixName=['.dep'], ExcludePostfixName=ExcludePostfixName)

            for i_RGBImageFileName in CurRGBImageFileNameGroup: # RGB Image
                i_ColormapImageFileName = i_RGBImageFileName.replace('Color', 'Depth')
                if i_ColormapImageFileName in CurColormapImageFileNameGroup: # Colomap Image
                    if i_ColormapImageFileName in CompareTrainvalFileNameGroup: # used colormap image
                        continue
                    
                    # sensor name
                    CurSensorName = ''
                    CurFileNameSplit = i_ColormapImageFileName.split('_')
                    if len(CurFileNameSplit[0].split('-')) == 1:
                        if int(CurFileNameSplit[0])>1000:
                            CurSensorName = CurFileNameSplit[0] + '_' + CurFileNameSplit[-1].split('.')[0]
                        else:
                            CurSensorName = CurFileNameSplit[0]
                    else:
                        CurSensorName = ''
                    CurRoomSensorName = CurSensorName
                    # 读取转换矩阵文件
                    if CurRoomSensorName in RGB2DepthParamsFileGroup['SensorName']:
                        CurSensorNameIdx = RGB2DepthParamsFileGroup['SensorName'].index(CurRoomSensorName)
                        RGB2DepthTransformMatrix = ReadTransformFile(RGB2DepthParamsFileGroup['RotParamFileName'][CurSensorNameIdx])
                        RGB2DepthTransformMatrix[2,0] = 0
                        RGB2DepthTransformMatrix[2,1] = 0
                    else:
                        RGB2DepthTransformMatrix = RGB2DepthTransformMatrixDefault
                        RGB2DepthTransformMatrix[2,0] = 0
                        RGB2DepthTransformMatrix[2,1] = 0
                    
                    # SrcRGBImageFileNameGroup
                    SrcRGBImageFileNameGroup.append(i_ColormapImageFileName)
                    # copy image file
                    SrcCurRGBImageFileName = os.path.join(i_SrcRGBImageFolderName, i_RGBImageFileName)
                    DestCurRGBImageFileName = os.path.join(DestRGBImageFolderName, i_RGBImageFileName)
                    SrcCurColormapImageFileName = os.path.join(i_SrcColormapImageFolderName, i_ColormapImageFileName)
                    DestCurColormapImageFileName = os.path.join(DestColormapImageFolderName, i_ColormapImageFileName)
                    copyfile(SrcCurRGBImageFileName, DestCurRGBImageFileName)
                    copyfile(SrcCurColormapImageFileName, DestCurColormapImageFileName)
                    # copy depth file
                    i_DepthFileName = i_RGBImageFileName.replace('Color', 'Depth').replace('.png', '.jpg.dep')
                    SrcCurDepthFileName = os.path.join(i_SrcDepthFolderName, i_DepthFileName)
                    DestCurDepthFileName = os.path.join(DestDepthFolderName, i_DepthFileName)
                    if os.path.exists(SrcCurDepthFileName):
                        copyfile(SrcCurDepthFileName, DestCurDepthFileName)
                    else:
                        print('depth file not exist: ', i_ColormapImageFileName)
                    # dataset copy image file
                    DestCurDatasetColormapImageFileName = os.path.join(DestDatasetTrainvalFolderName, i_ColormapImageFileName)
                    copyfile(SrcCurColormapImageFileName, DestCurDatasetColormapImageFileName)
                    DestCurDatasetRGBImageFileName = os.path.join(DestDatasetTrainvalFolderName, i_ColormapImageFileName.replace('.png', '_RGB.png'))
                    #     resize color image
                    CurColormapImageData = cv2.imread(SrcCurColormapImageFileName) # [424, 512, 3]
                    DestImgWidth, DestImgHeight = CurColormapImageData.shape[1], CurColormapImageData.shape[0]
                    CurRGBImageData = cv2.imread(SrcCurRGBImageFileName) # [540, 960, 3]
                    #     AnnoImageTransform
                    CurDestImageData = TwoModeDataTransform.AnnoImageTransform(CurRGBImageData, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
                    #     保存Color图像
                    cv2.imwrite(DestCurDatasetRGBImageFileName, CurDestImageData)
                    
        # save select image file
        TrainValTxtPrefixName = './coco/images/'
        DestDatasetTrainvalTxtFileName = os.path.join(DestTrainvalFolderName, 'val2014.txt')
        fp = open(DestDatasetTrainvalTxtFileName, 'w')
        for i_trainval in SrcRGBImageFileNameGroup: # [0, trainval]
            CurImageName = TrainValTxtPrefixName + 'val2014/' + i_trainval
            fp.writelines(CurImageName + '\n')
        fp.close()
        
        return SrcRGBImageFileNameGroup
    
    def GenFolderFileNameGroup(self, FolderName, FilePostfixName=None, ExcludePostfixName=None):
        """
        功能：获取文件夹下文件名
        """
        FileNameGroup = []
        for root, dirs, files in os.walk(FolderName):
            for filename in files:
                # find valid filename
                FilePostfixNameValid = False
                ExcludePostfixNameValid = False
                for i_FilePostfixName in FilePostfixName:
                    if filename.find(i_FilePostfixName)>-1:
                        FilePostfixNameValid = True
                for i_ExcludePostfixName in ExcludePostfixName:
                    if filename.find(i_ExcludePostfixName)>-1:
                        ExcludePostfixNameValid = True
                if FilePostfixNameValid==True and ExcludePostfixNameValid==False:
                    FileNameGroup.append(filename) 
        return FileNameGroup



if __name__ == '__main__':
    print('Start.')
    
    TestCase = 4 # TestCase = 1,测试挑选的未标注图像
                 # TestCase = 2,测试Depth检测不准确的图像，HQG
                 # TestCase = 3,测试挑选的未标注图像，多仓多天
                 # TestCase = 4,测试挑选的未标注图像，EK, 深度数据
    
    if TestCase == 1: # 测试挑选的未标注图像
        SrcDataFolderName = dict()
        SrcDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210315\SelectHardCaseImage\RGB',
                                     r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\SelectHardCaseImage\RGB']
        SrcDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210315\SelectHardCaseImage\depth',
                                     r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\SelectHardCaseImage\depth']
        SrcDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210315\SelectHardCaseImage\Colormap', 
                                          r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\SelectHardCaseImage\Colormap']
        # SrcDataFolderName['RotParamFileName'] = r'rgb2depth_param.txt'       
        SrcDataFolderName['RotParamFileName'] = dict()
        SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
        SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLG',
                                                                              r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLGSimulate',]

        
        DestDataFolderName = dict()
        DestDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210517_NoAnno\SelectHardCaseImage\RGB']
        DestDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210517_NoAnno\SelectHardCaseImage\depth']
        DestDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210517_NoAnno\SelectHardCaseImage\Colormap']
        DestDataFolderName['Dataset'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210517_NoAnno\dataset_test']
        
        CompareDataFolderName = dict()
        CompareDataFolderName['TrainvalFile'] = [r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_958\train2014.txt',  
                                                 r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_958\val2014.txt']
        
        CurGenNoAnnoDataset = GenNoAnnoDataset(SrcDataFolderName, DestDataFolderName, CompareDataFolderName)
        CurGenNoAnnoDataset.GenTestDataset()
    
    if TestCase == 2: # 测试Depth检测不准确的图像，HQG
        SrcDataFolderName = dict()
        SrcDataFolderName['RGB'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
        SrcDataFolderName['Depth'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
        SrcDataFolderName['Colormap'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
        # SrcDataFolderName['RotParamFileName'] = r'rgb2depth_param.txt'
        SrcDataFolderName['RotParamFileName'] = dict()
        SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
        SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLG',
                                                                              r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLGSimulate',]
        
        DestDataFolderName = dict()
        DestDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\SelectErrorImage\RGB']
        DestDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\SelectErrorImage\depth']
        DestDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\SelectErrorImage\Colormap']
        DestDataFolderName['Dataset'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\dataset_test']
        
        CompareDataFolderName = dict()
        CompareDataFolderName['Colormap'] = [r'W:\1205ToXiongBiao\huqiugen\LG_error_depth_images20210525']
    
        CurGenTestErrorDataset = GenTestErrorDataset(SrcDataFolderName, DestDataFolderName, CompareDataFolderName)
        CurGenTestErrorDataset.GenTestDataset()
    
    if TestCase == 3: # 测试挑选的未标注图像，多仓多天
        SelectDateFlag = 2 # SelectDateFlag==1, 第一批数据，[Y:\XbiaoData\龙岗新仓服务器文件备份\picture]
                           # SelectDateFlag==2, 第二批数据，[Y:\XbiaoData\龙岗新仓程序备份_20210728\server\picture\20210401_20210727]
        
        if SelectDateFlag == 1: # 第一批数据，[Y:\XbiaoData\龙岗新仓服务器文件备份\picture]
            SrcDataFolderName = dict()
            SrcDataFolderName['RGB'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
            SrcDataFolderName['Depth'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
            SrcDataFolderName['Colormap'] = [r'Y:\XbiaoData\龙岗新仓服务器文件备份\picture']
            # SrcDataFolderName['RotParamFileName'] = r'rgb2depth_param.txt'
            SrcDataFolderName['RotParamFileName'] = dict()
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLG',
                                                                                  r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLGSimulate',]
            
            DestDataFolderName = dict()
            DestDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210706_NoAnnoAll\SelectImage\RGB']
            DestDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210706_NoAnnoAll\SelectImage\depth']
            DestDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210706_NoAnnoAll\SelectImage\Colormap']
            DestDataFolderName['Dataset'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210706_NoAnnoAll\dataset_test']
            
            Opt = dict()
            Opt['SelectRoom'] = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012']
            
            Opt['SelectDate'] = [
                                    '20201010', # 1
                                    '20201011',
                                    '20201012',
                                    '20201013',
                                    '20201014',
                                    '20201015', # 2
                                 ]
            CurGenTestErrorDataset = GenAllNoAnnoDataset(SrcDataFolderName, DestDataFolderName, Opt)
            CurGenTestErrorDataset.GenTestDataset()
            
        elif SelectDateFlag == 2: # 第二批数据，[Y:\XbiaoData\龙岗新仓程序备份_20210728\server\picture\20210401_20210727]
            SrcDataFolderName = dict()
            SrcDataFolderName['RGB'] = [r'Y:\XbiaoData\龙岗新仓程序备份_20210728\server\picture\20210401_20210727']
            SrcDataFolderName['Depth'] = [r'Y:\XbiaoData\龙岗新仓程序备份_20210728\server\picture\20210401_20210727']
            SrcDataFolderName['Colormap'] = [r'Y:\XbiaoData\龙岗新仓程序备份_20210728\server\picture\20210401_20210727']
            # SrcDataFolderName['RotParamFileName'] = r'rgb2depth_param.txt'
            SrcDataFolderName['RotParamFileName'] = dict()
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLG',
                                                                                  r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\NewLGSimulate',]
            
            DestDataFolderName = dict()
            DestDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210804_NoAnnoAll\SelectImage\RGB']
            DestDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210804_NoAnnoAll\SelectImage\depth']
            DestDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210804_NoAnnoAll\SelectImage\Colormap']
            DestDataFolderName['Dataset'] = [r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210804_NoAnnoAll\dataset_test']
            
            Opt = dict()
            Opt['SelectRoom'] = ['1001', '1002', '1003', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1012']
            
            Opt['SelectDate'] = [
                                    '20210410', 
                                    '20210420',
                                    '20210430',
                                    '20210510',
                                    '20210520',
                                    '20210530', 
                                    '20210610',
                                    '20210620',
                                    '20210630', 
                                    '20210710',
                                    '20210720',
                                 ]
            CurGenTestErrorDataset = GenAllNoAnnoDataset(SrcDataFolderName, DestDataFolderName, Opt)
            CurGenTestErrorDataset.GenTestDataset()
    
    
    
    
    
    elif TestCase == 4: # 测试挑选的未标注图像，EK, 深度数据
        SelectDateFlag = 2  # SelectDateFlag = 1, 从HSF文件中选择图片，生成val 文件
                            # SelectDateFlag = 2, 从告警文件（picture）中选择图片
    
        if SelectDateFlag == 1:
            # ----------- RGBD_20211103 -----------
            SrcDepthImageFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\SelectHardCaseImage\Colormap'
            DestValFileName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\dataset_test\val2014.txt'
            
            # loop images
            SrcImageNameGroup = []
            for root, dirs, files in os.walk(SrcDepthImageFolderName):
                for filename in files:
                    if filename.endswith('.png'):
                        SrcImageNameGroup.append(filename)
            # write filename
            PrefixFileName = './coco/images/val2014/'
            fp = open(DestValFileName, 'w')
            for i_filename in SrcImageNameGroup:
                fp.writelines(PrefixFileName + i_filename + '\n')
            fp.close()
            
        elif SelectDateFlag == 2:
            # ----------- RGBD_20211105 -----------
            SrcDataFolderName = dict()
            SrcDataFolderName['RGB'] = [r'Y:\XbiaoData\二看程序备份_20211105\server\picture']
            SrcDataFolderName['Depth'] = [r'Y:\XbiaoData\二看程序备份_20211105\server\picture']
            SrcDataFolderName['Colormap'] = [r'Y:\XbiaoData\二看程序备份_20211105\server\picture']
            # SrcDataFolderName['RotParamFileName'] = r'rgb2depth_param.txt'
            SrcDataFolderName['RotParamFileName'] = dict()
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
            SrcDataFolderName['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'D:\xiongbiao\Data\HumanDete\RGB2DepthRotParams\SZ2KSS',]
            
            DestDataFolderName = dict()
            DestDataFolderName['RGB'] = [r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211105\SelectHardCaseImage\RGB']
            DestDataFolderName['Depth'] = [r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211105\SelectHardCaseImage\depth']
            DestDataFolderName['Colormap'] = [r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211105\SelectHardCaseImage\Colormap']
            DestDataFolderName['Dataset'] = [r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211105\dataset_test']
            
            Opt = dict()
            Opt['SelectRoom'] = ['1001', '1002']
            Opt['SelectDate'] = [
                                    '20211021', 
                                    '20211022',
                                    '20211023',
                                    '20211024',
                                    '20211025',
                                    '20211026', 
                                    '20211027',
                                    '20211028',
                                    '20211029', 
                                    '20211030',
                                    '20211031',
                                    '20211101',
                                    '20211102',
                                    '20211103',
                                    '20211104',
                                    '20211105',
                                 ]
            CurGenTestErrorDataset = GenAllNoAnnoDataset(SrcDataFolderName, DestDataFolderName, Opt)
            CurGenTestErrorDataset.GenTestDataset()
    
    
    
    