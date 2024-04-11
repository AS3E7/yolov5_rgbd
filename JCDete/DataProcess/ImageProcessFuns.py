# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:25:15 2021

@author: Administrator
"""

import numpy as np
import os
import cv2
import copy
from shutil import copyfile

from RGB2Depth import ReadTransformFile, TransformBBoxInfo


def ResizeImageAnnoInfo(SrcImageFolderName, SrcLabelFolderName, DestImageFolderName, DestLabelFolderName, DestImageSize, FilePostfixName = '', FilePrefixName=''):
    """
    对图像尺寸进行重新变换，同时变换标注文件信息
    """
    LabelFileTypePostfix = '.txt'
    ImageFileTypePostfix = '.png'
    for rt, dirs, files in os.walk(SrcLabelFolderName):
        for f in files:
            if f.endswith(LabelFileTypePostfix):
                print('1 ', f)
                # if not (len(FilePostfixName)>0 and f.find(FilePostfixName)>-1):
                #     continue
                if not (len(FilePrefixName)>0 and f.find(FilePrefixName)>-1):
                    continue
                # filename
                OneLabelFileName = os.path.join(SrcLabelFolderName, f)
                OneImageFileName = os.path.join(SrcImageFolderName, f.replace(LabelFileTypePostfix, ImageFileTypePostfix))
                OneDestLabelFileName = OneLabelFileName
                OneDestImageFileName = OneImageFileName
                # read image
                print(f)
                OneImageData = cv2.imread(OneImageFileName) # [270, 480, 3] / [540, 960, 3]
                if not OneImageData.shape[0] == DestImageSize[0]:
                    # 图像变换
                    OneImageDataResize = cv2.resize(OneImageData, (DestImageSize[1], DestImageSize[0])) # [540, 960, 3]
                    cv2.imwrite(OneDestImageFileName, OneImageDataResize)
                    # 标注文件变换
                    #   原始标注文件
                    OneLabelInfoSrc = []
                    fp_src = open(OneLabelFileName, 'r')
                    OneLabelInfoLines = fp_src.readlines()
                    for i_OneLabelInfo in OneLabelInfoLines:
                        for i_data in i_OneLabelInfo.split():
                            OneLabelInfoSrc.append(float(i_data))
                    OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                    OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5])
                    fp_src.close()
                    # print(OneLabelInfoSrc)
                    #   变换后标注信息
                    OneLabelInfoDest = copy.deepcopy(OneLabelInfoSrc)
                    ScaleRatio = DestImageSize[0]/OneImageData.shape[0]
                    for i_bbox in range(OneLabelInfoSrc.shape[0]):
                        OneLabelInfoDest[i_bbox, 1:] = ScaleRatio * OneLabelInfoSrc[i_bbox, 1:]
                    #   保存变换后标注文件
                    fp_dest = open(OneDestLabelFileName, 'w')
                    for i_bbox in range(OneLabelInfoDest.shape[0]):
                        CurBboxLine = ''
                        for i_data in range(OneLabelInfoDest[i_bbox].shape[0]):
                            if i_data == 0:
                                CurBboxLine = CurBboxLine + str(int(OneLabelInfoDest[i_bbox][i_data])) + ' '
                            else:
                                CurBboxLine = CurBboxLine + str(round(OneLabelInfoDest[i_bbox][i_data], 4)) + ' '
                        fp_dest.writelines(CurBboxLine + '\n')
                    fp_dest.close()
                    # print(OneLabelInfoDest)
                    
                    """
                    for i_bbox in range(OneLabelInfoSrc.shape[0]):
                        BboxXmin = int(OneImageData.shape[1]*(OneLabelInfoSrc[i_bbox, 1]-OneLabelInfoSrc[i_bbox, 3]/2))
                        BboxYmin = int(OneImageData.shape[0]*(OneLabelInfoSrc[i_bbox, 2]-OneLabelInfoSrc[i_bbox, 4]/2))
                        BboxXmax = int(OneImageData.shape[1]*(OneLabelInfoSrc[i_bbox, 1] + OneLabelInfoSrc[i_bbox, 3]/2))
                        BboxYmax = int(OneImageData.shape[0]*(OneLabelInfoSrc[i_bbox, 2] + OneLabelInfoSrc[i_bbox, 4]/2))
                        cv2.rectangle(OneImageData, (BboxXmin, BboxYmin), (BboxXmax, BboxYmax), (0,255,0), 3)
                    cv2.imshow('src image', OneImageData)
                    cv2.waitKey()
                    
                    for i_bbox in range(OneLabelInfoDest.shape[0]):
                        BboxXmin = int(OneImageData.shape[1]*(OneLabelInfoDest[i_bbox, 1]-OneLabelInfoDest[i_bbox, 3]/2))
                        BboxYmin = int(OneImageData.shape[0]*(OneLabelInfoDest[i_bbox, 2]-OneLabelInfoDest[i_bbox, 4]/2))
                        BboxXmax = int(OneImageData.shape[1]*(OneLabelInfoDest[i_bbox, 1] + OneLabelInfoDest[i_bbox, 3]/2))
                        BboxYmax = int(OneImageData.shape[0]*(OneLabelInfoDest[i_bbox, 2] + OneLabelInfoDest[i_bbox, 4]/2))
                        cv2.rectangle(OneImageDataResize, (BboxXmin, BboxYmin), (BboxXmax, BboxYmax), (0,255,0), 3)
                    cv2.imshow('dest image', OneImageDataResize)
                    cv2.waitKey()
                    """
                
        #         break
        #     break
        # break
                
    return 0

def ResizeImages(SrcFolderName, DestFolderName, DestImageSize, FileSubStrGroup = None):
    """
    resize 图像
    """
    ImageFileTypePostfix = '.png'
    for rt, dirs, files in os.walk(SrcFolderName):
        for f in files:
            if f.endswith(ImageFileTypePostfix):
                SelectImageFlag = False
                for i_Str in FileSubStrGroup:
                    if f.find(i_Str) > -1:
                        SelectImageFlag=True
                        break
                # read images
                if SelectImageFlag == True:
                    print(f)
                    OneImageFileName = os.path.join(SrcFolderName, f)
                    OneDestImageFileName = os.path.join(DestFolderName, f)
                    OneImageData = cv2.imread(OneImageFileName) # [270, 480, 3] / [540, 960, 3]
                    if not OneImageData.shape[0] == DestImageSize[0]:
                        # 图像变换
                        OneImageDataResize = cv2.resize(OneImageData, (DestImageSize[1], DestImageSize[0])) # [540, 960, 3]
                        cv2.imwrite(OneDestImageFileName, OneImageDataResize)

    return 0

def CopySameFiles(SrcFolderName, DestFolderName, FindFolderName, FilePostfixName=None):
    """
    复制相同文件
    """
    for rt, dirs, files in os.walk(SrcFolderName):
        for f in files:
            if not FilePostfixName==None:
                if not f.endswith(FilePostfixName):
                    continue
            SrcFindFileName = os.path.join(FindFolderName, f)
            DestFileName = os.path.join(DestFolderName, f)
            if not os.path.exists(SrcFindFileName):
                print('not exist file: {}'.format(f))
                continue
            copyfile(SrcFindFileName, DestFileName)

    return 0

def CopySemiSameFiles(SrcFolderName, FindFolderName, DestFolderName, FilePostfixGroupName=None, FilePrefixGroupName=None):
    """
    复制同帧不同类型文件
        如：根据image找对应的标注txt文件
    """
    # 数据后缀文件名
    if not FilePostfixGroupName is None:
        SrcFolderNamePostfixName = FilePostfixGroupName[0]
        DestFolderNamePostfixName = FilePostfixGroupName[1]
    else:
        SrcFolderNamePostfixName = ''
        DestFolderNamePostfixName = ''
    for rt, dirs, files in os.walk(SrcFolderName):
        for f in files:
            if len(SrcFolderNamePostfixName)==0:
                SrcFindFileName = os.path.join(FindFolderName, f)
                if not os.path.exists(SrcFindFileName):
                    print('not exist file: {}'.format(f))
                    continue
                DestFileName = os.path.join(DestFolderName, f)
                copyfile(SrcFindFileName, DestFileName)
                    
            if len(SrcFolderNamePostfixName)>0 and f.endswith(SrcFolderNamePostfixName):
                SrcFindFileName = os.path.join(FindFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName))
                # 数据前缀文件名
                if not FilePrefixGroupName is None:
                    SrcFindFileName = os.path.join(FindFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName).replace(FilePrefixGroupName[0], FilePrefixGroupName[1]))
                if not os.path.exists(SrcFindFileName):
                    print('not exist file: {}'.format(f))
                    continue
                DestFileName = os.path.join(DestFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName))
                # 数据前缀文件名
                if not FilePrefixGroupName is None:
                    DestFileName = os.path.join(DestFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName).replace(FilePrefixGroupName[0], FilePrefixGroupName[1]))
                copyfile(SrcFindFileName, DestFileName)
    return 0

def TransFileName(SrcFolderName, DestFolderName, ReplaceNameGroup):
    """
    复制相同文件
    """
    # 
    if not ReplaceNameGroup is None:
        SrcFolderRepleceName = ReplaceNameGroup[0]
        DestFolderRepleceName = ReplaceNameGroup[1]
    else:
        SrcFolderNamePostfixName = ''
        DestFolderNamePostfixName = ''
    # SrcFolderName
    for rt, dirs, files in os.walk(SrcFolderName):
        for f in files:
            SrcFindFileName = os.path.join(SrcFolderName, f)
            DestFileName = os.path.join(DestFolderName, f.replace(SrcFolderRepleceName, DestFolderRepleceName))
            copyfile(SrcFindFileName, DestFileName)

    return 0

# 两类数据转换
class TwoModeDataTransform():
    def __init__(self, SrcImageFolderName, SrcLabelFolderName, DestImageFolderName, DestLabelFolderName, DestImageSize, RotParamFileName):
        self.SrcImageFolderName = SrcImageFolderName
        self.SrcLabelFolderName = SrcLabelFolderName
        self.DestImageFolderName = DestImageFolderName
        self.DestLabelFolderName = DestLabelFolderName
        self.DestImageSize = DestImageSize # [512, 424]
        self.RotParamFileName = RotParamFileName
        self.SrcImagePostfixName = '.png'
        self.SrcLabelPostfixName = '.txt'
        
        if not os.path.exists(self.DestImageFolderName):
            os.makedirs(self.DestImageFolderName)
        if not os.path.exists(self.DestLabelFolderName):
            os.makedirs(self.DestLabelFolderName)
        
    def MultiAnnoFrameTransform(self):
        """
        功能：对多帧数据进行转换，包括：图像转换、label转换
        """
        # 读取转换矩阵文件
        if isinstance(self.RotParamFileName, str) == True: # 单个文件
            RGB2DepthTransformMatrix = ReadTransformFile(self.RotParamFileName)
        else: # 多个文件
            # RGB2DepthParamsFileGroup
            RGB2DepthParamsFileGroup = dict()
            RGB2DepthParamsFileGroup['SensorName'] = []
            RGB2DepthParamsFileGroup['RotParamFileName'] = []
            for i_RGB2DepthParamsFolderName in self.RotParamFileName:
                CurRGB2DepthParamsFolderName = i_RGB2DepthParamsFolderName
                CurFolderSensorName = os.listdir(CurRGB2DepthParamsFolderName)
                for i_SensorName in CurFolderSensorName:
                    CurFolderSensorCalibName = os.path.join(CurRGB2DepthParamsFolderName, i_SensorName, 'rot_param.txt')
                    RGB2DepthParamsFileGroup['SensorName'].append(i_SensorName)
                    RGB2DepthParamsFileGroup['RotParamFileName'].append(CurFolderSensorCalibName)
                
        # 目标文件大小
        DestImgWidth = self.DestImageSize[0]
        DestImgHeight = self.DestImageSize[1]
        # 遍历图像
        for root, dirs, files in os.walk(self.SrcLabelFolderName):
            for filename in files:
                if filename.endswith(self.SrcLabelPostfixName):
                    CurSrcImageFileName = os.path.join(self.SrcImageFolderName, filename.replace(self.SrcLabelPostfixName, self.SrcImagePostfixName))
                    CurSrcLabelFileName = os.path.join(self.SrcLabelFolderName, filename)
                    CurDestImageFileName = os.path.join(self.DestImageFolderName, filename.replace(self.SrcLabelPostfixName, self.SrcImagePostfixName))
                    CurDestLabelFileName = os.path.join(self.DestLabelFolderName, filename)
                    
                    # 读取转换矩阵文件
                    if isinstance(self.RotParamFileName, str) == True: # 单个文件
                        RGB2DepthTransformMatrix = RGB2DepthTransformMatrix
                    else:
                        CurSensorName = ''
                        CurFileNameSplit = filename.split('_')
                        if len(CurFileNameSplit[0].split('-')) == 1:
                            if int(CurFileNameSplit[0])>1000:
                                CurSensorName = CurFileNameSplit[0] + '_' + CurFileNameSplit[-1].split('.')[0]
                            else:
                                CurSensorName = CurFileNameSplit[0]
                        else:
                            CurSensorName = ''
                        CurRoomSensorName = CurSensorName
                        if CurRoomSensorName in RGB2DepthParamsFileGroup['SensorName']:
                            CurSensorNameIdx = RGB2DepthParamsFileGroup['SensorName'].index(CurRoomSensorName)
                            RGB2DepthTransformMatrix = ReadTransformFile(RGB2DepthParamsFileGroup['RotParamFileName'][CurSensorNameIdx])
                            RGB2DepthTransformMatrix[2,0] = 0
                            RGB2DepthTransformMatrix[2,1] = 0
                        else:
                            print('sensor name = {} error'.format(CurRoomSensorName))
                    
                    # 图像转换
                    #    读取图像
                    CurSrcImageData = cv2.imread(CurSrcImageFileName) # [540, 960, 3]
                    CurSrcImageWidth = CurSrcImageData.shape[1]
                    CurSrcImageHeight = CurSrcImageData.shape[0]
                    #     AnnoImageTransform
                    CurDestImageData = TwoModeDataTransform.AnnoImageTransform(CurSrcImageData, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
                    #    保存图像
                    cv2.imwrite(CurDestImageFileName, CurDestImageData)
                    
                    # label转换
                    #    读取label
                    OneLabelInfoSrc = []
                    fp_src = open(CurSrcLabelFileName, 'r')
                    OneLabelInfoLines = fp_src.readlines()
                    for i_OneLabelInfo in OneLabelInfoLines:
                        for i_data in i_OneLabelInfo.split():
                            OneLabelInfoSrc.append(float(i_data))
                    OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                    OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5])
                    fp_src.close()
                    
                    #    AnnoLabelTransform
                    # print('OneLabelInfoSrc = ', OneLabelInfoSrc)
                    CurDestLabelData = TwoModeDataTransform.AnnoLabelTransform(OneLabelInfoSrc, RGB2DepthTransformMatrix, CurSrcImageWidth, CurSrcImageHeight, DestImgWidth, DestImgHeight)
                    # print('CurDestLabelData = ', CurDestLabelData)
                    
                    #    保存label
                    OneLabelInfoDest = CurDestLabelData
                    fp_dest = open(CurDestLabelFileName, 'w')
                    for i_bbox in range(OneLabelInfoDest.shape[0]):
                        CurBboxLine = ''
                        for i_data in range(OneLabelInfoDest[i_bbox].shape[0]):
                            if i_data == 0:
                                CurBboxLine = CurBboxLine + str(int(OneLabelInfoDest[i_bbox][i_data])) + ' '
                            else:
                                CurBboxLine = CurBboxLine + str(round(OneLabelInfoDest[i_bbox][i_data], 4)) + ' '
                        fp_dest.writelines(CurBboxLine + '\n')
                    fp_dest.close()
                    
            #         break
            #     break
            # break
                    

        return 0
    
    # @classmethod
    def AnnoImageTransform(SrcImage, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight):
        """
        功能：图像转换
        """
        MethodFlag = 1 # MethodFlag=1,使用cv2.warpAffine
                       # MethodFlag=2,使用遍历点，矩阵相乘
        if MethodFlag == 1:
            # RGB2DepthHCV = RGB2DepthTransformMatrix.transpose()[:2,:]
            # RGB2DepthHCV[0,-1] = RGB2DepthTransformMatrix[1,-1]
            # RGB2DepthHCV[1,-1] = RGB2DepthTransformMatrix[0,-1]
            # DestImage = cv2.warpAffine(SrcImage, RGB2DepthHCV, (DestImgWidth, DestImgHeight)) # matrix=[2,3], DestImage=[424,512,3]
        
            RGB2DepthHCV = copy.deepcopy(RGB2DepthTransformMatrix) # deepcopy
            RGB2DepthHCV = RGB2DepthHCV.transpose()[:2,:]
            RGB2DepthHCV[0,-1] = RGB2DepthTransformMatrix[1,-1]
            RGB2DepthHCV[1,-1] = RGB2DepthTransformMatrix[0,-1]
            RGB2DepthHCV[0,0] = RGB2DepthTransformMatrix[1,1]
            RGB2DepthHCV[1,1] = RGB2DepthTransformMatrix[0,0]
            DestImage = cv2.warpAffine(SrcImage, RGB2DepthHCV, (DestImgWidth, DestImgHeight)) # matrix=[2,3], DestImage=[424,512,3]

        
        elif MethodFlag == 2:
            SrcImgWidth = SrcImage.shape[1]
            SrcImgHeight = SrcImage.shape[0]
            RGB2DepthTransformMatrixInv = np.linalg.inv(RGB2DepthTransformMatrix)
            DestImage = np.zeros([DestImgHeight, DestImgWidth, 3])
            for i_h in range(DestImgHeight):
                for i_w in range(DestImgWidth):
                    CurPoints = np.array([[i_h], [i_w], [1]])
                    NewPoints = np.round(np.dot(RGB2DepthTransformMatrixInv, CurPoints))
                    if NewPoints[0][0] < 0 or NewPoints[0][0] >= SrcImgHeight:
                        continue
                    if NewPoints[1][0] < 0 or NewPoints[1][0] >= SrcImgWidth:
                        continue
                    DestImage[i_h, i_w, 0] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 0]
                    DestImage[i_h, i_w, 1] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 1]
                    DestImage[i_h, i_w, 2] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 2]

        return DestImage
    
    def AnnoLabelTransform(SrcAnnoInfo, RGB2DepthTransformMatrix, SrcImgWidth, SrcImgHeight, DestImgWidth, DestImgHeight):
        """
        功能：label 转换
        """
        SrcBboxes = np.zeros([SrcAnnoInfo.shape[0], 4])
        for i_obj in range(SrcAnnoInfo.shape[0]):
             CurObjxxyy = TwoModeDataTransform.convert_xywh2xxyy([SrcImgWidth, SrcImgHeight], SrcAnnoInfo[i_obj,1:])
             SrcBboxes[i_obj, 0] = CurObjxxyy[1]
             SrcBboxes[i_obj, 1] = CurObjxxyy[0]
             SrcBboxes[i_obj, 2] = CurObjxxyy[3]
             SrcBboxes[i_obj, 3] = CurObjxxyy[2]
        SrcScores = np.zeros([SrcAnnoInfo.shape[0]])
        SrcLabels = SrcAnnoInfo[:,0]
        # TransformBBoxInfo, SrcBboxes=[y1, x1, y2, x2]
        DestBboxes, DestScores, DestLabels = TransformBBoxInfo(SrcBboxes, SrcScores, SrcLabels, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
        DestObjNum = np.array(DestBboxes).shape[0]
        if DestObjNum>0:
            DestAnnoObjValidIndex = []
            DestAnnoInfo = np.zeros([DestObjNum, SrcAnnoInfo.shape[1]])
            DestAnnoInfo[:,0]=np.array(DestLabels)
            # DestAnnoInfo[:,1:]=np.array(DestBboxes)
            DestBboxes = np.array(DestBboxes)
            for i_obj in range(DestBboxes.shape[0]):
                # 排除部分边界目标，SrcBboxes=[y1, x1, y2, x2]
                if DestBboxes[i_obj,1]>(DestImgWidth-30):
                    continue
                if DestBboxes[i_obj,3]<30:
                    continue
                DestAnnoObjValidIndex.append(i_obj) # 有效目标序号
                # convert_xxyy2xywh
                CurObjxxyy = [DestBboxes[i_obj,1], DestBboxes[i_obj,3], DestBboxes[i_obj,0], DestBboxes[i_obj,2]]
                CurObjxywh = TwoModeDataTransform.convert_xxyy2xywh([DestImgWidth, DestImgHeight], CurObjxxyy)
                DestAnnoInfo[i_obj,1] = CurObjxywh[0]
                DestAnnoInfo[i_obj,2] = CurObjxywh[1]
                DestAnnoInfo[i_obj,3] = CurObjxywh[2]
                DestAnnoInfo[i_obj,4] = CurObjxywh[3]
            # 有效目标
            DestAnnoInfo = DestAnnoInfo[DestAnnoObjValidIndex,:]
        else:
            DestAnnoInfo = np.empty([0, SrcAnnoInfo.shape[1]])
        return DestAnnoInfo
    
    def convert_xxyy2xywh(size, box):
        """
        功能：转换bbox 坐标，[x1,x2,y1,y2]-->[xc,yc,w,h]
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 + 2
        y = (box[2] + box[3]) / 2.0 + 2
        w = box[1] - box[0] + 2 # 手动增加两个像素点
        h = box[3] - box[2] + 2 # 手动增加两个像素点
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
    
    def convert_xywh2xxyy(size, box):
        """
        功能：转换bbox 坐标，[xc,yc,w,h]-->[x1,y1,x2,y2]
        """
        x1 = (box[0] - box[2]/2)*size[0]
        x2 = (box[0] + box[2]/2)*size[0]
        
        y1 = (box[1] - box[3]/2)*size[1]
        y2 = (box[1] + box[3]/2)*size[1]
        return (x1, y1, x2, y2)


if __name__ == '__main__':
    TestCase = 1 # 测试函数：AnnoImageTransform
    
    if TestCase == 1:
        FileName = r'D:\xiongbiao\Data\HumanDete\NewLG\PlotDatasetAnnoInfo\CalibRGB2Depth_0621\RGB\167_Color2020-06-24-173000_02250.png'
        RGB2DepthTransformMatrix = np.array([[ 7.0068e-01,  1.4911e-02,  1.1105e+01],
                               [-2.8385e-02,  6.7502e-01, -6.5910e+01],
                               [0,  0,  1.0000e+00]])
        
        SrcImage = cv2.imread(FileName)
        SrcImgWidth = SrcImage.shape[1]
        SrcImgHeight = SrcImage.shape[0]
        DestImgWidth = 512
        DestImgHeight = 424
        
        if 0:
            RGB2DepthHCV = RGB2DepthTransformMatrix.transpose()[:2,:]
            RGB2DepthHCV[0,-1] = RGB2DepthTransformMatrix[1,-1]
            RGB2DepthHCV[1,-1] = RGB2DepthTransformMatrix[0,-1]   
            DestImage = cv2.warpAffine(SrcImage, RGB2DepthHCV, (DestImgWidth, DestImgHeight)) # matrix=[2,3], DestImage=[424,512,3]
            cv2.imwrite('test_trans_image_1.png', DestImage)
            
        if 1:
            import time
            
            RGB2DepthTransformMatrixInv = np.linalg.inv(RGB2DepthTransformMatrix)
            DestImage = np.zeros([424,512,3])
            t1 = time.time()
            for i_h in range(DestImgHeight):
                for i_w in range(DestImgWidth):
                    CurPoints = np.array([[i_h], [i_w], [1]])
                    NewPoints = np.round(np.dot(RGB2DepthTransformMatrixInv, CurPoints))
                    if NewPoints[0][0] < 0 or NewPoints[0][0] >= SrcImgHeight:
                        continue
                    if NewPoints[1][0] < 0 or NewPoints[1][0] >= SrcImgWidth:
                        continue
                    DestImage[i_h, i_w, 0] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 0]
                    DestImage[i_h, i_w, 1] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 1]
                    DestImage[i_h, i_w, 2] = SrcImage[int(NewPoints[0][0]), int(NewPoints[1][0]), 2]
            print('time = ', (time.time()-t1)) # 2.8774 s
                    
            cv2.imwrite('test_trans_image_2.png', DestImage)

