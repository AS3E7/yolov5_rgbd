# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:44:58 2020

@author: HYD
"""
import numpy as np
import cv2
from util import CloudPointFuns

# 深度数据图像转换参数
ImageTransParams = dict()
ImageTransParams['DepthMin'] = 2000 # 按传感器距离值，进行伪彩色配色
ImageTransParams['DepthMax'] = 5000
ImageTransParams['HeightMin'] = -250 # 深度数据配准，去数据Z方向高度值，进行伪彩色配色
ImageTransParams['HeightMax'] =  2500

def DepthPreproces(Img0, Pts0, KintWidth, KintHeight, H, ImageTransMethod, FilpFlag = 0):
    """
    功能：深度数据预处理
    输入：
        ImageTransMethod：图像转换方法
                         # ImageTransMethod == 0, 输入深度，不进行变换
                         # ImageTransMethod == 1, 按传感器距离值，进行伪彩色配色
                         # ImageTransMethod == 2, 深度数据配准，去数据Z方向高度值，进行伪彩色配色
    """
    
#    FilpFlag = SensorInfo.isFlip # SensorInfo->isFlip
    
    # preprocess image data
    PrepValidFlag = 0 # Find Valid Points
    PrepDepth2ImgFlag = 1 # Depth to Image
    
    if ImageTransMethod == 0: # 输入深度，不进行变换
        Img0 = Img0
    elif ImageTransMethod == 1: # 按传感器距离值，进行伪彩色配色
        Img0 = Img0/32
        ThodIdx = (Img0>round(8000/32))
        Img0[ThodIdx] = round(8000/32) 
        
        # 选择一定距离范围内数据颜色转换
        if True: # True / False
            # 设置有效数据点为：2000mm - 5000mm
            Img0[Img0>round(ImageTransParams['DepthMax']/32)] = round(ImageTransParams['DepthMax']/32)
            Img0[Img0<round(ImageTransParams['DepthMin']/32)] = round(ImageTransParams['DepthMin']/32)
            Img0 = np.round((Img0 - ImageTransParams['DepthMin']/32)/(ImageTransParams['DepthMax']/32 - ImageTransParams['DepthMin']/32)*255) + 1
    elif ImageTransMethod == 2: # 深度数据配准，去数据Z方向高度值，进行伪彩色配色
#        Img0 = ImageTrans_CalibHeight(Pts0, H, ImageTransParams['HeightMin'], ImageTransParams['HeightMax'])
        Img0 = ImageTrans_CalibHeight(Pts0, H, -ImageTransParams['HeightMax'], -ImageTransParams['HeightMin'])

                 
    # read image data, DepthImg=[424 x 512]
    DepthImg = np.reshape(Img0, (KintHeight, KintWidth))
    DepthToCloudPointIdx = range(KintHeight * KintWidth)
    DepthToCloudPointIdx = np.reshape(DepthToCloudPointIdx, (KintHeight, KintWidth))
    
    if PrepValidFlag == 1:
        FilterWinSize = 5
        hf = np.ones([FilterWinSize, FilterWinSize])
        hf[int(np.floor(FilterWinSize / 2)), int(np.floor(FilterWinSize / 2))] = 0
        ImgFilter = cv2.filter2D(DepthImg,-1,hf)
        DepthImg = ImgFilter
        NoisePos = (ImgFilter == 0)
        NoisePos = NoisePos.flatten('F')
        Pts0[:, NoisePos] = 0
    if PrepDepth2ImgFlag == 1:
        ImgFilpFlag = FilpFlag
        if ImgFilpFlag == 1: # flipud
            DepthImg = np.flipud(DepthImg)
            DepthToCloudPointIdx = np.flipud(DepthToCloudPointIdx)
            
#        # depth convert to RGB, using clormap to R,G,B channels (jet colorbar)
        DepthImgColor1 = DepthImg.astype(np.uint8)
        DepthImgColor2 = cv2.applyColorMap(DepthImgColor1, cv2.COLORMAP_JET) # [424, 512, 3]
        DepthImgColor = DepthImgColor2.transpose(2, 0, 1) # [3, 424, 512]
        
    else:
        DepthImgColor = DepthImg
    
    return DepthImgColor, Pts0, DepthToCloudPointIdx
    
def ImageTrans_CalibHeight(Pts, H, HeightMin, HeightMax):
    """
    功能：深度数据配准，去数据Z方向高度值，进行伪彩色配色
    """
    # NewPts
    NewPts = CloudPointFuns.Rot3D(H, Pts)# [3 x N]
    NewPts_Z = NewPts[-1,:]
    NewPts_Z = -NewPts_Z # NewPts_Z = H[2,3] - NewPts_Z
    NewPts_Z = NewPts_Z * 1000 # new pts height, [unit: mm]
    
#    CloudPointFuns.SavePt3D2Ply('newpts.ply', NewPts.transpose(), 'XYZ')

    # [HeightMin, HeightMax]
    DesDepthData = NewPts_Z
    SrcDepthData = NewPts_Z
    LowerDepthIdx = (SrcDepthData < HeightMin)
    DesDepthData[LowerDepthIdx] = HeightMin
    UpperDepthIdx = (SrcDepthData > HeightMax)
    DesDepthData[UpperDepthIdx] = HeightMax
    
    # 转换至图片数据范围 [0 - 255]
    ColorRange = [0, 255]
    DesDepthData = (DesDepthData - HeightMin)/((HeightMax-HeightMin)/(ColorRange[1]-ColorRange[0]))

    return DesDepthData
    
def RGBPreproces(RGBData):
    """
    功能：深度数据预处理
    输入：
        DepthData
    """
    print('RGBPreproces')
    
    return 0
    
    
    
    