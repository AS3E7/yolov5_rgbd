# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:03:17 2020

@author: HYD
"""

import numpy as np
import copy
import cv2
import os

def CalBGInfo(Pts, CalMethod = 1, DepthWidth = 512, DepthHeight = 424, StdSet = 0.13):
    """
    功能：计算数据的背景信息
    """
    # if CalMethod == 1, 使用一帧数据作为背景，单数数据作为均值
    # if CalMethod == 2, 使用多帧数据计算背景，平均数/均值方法
#    StdSet = 0.13 # init:0.05
    
    BGInfo = []
    if CalMethod == 1:
        if Pts.shape[0]>0:
            # BGInfo = [4,217088]
            CurPtsStd = StdSet * np.ones([1, Pts.shape[1]]) # std
            BGInfo = np.concatenate((Pts, CurPtsStd))     
    return BGInfo
    
def CalImageSubBG(SrcPts, BGInfo, CalMethod = 1, DepthWidth = 512, DepthHeight = 424):
    """
    功能：计算去除背景后的数据
    输入：
        SrcPts：输入原始点云数据，[3 x N]
        BGInfo：输入原始点云数据，[4 x N]
    输出：
        DestPts：输出前景点云数据
    """
    # if CalMethod == 1, 平均数/均值方法
    
    DestPts = copy.copy(SrcPts)
    FGDistK = 1 # 距离方差系数
    if CalMethod == 1:
        DestPts = copy.copy(SrcPts)
        InValidPtsIdx = (abs(np.sqrt((SrcPts[0,:] - BGInfo[0,:])**2 + (SrcPts[1,:] - BGInfo[1,:])**2 + (SrcPts[2,:] - BGInfo[2,:])**2)) < FGDistK*BGInfo[-1,:])
        DestPts[:,InValidPtsIdx] = 0
            
    return DestPts
    
def CalFGSubNoise(SrcPts, DepthWidth = 512, DepthHeight = 424):
    """
    功能：去背景之后的数据进行去噪处理，去除深度数据边界无效数据点
    输入：
        SrcPts：[3 x N]
    """
    # SrcPts-Z
    FGSrcPts_Z = SrcPts[2,:]
    FGSrcPts_Z = 1000*FGSrcPts_Z/32
    DepthImage = np.reshape(FGSrcPts_Z, [DepthHeight, DepthWidth])
    
#    # 形态学处理
#    kernel = np.ones((3,3),np.uint8)
#    erosion = cv2.erode(DepthImage,kernel) # 腐蚀
#    kernel = np.ones((6,6),np.uint8)
#    dst = cv2.dilate(erosion,kernel) # 膨胀
#    ImgFilter = dst

    kernel = np.ones((4,4),np.uint8)
    erosion = cv2.erode(DepthImage,kernel) # 腐蚀
    kernel = np.ones((6,6),np.uint8)
    dst = cv2.dilate(erosion,kernel) # 膨胀
    ImgFilter = dst
    
    # SrcPtsValid
    DestPts = copy.copy(SrcPts)
    NoisePos = (ImgFilter == 0)
#    NoisePos = NoisePos.flatten('F') # 按列排序
    NoisePos = NoisePos.flatten('A') # 按行排序
    DestPts[:, NoisePos] = 0
    
#    cv2.imshow('image1', DepthImage)
#    cv2.imshow('image2', ImgFilter)
    
    return DestPts
    
    
if __name__ == '__main__':
    print('Start BGProcess.')
    import os, cv2
    from CloudPointFuns import ReadKinect2FromPly, SavePt3D2Ply
    
    CurDir = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete'
    BGInfoOutlierDistLimit = 0.13
    DepthWidth = 512
    DepthHeight = 424
    BGSrcPts = ReadKinect2FromPly(os.path.join(CurDir, 'log', 'bg.ply')) # [3 x N]
    BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit) # [4 x 217088]
    
    print(BGSrcPts.shape, BGInfo.shape)
    
    SelectPlyFileName = r'X:\PlyData\LGKSS\C_102_2\167\2020-07-01-134000\Depth2020-07-01-134000_1593582423680_01540.ply'
    SelectSrcPts = ReadKinect2FromPly(SelectPlyFileName)
    
    FGSrcPts = CalImageSubBG(SelectSrcPts, BGInfo)
    print(FGSrcPts.shape)
    SavePt3D2Ply('temp_fg.ply', FGSrcPts.transpose(), 'XYZ')
    
    FGSrcPts_Z = FGSrcPts[2,:]
    FGSrcPts_Z = 1000*FGSrcPts_Z/32
    DepthImage = np.reshape(FGSrcPts_Z, [DepthHeight, DepthWidth])

    DepthImg = DepthImage
    print(DepthImage.shape)
    cv2.imshow('image1', DepthImg)
    
    if 1:
        FilterWinSize = 5
#        hf = np.ones([FilterWinSize, FilterWinSize])
#        hf[int(np.floor(FilterWinSize / 2)), int(np.floor(FilterWinSize / 2))] = 0
#        ImgFilter = cv2.filter2D(DepthImg,-1,hf)
        

#        kernel = np.ones((2,2),np.uint8)
#        erosion = cv2.erode(DepthImg,kernel)
#        kernel = np.ones((3,3),np.uint8)
#        dst = cv2.dilate(erosion,kernel)
#        ImgFilter = dst
        
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(DepthImg,kernel)
        kernel = np.ones((6,6),np.uint8)
        dst = cv2.dilate(erosion,kernel)
        ImgFilter = dst


        DepthImg2 = ImgFilter
        NoisePos = (ImgFilter == 0)
        NoisePos = NoisePos.flatten('F')
        FGSrcPts[:, NoisePos] = 0
    
    SavePt3D2Ply('temp.ply', FGSrcPts.transpose(), 'XYZ')
    
    cv2.imshow('image2', DepthImg2)
    
    