# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 20:33:16 2020

@author: Administrator
"""
import numpy as np
import os
import copy
import cv2

def TransformRGBBBoxInfoToDepth(SrcBboxInfo, SrcScoreInfo, SrcLabelInfo, RotParamFileName, DestImgWidth, DestImgHeight):
    """
    变换RGB-Depth坐标信息
    输入：
        SrcBboxInfo：Bbox 坐标格式，（y1, x1, y2, x2）
    """

    # read rot param filename
    RGB2DepthTransformMatrix = ReadTransformFile(RotParamFileName)
    # trans bbox info
    DestBboxInfo, DestScoreInfo, DestLabelInfo = TransformBBoxInfo(SrcBboxInfo, SrcScoreInfo, SrcLabelInfo, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
    
    return DestBboxInfo, DestScoreInfo, DestLabelInfo


def TransformBBoxInfo(SrcBboxes, SrcScores, SrcLabels, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight):
    """
    功能：BBox位置转换
        SrcBbox : [xmin. ymin, xmax, ymax]
    """
#    print('SrcBboxes = {}'.format(SrcBboxes))
    DestBboxes = []
    DestScores = []
    DestLabels = []
    for i_obj, SrcBbox in enumerate(SrcBboxes):
        SrcPoint = np.array([[SrcBbox[0], SrcBbox[1]], [SrcBbox[2], SrcBbox[3]]])
        DestPoint = []
        for i_Point in SrcPoint:
            TempCurPoint = ComputeTwoImageAxisPixel2Pixel(i_Point, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
#            if TempCurPoint[0]>-1 and TempCurPoint[1]>-1:
#                DestPoint.append(TempCurPoint)
#            print('i_Point = ', i_Point, TempCurPoint)

            if not (TempCurPoint[0]<0 and TempCurPoint[1]<0): # 修改边界目标框的影响
                TempCurPoint[0] = max(TempCurPoint[0], 0)
                TempCurPoint[0] = min(TempCurPoint[0], DestImgHeight)
                TempCurPoint[1] = max(TempCurPoint[1], 0)
                TempCurPoint[1] = min(TempCurPoint[1], DestImgWidth)
                
                # TempCurPoint[0] = max(TempCurPoint[0], 0)
                # TempCurPoint[0] = min(TempCurPoint[0], DestImgWidth)
                # TempCurPoint[1] = max(TempCurPoint[1], 0)
                # TempCurPoint[1] = min(TempCurPoint[1], DestImgHeight)
                
                DestPoint.append(TempCurPoint)
        # print('DestPoint = ', DestPoint)
        # DestBbox
        if len(DestPoint)==2: # 存在目标框两个角点
            DestBbox = [DestPoint[0][0], DestPoint[0][1], DestPoint[1][0], DestPoint[1][1]]
            #去除边界无效目标数据
            CurBboxMinPixel = 2 
            if (abs(DestBbox[2]-DestBbox[0])<CurBboxMinPixel) or (abs(DestBbox[3]-DestBbox[1])<CurBboxMinPixel): 
                continue
            # 保存结果
            DestBboxes.append(DestBbox)
            DestScores.append(SrcScores[i_obj])
            DestLabels.append(SrcLabels[i_obj])
#    print('DestBboxes = {}'.format(DestBboxes))
    
    return DestBboxes, DestScores, DestLabels

def ReadTransformFile(RotParamFileName):
    """
    读取RGB-Depth变换文件，获取变换参数
    """
    CurSenorRotParamFileName = RotParamFileName
    CurSenorRotParam = []
    fp = open(CurSenorRotParamFileName)
    while True:
        line = fp.readline()
        line = line.strip().split(',')
        if len(line) > 1:
            for p_num in line:
                CurSenorRotParam.append(float(p_num))
        else:
            break
    fp.close()
    RGB2DepthTransformMatrix = np.reshape(np.array(CurSenorRotParam), [3,3])
    
    return RGB2DepthTransformMatrix

def ComputeTwoImageAxisPixel2Pixel(SrcPixel, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight):
    """
    Function: 计算两个图像中某个点的对应关系
    """
   # 初始化目标点坐标
    DestPixel = [-1, -1]
    
    # 计算点对应关系
    x0 = SrcPixel[0]
    y0 = SrcPixel[1]
    
#    x0 = SrcPixel[1]
#    y0 = SrcPixel[0]
    
    TempSrcPts = np.array([[x0],[y0],[1]])
    TempDestPts = np.dot(RGB2DepthTransformMatrix, TempSrcPts)
    
    new_x0 = int(TempDestPts[0])
    new_y0 = int(TempDestPts[1])
    # new_x0 = TempDestPts[0]
    # new_y0 = TempDestPts[1]

#    new_x0 = int(TempDestPts[1])
#    new_y0 = int(TempDestPts[0])
    
    # 限制数据范围
#            if (new_x0>-1) and (new_x0<TwoImageRelationParam.DestImgWidth) and (new_y0>-1) and (new_y0<TwoImageRelationParam.DestImgHeight):
    if not (((new_x0<0) or (new_x0>DestImgWidth)) and ((new_y0<0) or (new_y0>DestImgHeight))):

        DestPixel[0] = new_x0
        DestPixel[1] = new_y0
    else:
        DestPixel[0] = -1
        DestPixel[1] = -1
    
    return  DestPixel   
    

if __name__ == '__main__':
    import time
    print('RGB2Depth.')
    TestRGB2DepthFlag = 2
    
    if TestRGB2DepthFlag == 1: # 测试 bbox 坐标变换
        # filename
        RotParamFileName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD\data\rgb2depth_param.txt'
        # image info
        DestImgWidth = 512
        DestImgHeight = 424
        # SrcBboxInfo
#        SrcBboxInfo = np.array([[1,2,3,4],[10,20,30,40]]) # [x1, y1, x2, y2]
#        SrcScoreInfo = np.array([0.9,0.8]) # 
#        SrcLabelInfo = np.array([0,1]) # 
        
        SrcBboxInfo = np.array([[         63,         180,          94 ,        209]]) # [x1, y1, x2, y2]
        SrcScoreInfo = np.array([ 0.82757]) # 
        SrcLabelInfo = np.array([ 0]) # 
        # [[110, 169, 152, 208]] [0.82757] [0]
        
         # TransformRGBBBoxInfoToDepth
        DestBboxes, DestScores, DestLabels = TransformRGBBBoxInfoToDepth(SrcBboxInfo, SrcScoreInfo, SrcLabelInfo, RotParamFileName, DestImgWidth, DestImgHeight)
        print(DestBboxes, DestScores, DestLabels)
    
    elif TestRGB2DepthFlag == 2: # 测试整个图像坐标变换
        # filename
        RotParamFileName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD\data\rgb2depth_param.txt'
        # RotParamFileName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD\data\rgb2depth_param_540.txt'


        RGB2DepthTransformMatrix = ReadTransformFile(RotParamFileName)
        # image info
        DestImgWidth = 512
        DestImgHeight = 424
        # image data
        SrcImage = np.random.random([540,960,3]) # [540,960,3]
        DestImage = np.zeros([424,512,3]) # [424,512,3]
        # 转换图像坐标
        DestImagePixelIndex = np.zeros([424,512])
        t1 = time.time()
        
        if 0:
            for i in range(DestImgWidth):
                for j in range(DestImgHeight):
                    i_Point = np.array([i, j])
                    TempCurPoint = ComputeTwoImageAxisPixel2Pixel(i_Point, RGB2DepthTransformMatrix, DestImgWidth, DestImgHeight)
                    
                    if not (TempCurPoint[0]<0 and TempCurPoint[1]<0): # 修改边界目标框的影响
                        TempCurPoint[0] = max(TempCurPoint[0], 0)
                        TempCurPoint[0] = min(TempCurPoint[0], DestImgWidth)
                        TempCurPoint[1] = max(TempCurPoint[1], 0)
                        TempCurPoint[1] = min(TempCurPoint[1], DestImgHeight)
                    
                    # print('TempCurPoint = ', TempCurPoint)
                    
        if 0:
            DestImagePixelFlattenIndex = np.ones([3, DestImgWidth*DestImgHeight])
            DestImagePixelFlattenAllList = [ii for ii in range(DestImgWidth*DestImgHeight)]
            DestImagePixelFlattenX = [ii%DestImgWidth for ii in DestImagePixelFlattenAllList]
            DestImagePixelFlattenY = [ii/DestImgWidth for ii in DestImagePixelFlattenAllList]
            DestImagePixelFlattenIndex[0,:] = DestImagePixelFlattenX
            DestImagePixelFlattenIndex[1,:] = DestImagePixelFlattenY
            
            DestImagePixelFlattenIndexTrans = np.dot(RGB2DepthTransformMatrix, DestImagePixelFlattenIndex)
            DestImagePixelFlattenIndexTrans = np.reshape(DestImagePixelFlattenIndexTrans, [3, DestImgHeight, DestImgWidth])   
            DestImagePixelFlattenIndexTrans = np.uint8(DestImagePixelFlattenIndexTrans)
            
        if 1:
            SelectRGBFileName = r'D:\xiongbiao\Code\LGPoseDete\Temp\101_Depth2020-05-27-111633_00070_RGB.png'
            SelectDepthFileName = r'D:\xiongbiao\Code\LGPoseDete\Temp\101_Depth2020-05-27-111633_00070.png'
            
            SrcImage = cv2.imread(SelectRGBFileName) # [540,960,3]
            
            RGB2DepthHCV = RGB2DepthTransformMatrix.transpose()[:2,:]
            RGB2DepthHCV[0,-1] = RGB2DepthTransformMatrix[1,-1]
            RGB2DepthHCV[1,-1] = RGB2DepthTransformMatrix[0,-1]
            
            DestImage = cv2.warpAffine(SrcImage,RGB2DepthHCV,(DestImgWidth, DestImgHeight)) # matrix=[2,3], DestImage=[424,512,3]
            
            cv2.imshow('trans image', DestImage)
            cv2.waitKey()
            
    
        t2 = time.time()
        print('trans rgb2depth time = {}'.format(t2-t1))
    
    