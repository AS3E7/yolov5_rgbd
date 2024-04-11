# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:46:51 2020

@author: HYD
"""

import numpy as np

def SelectDeteMode(RGBData, RGBWidth, RGBHeight, DepthData, DepthWidth, DepthHeight, WorldTime):
    """
    功能：选择检测模型(选择是否使用RGB/Colormap检测模型)
    输出：选择哪种数据，'rgb'/'colormap'
    """
    SelectDeteModeName = 'colormap'
#    print(RGBData.shape, RGBWidth, RGBHeight)
#    print(DepthData.shape, DepthWidth, DepthHeight)
    
    # 选择中间部分数据判断有效性
    InValidPtsRate = 0.425 # init:0.5
    DisBorderTemp = 5
    DepthImageData = np.reshape(DepthData, [DepthHeight, DepthWidth])
    CurSelectDepthData = DepthImageData[int(DepthHeight/DisBorderTemp):int(DepthHeight-DepthHeight/DisBorderTemp), int(DepthWidth/DisBorderTemp):int(DepthWidth-DepthWidth/DisBorderTemp)]
    CurSelectDepthDataInvalidPtsIdx = (CurSelectDepthData==0)
    CurSelectDepthDataInvalidPts = CurSelectDepthData[CurSelectDepthDataInvalidPtsIdx]
    CurSelectDepthDataInvalidPtsRate = CurSelectDepthDataInvalidPts.shape[0]/CurSelectDepthData.flatten().shape[0]
    if CurSelectDepthDataInvalidPtsRate > InValidPtsRate:
        SelectDeteModeName = 'rgb'
    
    return SelectDeteModeName

if __name__ == '__main__':
    print('Start.')
    import os
    import cv2
    from Depth2Pts import ReadOneFrameDepthFile
    
    DepthWidth = 512
    DepthHeight = 424
    
#    SelectFrameName = '-2020-09-04-103926_10004'
#    SelectFrameName = '-2020-09-06-122748_10004'

#    SelectFrameName = '2020-07-10-130000_1594357212173_00060'
    SelectFrameName = '2020-07-10-153000_1594366211828_00060'

    SelectDepthImageFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_Data\LGOld_stronglight_test_image'
    SelectdepthFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_Data\LGOld_stronglight_test_image'
    SelectRGBImageFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_Data\LGOld_stronglight_test_image'
    
    SelectDepthImageFileName = os.path.join(SelectDepthImageFolderName, 'Depth' + SelectFrameName + '.jpg')
    SelectRGBImageFileName = os.path.join(SelectRGBImageFolderName, 'Color' + SelectFrameName + '.jpg')
#    SelectdepthFileName = os.path.join(SelectdepthFolderName, 'Depth' + SelectFrameName + '.jpg.dep')
#    SelectdepthFileName = os.path.join(SelectdepthFolderName, 'Depth' + SelectFrameName + '.depth')
    
    SelectdepthFileName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\NewLG_Bbox_Data\Select_Images\AllFrameTest\MultiClass_C_102_20200710\depth_20200718\164_Depth2020-07-10-131000_02795.depth'
        
    # read depth
    CurDepthData = ReadOneFrameDepthFile(SelectdepthFileName, DepthWidth, DepthHeight)
    CurDepthData = CurDepthData
    print(CurDepthData.shape)
    cv2.imshow('test.png', CurDepthData)
    
    # 选择中间部分数据判断有效性
    DisBorderTemp = 5
    CurSelectDepthData = CurDepthData[int(DepthHeight/DisBorderTemp):int(DepthHeight-DepthHeight/DisBorderTemp), int(DepthWidth/DisBorderTemp):int(DepthWidth-DepthWidth/DisBorderTemp)]
    CurSelectDepthDataInvalidPtsIdx = (CurSelectDepthData==0)
    CurSelectDepthDataInvalidPts = CurSelectDepthData[CurSelectDepthDataInvalidPtsIdx]
    CurSelectDepthDataInvalidPtsRate = CurSelectDepthDataInvalidPts.shape[0]/CurSelectDepthData.flatten().shape[0]
    print(CurSelectDepthDataInvalidPts.shape, CurSelectDepthDataInvalidPtsRate)
    
#    # CurRGBData
#    CurRGBFileName = os.path.join(SelectRGBImageFolderName, 'Color-2020-09-05-092917_10005.jpg')
#    CurRGBData = cv2.imread(CurRGBFileName)
#    print('CurRGBData size = {}'.format(CurRGBData.shape))
    
    cv2.imshow('test_select.png', CurSelectDepthData)
    
    print('End.')
    