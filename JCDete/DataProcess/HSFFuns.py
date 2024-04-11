# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 18:07:06 2020

@author: HYD
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from io_hsf import io_hsf

import FileHsf

def ReadOneFrameDepthFile(FileName, DepthWidth, DepthHeight):
    """
    读取一帧depth文件
    """
    fp = open(FileName, "rb")
    ImgSize = DepthWidth * DepthHeight
    DataChannelSize = 1
    DepthData = np.fromfile(fp, np.int16, count=ImgSize * DataChannelSize) # depth data
    DepthData = np.reshape(DepthData, [DepthHeight, DepthWidth])
    fp.close()
    
    return DepthData

def GenImageFromDepth(DepthData, TransDepthRange, SaveImageName = '', TransFormat = 'Gray'):
    """
    从深度数据中获取图片结果
    """
    # transform depth data, 'Gray'/'Colormap'
    if TransFormat == 'Gray':
        ImgDataSave = TransDepthData(DepthData, TransDepthRange) # 深度值范围变化, 8位图片数据
    elif TransFormat == 'Gray24':
        ImgDataSaveSrc = TransDepthData(DepthData, TransDepthRange) # 深度值范围变化, 24位图片数据
        ImgDataSave = np.zeros([ImgDataSaveSrc.shape[0], ImgDataSaveSrc.shape[1], 3])
        ImgDataSave[:,:,0] = ImgDataSaveSrc
        ImgDataSave[:,:,1] = ImgDataSaveSrc
        ImgDataSave[:,:,2] = ImgDataSaveSrc
    elif TransFormat == 'Colormap':
        ImgDataSave = cv2.applyColorMap(TransDepthData(DepthData, TransDepthRange).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
    
    # save data
    if len(SaveImageName) > 0:
        imageio.imwrite(SaveImageName, ImgDataSave) # save image-data
        
    return ImgDataSave
    
def TransDepthData(SrcDepthData, ShowDepthRange = [0, 8000]):
    # 转换深度数据的数据范围
    # inputs:
    #       SrcDepthData: source Depth Data
    #       ShowDepthRange = [0, 4000] # init: [0mm, 8000mm]
    # outputs: 
    #       DesDepthData: des depth data
    
    # 设置颜色变化范围，单位 mm
#    ShowDepthRange = [0, 8000] # init: [0, 8000]
#    ShowDepthRange = [2000, 5000] # 此处设置 2m-5m

#    DesDepthData = SrcDepthData
    DesDepthData = np.array(SrcDepthData)
    
    # 显示depth 距离范围
    LowerDepthIdx = (SrcDepthData < ShowDepthRange[0])
    
    DesDepthData[LowerDepthIdx] = ShowDepthRange[0]
    UpperDepthIdx = (SrcDepthData > ShowDepthRange[1])
    DesDepthData[UpperDepthIdx] = ShowDepthRange[1]
    
    # 转换至图片数据范围 [0 - 255]
    ColorRange = [0, 255]
    DesDepthData = (DesDepthData - ShowDepthRange[0])/((ShowDepthRange[1]-ShowDepthRange[0])/(ColorRange[1]-ColorRange[0]))
    
    return DesDepthData
    
def ReadCalibParamsInfo(CalibFileName):
    """
    读取配置文件信息
    输入：
        CalibFileName：配置文件名
    输出：
        H：配准参数
    """
    fp = open(CalibFileName)
    H = []
    while 1:
        line = fp.readline().strip()
#        print(line)
        if line == '[H]':
            continue
        elif len(line) < 2:
            break
        else:
            line = line.split(' ')
            for i_line in line:
#                print(i_line)
                if len(i_line) > 0:
                    H.append(float(i_line))
    fp.close()
    H = np.array(H).reshape((4,4))
    
    return H

def SaveHSFFile(FileName, DataFormat, FrameRange, SaveFileHeadName, TransDepthRange, SelectColorHSFFileFlag, SelectDepthHSFFileFlag, SelectSrcDepthHSFFileFlag, SelectSrcDepthEachFrameFlag):
    """
    保存 HSF 文件中的图片信息
        使用 io_hsf 读取压缩文件
    """
    # file frame number
    FileInfo = io_hsf.GetHsfFileInfo(FileName, DataFormat=DataFormat)
    print('FileInfo = ', FileInfo)
    CurFrmNum = FileInfo['FrameNum']
    print('current frame number = {}'.format(CurFrmNum))
    
    # read .HSF file
    for FrameIdx in FrameRange:
        if FrameIdx>CurFrmNum-1:
            print('  FrameIdx out range')
            break
        
        print('FrameIdx = {}, DataFormat = {}'.format(FrameIdx, DataFormat))
        CurFrmData, TempFrmTime = io_hsf.GetHSFDataFrames(FileName, FrameIdx, DataFormat=DataFormat) # CurFrmData = [512,424] / CurFrmData = [1920,1080,3]
        # save as image
        # RGB
        if SelectColorHSFFileFlag == 1 and DataFormat == 'Color':
            SaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.png' # save file name
            imageio.imwrite(SaveFileName, CurFrmData) # save image-data
        # Colormap
        if SelectDepthHSFFileFlag == 1 and DataFormat == 'Depth':
            SaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.png' # save file name
            # colormap image
            ImgDataSave = cv2.applyColorMap(TransDepthData(CurFrmData, TransDepthRange).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
            imageio.imwrite(SaveFileName, ImgDataSave) # save image-data
        # Gray
        if SelectSrcDepthHSFFileFlag == 1 and DataFormat == 'Depth':
            SaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.png' # save file name
            # gray image
            ImgDataSave = TransDepthData(CurFrmData, TransDepthRange) # 深度值范围变化
            imageio.imwrite(SaveFileName, ImgDataSave) # save image-data
        # depth
        if SelectSrcDepthEachFrameFlag == 1 and DataFormat == 'Depth':
            DepthSaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.depth' # save .depth file name
            # depth data
            DepthDataSave = CurFrmData.flatten() # 深度数据
            fpDepth = open(DepthSaveFileName, 'wb')
            fpDepth.write(DepthDataSave)
            fpDepth.close()
    
    return 0
    


if __name__ == '__main__':
    print('Start.')
    TestReadCalibParamsInfoFlag = 1 # 读取配置文件信息
    
    if TestReadCalibParamsInfoFlag == 1:
        CalibFileName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\CalibParam\H_171.txt'
        CalibH = ReadCalibParamsInfo(CalibFileName)
        print('  CalibH = {}'.format(CalibH))
    
    print('End.')
    