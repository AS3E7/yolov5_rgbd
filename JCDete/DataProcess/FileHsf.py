# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:09:34 2019

HSF file functions

from FileHsf import GetHSFDataFrameNum, GetHSFDataOneFrameInfo, ReadHSFFile, SaveHSFFile

@author: Administrator
"""

import os
import struct
#import imageio 

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio 

# 其他参数类
class SaveParam():
    def __init__(self):
        self.SaveColor = 0 # 保存 Color
        self.SaveDepthPseudoColor = 0 # 保存Depth伪彩色
        self.SaveDepthGray = 0 # 保存Depth灰度
        self.SaveDepthEachFrame = 0 # 保存Depth每帧的 depth 文件

def GetHSFDataFrameNum(FileName, DataFormat='Depth'):
    """
    功能：获取文件帧数
    输入：文件名
    输出：文件帧数
    """
    NumFrame = 0
    
    # read file size
    fsize = os.path.getsize(FileName)
    
    # read .HSF file
    fp = open(FileName, "rb")
    
    # data type
    OneFrmTimeSize = 8 # (Byte)
    OneFrmByteSize = 8 # (Byte)
    
    # read file head
    DataBeginHead = fp.read(2)
    DataSenSorType = fp.read(2)
    DataDataType = fp.read(2)
    DataDataFormat = fp.read(2)
    DataPressType = fp.read(2)
    DataContentFormat = fp.read(2)
    DataWidth = fp.read(2)
    DataHeight = fp.read(2)
    DataFiled1 = fp.read(2)
    DataFiled2 = fp.read(2)
    DataFiled3 = fp.read(2)
    DataEndHead = fp.read(2)
    
    DataBeginHead = struct.unpack("H", DataBeginHead)[0] # unsigned short
    DataSenSorType = struct.unpack("H", DataSenSorType)[0]
    DataDataType = struct.unpack("H", DataDataType)[0]
    DataDataFormat = struct.unpack("H", DataDataFormat)[0]
    DataPressType = struct.unpack("H", DataPressType)[0]
    DataContentFormat = struct.unpack("H", DataContentFormat)[0]
    DataWidth = struct.unpack("H", DataWidth)[0]
    DataHeight = struct.unpack("H", DataHeight)[0]
    DataFiled1 = struct.unpack("H", DataFiled1)[0]
    DataFiled2 = struct.unpack("H", DataFiled2)[0]
    DataFiled3 = struct.unpack("H", DataFiled3)[0]
    DataEndHead = struct.unpack("H", DataEndHead)[0]
    
#    print("Data Title Info : BeginHead = {:3}, SenSorType = {:3}, DataType = {:3}, DataFormat = {:3}, \
#                PressType = {:3}, ContentFormat = {:3}, Width = {:3}, Height = {:3}, \
#                Filed1 = {:3}, Filed2 = {:3}, Filed3 = {:3}, EndHead = {:3}".format( \
#                DataBeginHead, DataSenSorType, DataDataType, DataDataFormat, \
#                DataPressType, DataContentFormat, DataWidth, DataHeight, \
#                DataFiled1, DataFiled2, DataFiled3, DataEndHead))
    
    # data information
    ImgWidth = DataWidth
    ImgHeight = DataHeight
    if DataFormat == 'Depth': # WDEPTH
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    elif DataFormat == 'Color': # WCOLOR
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 3
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    else: # WNORMAL
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
        
    fp.close()
    
    # frame number
    NumFrame = int(np.floor((fsize-12*2)/EveryFrmDataByteSize))    
    
    return NumFrame

def GetHSFDataFrames(CurHSFFileName, CurFrameIdx, DataFormat='Depth'):
    """
    功能：获取文件一帧或多帧数据
    输入：
        CurHSFFileName：文件名
        CurFrameIdx：帧序号
    输出：文件帧数
    """ 
    
    # ========== read HSF file ==========
    # read .HSF file
    fp = open(CurHSFFileName, "rb")
    
    # data type
    OneFrmTimeSize = 8 # (Byte)
    OneFrmByteSize = 8 # (Byte)
    
    # read file head
    DataBeginHead = fp.read(2)
    DataSenSorType = fp.read(2)
    DataDataType = fp.read(2)
    DataDataFormat = fp.read(2)
    DataPressType = fp.read(2)
    DataContentFormat = fp.read(2)
    DataWidth = fp.read(2)
    DataHeight = fp.read(2)
    DataFiled1 = fp.read(2)
    DataFiled2 = fp.read(2)
    DataFiled3 = fp.read(2)
    DataEndHead = fp.read(2)
    
    DataBeginHead = struct.unpack("H", DataBeginHead)[0] # unsigned short
    DataSenSorType = struct.unpack("H", DataSenSorType)[0]
    DataDataType = struct.unpack("H", DataDataType)[0]
    DataDataFormat = struct.unpack("H", DataDataFormat)[0]
    DataPressType = struct.unpack("H", DataPressType)[0]
    DataContentFormat = struct.unpack("H", DataContentFormat)[0]
    DataWidth = struct.unpack("H", DataWidth)[0]
    DataHeight = struct.unpack("H", DataHeight)[0]
    DataFiled1 = struct.unpack("H", DataFiled1)[0]
    DataFiled2 = struct.unpack("H", DataFiled2)[0]
    DataFiled3 = struct.unpack("H", DataFiled3)[0]
    DataEndHead = struct.unpack("H", DataEndHead)[0]

    # data information
    ImgWidth = DataWidth
    ImgHeight = DataHeight
    if DataFormat == 'Depth': # WDEPTH
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    elif DataFormat == 'Color': # WCOLOR
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 3
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    else: # WNORMAL
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
        
    # read on selct frame
    if isinstance(CurFrameIdx,int): # 读取一帧数据
        TempFrmData, TempFrmTime, CurFrameData = ReadHSFFile(CurHSFFileName, DataFormat, CurFrameIdx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth)
        CurFrmData = TempFrmData
    else: # 读取多帧数据
        SelectFrmData = []
        for s_idx in CurFrameIdx:
            TempFrmData, TempFrmTime, CurFrameData = ReadHSFFile(CurHSFFileName, DataFormat, s_idx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth)
            SelectFrmData.append(TempFrmData)
        SelectFrmData = np.array(SelectFrmData)
        CurFrmData = SelectFrmData
        
    fp.close()
    
    return CurFrmData, TempFrmTime

def ReadHSFFile(FileName, DataFormat, FrameIdx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth):
    # inputs:
    #       FileName: .hsf file name
    #       DataFormat: 'rgb' or 'depth'
    #       FrameIdx: frame idx
    #       fp: .HSF file
    #       ImgSize, ImgHeight, ImgWidth [image info]
    #       EveryFrmDataByteSize, DataPtsByteSize, DataPtsChannelSize [data byte info]
    #       SaveDepthFrame: save depth frame as .depth file
    # outputs: 
    #       TempFrmData: [H x W] or [H x W x 3]
    #       TempFrmTime: temp frame time
    
    # data type
    OneFrmTimeSize = 8 # Frame Time (Byte)
    OneFrmByteSize = 8 # Frame Image Size (Byte)
    
    # file head size
    DataTitleByteSize = 12*2 # data title size 24(Byte)
    
    # 当前帧数据
    CurFrameData = []
    
    # data
    if DataFormat == 'Depth': # WDEPTH
        iFrame = FrameIdx
        fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame) # offset
        TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data, # unsigned long long 
        TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data, # unsigned long long 
        if DataPtsByteSize == 2:
            TempFrmData =np.fromfile(fp, np.int16, count=ImgSize * DataPtsChannelSize) # depth data, 
            CurFrameData = TempFrmData
        else:
            print('Depth DataFormat is 2.')
        TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])

    elif DataFormat == 'Color': # WCOLOR
        iFrame = FrameIdx
        fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame) # offset
        TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data 
        TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data
        # read data
        if DataPtsByteSize == 1:
            TempFrmData1 = np.fromfile(fp, np.uint8, count=ImgSize * DataPtsChannelSize) # color data, # unsigned char
            CurFrameData = TempFrmData1
        else:
            print('Color DataFormat is 1.')
        # reshape data
        if DataPtsChannelSize > 0:
            TempFrmData1 = np.reshape(TempFrmData1, [ImgHeight, ImgWidth, DataPtsChannelSize]) # reshape [height, weight, 3]

            TempFrmData2 = np.zeros([ImgHeight, ImgWidth, DataPtsChannelSize], dtype=np.uint8) # [bgr -> rgb]
            for i_channel in range(DataPtsChannelSize):
                TempFrmData2[:,:,i_channel] = TempFrmData1[:,:,DataPtsChannelSize-i_channel-1]
            
            TempFrmData = TempFrmData2
            
        else:
            TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])

    else: # WNORMAL
        iFrame = FrameIdx
        fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame) # offset
        TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data
        TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data
        TempFrmData = np.fromfile(fp, np.int16, count=ImgSize * DataPtsChannelSize) # depth data
        TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth]) 

    return TempFrmData, TempFrmTime, CurFrameData

# --------------------------------------------------------------------
# 读取和保存 .HSF 文件
#       SaveHSFFile(FileName, DataFormat, FrameRange, SaveFileHeadName)
# --------------------------------------------------------------------
def SaveHSFFile(FileName, DataFormat, FrameRange, SaveFileHeadName, TransDepthRange, Param):
    # inputs:
    #       FileName: .hsf file name
    #       DataFormat: 'rgb' or 'depth'
    #       FrameRange: frame idx
    #       SaveFileHeadName: save file name head
    # outputs: 
    #      
    
    CurFrmNum = GetHSFDataFrameNum(FileName, DataFormat=DataFormat)
    
    # imshow and save file
    ShowImgFlag = 1 
    SaveFileFlag = 1
    SaveDepthSrcData = Param.SaveDepthGray # 保存原始 depth数据图片
    SaveDepthPseudoData = Param.SaveDepthPseudoColor # 保存配色后的 depth数据图片
    SaveDepthEachFrame = Param.SaveDepthEachFrame # 保存Depth每帧的 depth 文件
    
    # read .HSF file
    fp = open(FileName, "rb")
    
    # data type
    OneFrmTimeSize = 8 # (Byte)
    OneFrmByteSize = 8 # (Byte)
    
    # read file head
    DataBeginHead = fp.read(2)
    DataSenSorType = fp.read(2)
    DataDataType = fp.read(2)
    DataDataFormat = fp.read(2)
    DataPressType = fp.read(2)
    DataContentFormat = fp.read(2)
    DataWidth = fp.read(2)
    DataHeight = fp.read(2)
    DataFiled1 = fp.read(2)
    DataFiled2 = fp.read(2)
    DataFiled3 = fp.read(2)
    DataEndHead = fp.read(2)
    
    DataBeginHead = struct.unpack("H", DataBeginHead)[0] # unsigned short
    DataSenSorType = struct.unpack("H", DataSenSorType)[0]
    DataDataType = struct.unpack("H", DataDataType)[0]
    DataDataFormat = struct.unpack("H", DataDataFormat)[0]
    DataPressType = struct.unpack("H", DataPressType)[0]
    DataContentFormat = struct.unpack("H", DataContentFormat)[0]
    DataWidth = struct.unpack("H", DataWidth)[0]
    DataHeight = struct.unpack("H", DataHeight)[0]
    DataFiled1 = struct.unpack("H", DataFiled1)[0]
    DataFiled2 = struct.unpack("H", DataFiled2)[0]
    DataFiled3 = struct.unpack("H", DataFiled3)[0]
    DataEndHead = struct.unpack("H", DataEndHead)[0]
    
    print("Data Title Info : BeginHead = {:3}, SenSorType = {:3}, DataType = {:3}, DataFormat = {:3}, \
                PressType = {:3}, ContentFormat = {:3}, Width = {:3}, Height = {:3}, \
                Filed1 = {:3}, Filed2 = {:3}, Filed3 = {:3}, EndHead = {:3}".format( \
                DataBeginHead, DataSenSorType, DataDataType, DataDataFormat, \
                DataPressType, DataContentFormat, DataWidth, DataHeight, \
                DataFiled1, DataFiled2, DataFiled3, DataEndHead))
    
    # data information
    ImgWidth = DataWidth
    ImgHeight = DataHeight
    if DataFormat == 'Depth': # WDEPTH
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    elif DataFormat == 'Color': # WCOLOR
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 3
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize
    else: # WNORMAL
        DataPtsByteSize = DataDataFormat
        DataPtsChannelSize = 1
        ImgSize = ImgWidth * ImgHeight
        EveryFrmDataByteSize = OneFrmTimeSize + OneFrmByteSize + ImgSize*DataPtsChannelSize*DataPtsByteSize

    # read .HSF file
    for FrameIdx in FrameRange:
        if FrameIdx>CurFrmNum-1:
            print('  FrameIdx out range')
            break
        
        print('FrameIdx = {}'.format(FrameIdx))
        CurFrameData = []
        TempFrmData, TempFrmTime, CurFrameData = ReadHSFFile(FileName, DataFormat, FrameIdx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth)

        # save .kdepth file
        if SaveFileFlag == 1:
            # 保存 depth 文件
            if SaveDepthEachFrame == 1 and len(CurFrameData)>1:
                DepthSaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.depth' # save .depth file name
                fpDepth = open(DepthSaveFileName, 'wb')
                fpDepth.write(CurFrameData)
                fpDepth.close()
                
        # imshow file
        if ShowImgFlag == 1:
            if DataFormat == 'Depth': # WDEPTH
                plt.imshow(TransShowDepthData(TempFrmData, TransDepthRange = TransDepthRange),cmap = plt.cm.jet) # gray  = plt.cm.gray
                if SaveDepthSrcData == 1: # 原始 depth 数据
                    ImgDataSave = TransShowDepthData(TempFrmData, TransDepthRange = TransDepthRange) # 深度值范围变化
                elif SaveDepthPseudoData == 1: # 配色后 depth 数据
                    ImgDataSave = cv2.applyColorMap(TransShowDepthData(TempFrmData, TransDepthRange = TransDepthRange).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
                
                
            elif DataFormat == 'Color': # WDEPTH
                plt.imshow(TempFrmData) # gray  = plt.cm.gray
                ImgDataSave = TempFrmData
            else:
                plt.imshow(TransShowDepthData(TempFrmData, TransDepthRange = TransDepthRange),cmap = plt.cm.jet) # gray  = plt.cm.gray
                                                                              # jet = plt.cm.jet
                ImgDataSave = cv2.applyColorMap(TransShowDepthData(TempFrmData, TransDepthRange = TransDepthRange).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
                
        # save image file
        if SaveFileFlag == 1:
            # 保存图片
            SaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.png' # save file name
            # plt.savefig(SaveFileName) # save figure
            # imsave(SaveFileName, ImgDataSave) # save image-data
            imageio.imwrite(SaveFileName, ImgDataSave) # save image-data

            
#        time.sleep(0.001)
            
    fp.close()
    return 0
    
    
# --------------------------------------------------------------------
# 转换待显示的Depth 数据
#       TransShowDepthImgae(SrcDepthData)
# --------------------------------------------------------------------
def TransShowDepthData(SrcDepthData, TransDepthRange = [0, 4000]):
    # inputs:
    #       SrcDepthData: source Depth Data
    # outputs: 
    #       DesDepthData: des depth data
    
    # 设置颜色变化范围，单位 mm
#    ShowDepthRange = [0, 8000] # init: [0, 8000]
#    ShowDepthRange = [2000, 5000] # 此处设置 2m-5m

#    ShowDepthRange = [0, 4000] # init: [0mm, 8000mm]
    ShowDepthRange = TransDepthRange
    
    DesDepthData = SrcDepthData
    
    # 显示depth 距离范围
    LowerDepthIdx = (SrcDepthData < ShowDepthRange[0])
    DesDepthData[LowerDepthIdx] = ShowDepthRange[0]
    UpperDepthIdx = (SrcDepthData > ShowDepthRange[1])
    DesDepthData[UpperDepthIdx] = ShowDepthRange[1]
    
    # 转换至图片数据范围 [0 - 255]
    ColorRange = [0, 255]
    DesDepthData = (DesDepthData - ShowDepthRange[0])/((ShowDepthRange[1]-ShowDepthRange[0])/(ColorRange[1]-ColorRange[0]))
    
    return DesDepthData


if __name__ == '__main__':

    import cv2
    import matplotlib.pyplot as plt

    # 本模块的测试用例
    TestCase = 1
    
    if(TestCase == 1):
        HsfFile = r'G:\GateCounter\未整理\201910_珠海\20191019(珠海拱北口岸测试)\帽子误报\Depth2019-10-18 211725.399.HSF.HSF'
        FrmNum = GetHSFDataFrameNum(HsfFile, DataFormat='Depth')
        print('HSF file = ', HsfFile)
        print('FrameNum = ', FrmNum)
        
        plt.figure("HsfFileDemo")
        
        for nk in range(FrmNum):            
            FrmData, FrmTime = GetHSFDataFrames(HsfFile, nk, DataFormat='Depth')
            print('FrmIdx = ', nk, ', FrmTime = ', FrmTime, ', DataSize = ', FrmData.shape)

            plt.clf()   
            plt.imshow(FrmData)
            plt.title('Frame #{}'.format(nk))
            plt.show()                        
            plt.pause(0.05) 
        
    pass