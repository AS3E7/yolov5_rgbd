# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:09:34 2019

HSF file functions

@author: Administrator
"""

import os
import struct
#import imageio 

#-----------
import csv
import zlib
#-----------

import numpy as np

#from DataPreprocess import TransShowDepthData

def GetHsfFileInfo(FileName, DataFormat='Depth'):
    """
    功能：获取文件帧数
    输入：文件名
    输出：文件帧数
    """
    
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
    
    CsvPath = ''
    
    if DataPressType == 5: # 解压数据读取头部
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
        FrameNum = int(np.floor((fsize-12*2)/EveryFrmDataByteSize)) 
    
    elif DataPressType == 2: # 压缩数据读取头部
        CsvPath = FileName[:-4] + ".csv"
        if os.path.exists(CsvPath):
            with open(CsvPath) as CsvFp:
                CsvReader = csv.reader(CsvFp)
                count = 0
                for row in CsvReader:
                    try:
                        int(row[0])
                        count += 1
                    except ValueError:
                        pass
        else:
            count = 0
        FrameNum = count
        
    fp.close()
    
    # frame number
    
    FileInfo = {'FileName': FileName,
                'FileFormat': DataFormat,                
                'FrameNum': FrameNum,
                'DataBeginHead': DataBeginHead,
                'DataSenSorType': DataSenSorType,
                'DataDataType': DataDataType,
                'DataDataFormat': DataDataFormat,
                'DataPressType': DataPressType,
                'DataContentFormat': DataContentFormat,
                'DataWidth': DataWidth,
                'DataHeight': DataHeight,
                'DataFiled1': DataFiled1,
                'DataFiled2': DataFiled2,
                'DataFiled3': DataFiled3,
                'DataEndHead': DataEndHead,
                'CsvPath': CsvPath,
               }    
               
    return FileInfo

def GetHSFDataFrames(FileName, FrameIdx, fp = None, file_info = None, DataFormat='Depth'):
    """
    功能：获取文件一帧或多帧数据
    输入：
        FileName：文件名
        FrameIdx：帧序号
    输出：文件帧数
    
    模式1：不指定fp和file_info，从文件头读起。接口简单，效率较低
    模式2：载入已经准备好的fp和file_info。接口复杂一些，效率高
    """ 
    
    if(not file_info):
        file_info = GetHsfFileInfo(FileName, DataFormat)

    DataDataFormat = file_info['DataDataFormat']
    DataWidth = file_info['DataWidth']
    DataHeight = file_info['DataHeight']

#    DataBeginHead = file_info['DataBeginHead']
#    DataSenSorType = file_info['DataSenSorType']
#    DataDataType = file_info['DataDataType']
    DataPressType = file_info['DataPressType']
#    DataContentFormat = file_info['DataContentFormat']
#    DataFiled1 = file_info['DataFiled1']
#    DataFiled2 = file_info['DataFiled2']
#    DataFiled3 = file_info['DataFiled3']
#    DataEndHead = file_info['DataEndHead']
    CsvPath = file_info['CsvPath']
    
    # 如果输入fp为有效指针，则保持打开
    # 否则在本函数中打开，调用结束后关闭
    b_close_fp = False
    if(not fp):
        fp = open(FileName, "rb")
        b_close_fp = True
    
    # ========== read HSF file ==========

    # data type
    OneFrmTimeSize = 8 # (Byte)
    OneFrmByteSize = 8 # (Byte)

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
    if isinstance(FrameIdx,int): # 读取一帧数据
        TempFrmData, TempFrmTime, CurFrameData = ReadHSFFile(FileName, DataFormat, FrameIdx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth, DataPressType, CsvPath)
        CurFrmData = TempFrmData
    else: # 读取多帧数据
        SelectFrmData = []
        for s_idx in FrameIdx:
            TempFrmData, TempFrmTime, CurFrameData = ReadHSFFile(FileName, DataFormat, s_idx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth, DataPressType, CsvPath)
            SelectFrmData.append(TempFrmData)
        SelectFrmData = np.array(SelectFrmData)
        CurFrmData = SelectFrmData
    
    if(b_close_fp):  
        fp.close()
    
    return CurFrmData, TempFrmTime

def ReadHSFFile(FileName, DataFormat, FrameIdx, fp, EveryFrmDataByteSize, DataPtsByteSize, ImgSize, DataPtsChannelSize, ImgHeight, ImgWidth, DataPressType, CsvPath):
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
    if DataPressType == 5: # 解压文件
        if DataFormat == 'Depth': # WDEPTH
            iFrame = FrameIdx
            fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame - fp.tell(), 1) # offset from current position
            TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data, # unsigned long long 
            TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data, # unsigned long long 
            if DataPtsByteSize == 2:
                TempFrmData =np.fromfile(fp, np.int16, count=ImgSize * DataPtsChannelSize) # depth data, 
                CurFrameData = TempFrmData
            else:
                assert DataPtsByteSize == 2 # Depth数据字节错误
            TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])
    
        elif DataFormat == 'Color': # WCOLOR
            iFrame = FrameIdx
            fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame - fp.tell(), 1) # offset
            TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data 
            TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data
            # read data
            if DataPtsByteSize == 1:
                TempFrmData1 = np.fromfile(fp, np.uint8, count=ImgSize * DataPtsChannelSize) # color data, # unsigned char
                CurFrameData = TempFrmData1
            else:
                assert DataPtsByteSize == 1 # Color数据字节错误
            # reshape data
            if DataPtsChannelSize > 0:
                TempFrmData1 = np.reshape(TempFrmData1, [ImgHeight, ImgWidth, DataPtsChannelSize]) # reshape [height, weight, 3]
    
                TempFrmData2 = np.zeros([ImgHeight, ImgWidth, DataPtsChannelSize], dtype=np.uint8) # [bgr -> rgb]
                for i_channel in range(DataPtsChannelSize):
                    TempFrmData2[:,:,i_channel] = TempFrmData1[:,:,DataPtsChannelSize-i_channel-1]
                
                TempFrmData = TempFrmData2
                
            else:
                TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])
                
    elif DataPressType == 2: # 压缩文件
        SkipByteSize = SkipByte(CsvPath, FrameIdx)
        fp.seek(DataTitleByteSize + SkipByteSize - fp.tell(), 1)
        
        while True:
            TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data, # unsigned long long 
#            print(TempFrmTime)
#            if int(TempFrmTime/100000000000) == 15:
#                break
            if int(TempFrmTime/100000000000) == 15 or int(TempFrmTime/100000000000) == 16:
                break
            
        TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data, # unsigned long long
        TempFrmData = zlib.decompress(fp.read(TempFrmSize))
        
        if DataFormat == 'Depth': # WDEPTH
            TempFrmData = np.frombuffer(TempFrmData, np.int16, count=ImgSize * DataPtsChannelSize) # depth data, 
            TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])
        elif DataFormat == 'Color': # WCOLOR
            TempFrmData1 = np.frombuffer(TempFrmData, np.uint8, count=ImgSize * DataPtsChannelSize) # color data, # unsigned char
            if DataPtsChannelSize > 0:
                TempFrmData1 = np.reshape(TempFrmData1, [ImgHeight, ImgWidth, DataPtsChannelSize]) # reshape [height, weight, 3]
    
                TempFrmData2 = np.zeros([ImgHeight, ImgWidth, DataPtsChannelSize], dtype=np.uint8) # [bgr -> rgb]
                for i_channel in range(DataPtsChannelSize):
                    TempFrmData2[:,:,i_channel] = TempFrmData1[:,:,DataPtsChannelSize-i_channel-1]
                
                TempFrmData = TempFrmData2
                
            else:
                TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth])
        CurFrameData = TempFrmData
            
    else: # WNORMAL
        iFrame = FrameIdx
        fp.seek(DataTitleByteSize + EveryFrmDataByteSize*iFrame - fp.tell(), 1) # offset
        TempFrmTime = struct.unpack("Q", fp.read(OneFrmTimeSize))[0] # time data
        TempFrmSize = struct.unpack("Q", fp.read(OneFrmByteSize))[0] # size data
        TempFrmData = np.fromfile(fp, np.int16, count=ImgSize * DataPtsChannelSize) # depth data
        TempFrmData = np.reshape(TempFrmData, [ImgHeight, ImgWidth]) 

    return TempFrmData, TempFrmTime, CurFrameData

# --------------------------------------------------------------------
# 读取和保存 .HSF 文件
#       SaveHSFFile(FileName, DataFormat, FrameRange, SaveFileHeadName)
# --------------------------------------------------------------------
def SaveHSFFile(FileName, DataFormat, FrameRange, SaveFileHeadName, Param):
    # inputs:
    #       FileName: .hsf file name
    #       DataFormat: 'rgb' or 'depth'
    #       FrameRange: frame idx
    #       SaveFileHeadName: save file name head
    # outputs: 
    #       
    
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
                
#        # imshow file
#        if ShowImgFlag == 1:
#            if DataFormat == 'Depth': # WDEPTH
#                plt.imshow(TransShowDepthData(TempFrmData),cmap = plt.cm.jet) # gray  = plt.cm.gray
#                if SaveDepthSrcData == 1: # 原始 depth 数据
#                    ImgDataSave = TransShowDepthData(TempFrmData) # 深度值范围变化
#                elif SaveDepthPseudoData == 1: # 配色后 depth 数据
#                    ImgDataSave = cv2.applyColorMap(TransShowDepthData(TempFrmData).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
#                
#                
#            elif DataFormat == 'Color': # WDEPTH
#                plt.imshow(TempFrmData) # gray  = plt.cm.gray
#                ImgDataSave = TempFrmData
#            else:
#                plt.imshow(TransShowDepthData(TempFrmData),cmap = plt.cm.jet) # gray  = plt.cm.gray
#                                                                              # jet = plt.cm.jet
#                ImgDataSave = cv2.applyColorMap(TransShowDepthData(TempFrmData).astype(np.uint8), cv2.COLORMAP_JET) # [Height, Width, 3]
#                
#        # save image file
#        if SaveFileFlag == 1:
#            # 保存图片
#            SaveFileName = SaveFileHeadName + '_' + str(FrameIdx).zfill(5) + '.png' # save file name
#            # plt.savefig(SaveFileName) # save figure
#            # imsave(SaveFileName, ImgDataSave) # save image-data
#            imageio.imwrite(SaveFileName, ImgDataSave) # save image-data

            
#        time.sleep(0.001)
            
    fp.close()
    return 0

def SkipByte(CsvPath, FrameNum):
    '''
    Function:
        跳过帧数统计
        
    Input:
        CsvPath(str) : CSV路径
        FrameNum(int) : 想要跳转到的帧数
        
    Output:
        SkipNum(int) : 需跳过的字节数
    '''
    CsvData = []
    with open(CsvPath) as CsvFp:
        CsvReader = csv.reader(CsvFp)
        count = 0
        for row in CsvReader:
            try:
                int(row[0])
                row[0] = count
                count += 1
                CsvData.append(row)
            except ValueError:
                pass
                # print("csv File has error")
    SkipNum = 0 # 2 * 12
    for i in range(FrameNum):
        SkipNum += 16 + (int(CsvData[i][4]))
    return SkipNum
    
    
# --------------------------------------------------------------------
# 待保存数据信息
#       DataInfo()
# --------------------------------------------------------------------
class DataInfo():
    def __init__(self):
        self.ColorHSFFileName = ''# color data
        self.DepthHSFFileName = ''# depth data
        self.SaveColor = 0# 是否保存Color数据
        self.SaveDepth = 0# 是否保存Depth数据，配色图片
        self.SaveSrcDepth = 0# 是否保存Depth数据，灰度图片
        self.SaveDepthEachFrame = 0# 是否保存Depth每帧数据，保存为depth 文件
        self.FrameRange = 0# 选择帧数
        self.SaveFolder = ''# 图片保存地址

if __name__ == '__main__':

#    import cv2
#    import matplotlib.pyplot as plt
#
#    # 本模块的测试用例
#    TestCase = 1
#    
#    if(TestCase == 1):
#        HsfFile = r'G:\GateCounter\未整理\201910_珠海\20191019(珠海拱北口岸测试)\帽子误报\Depth2019-10-18 211725.399.HSF.HSF'
#        FrmNum = GetHSFDataFrameNum(HsfFile, DataFormat='Depth')
#        print('HSF file = ', HsfFile)
#        print('FrameNum = ', FrmNum)
#        
#        plt.figure("HsfFileDemo")
#        
#        for nk in range(FrmNum):            
#            FrmData, FrmTime = GetHSFDataFrames(HsfFile, nk, DataFormat='Depth')
#            print('FrmIdx = ', nk, ', FrmTime = ', FrmTime, ', DataSize = ', FrmData.shape)
#
#            plt.clf()   
#            plt.imshow(FrmData)
#            plt.title('Frame #{}'.format(nk))
#            plt.show()                        
#            plt.pause(0.05) 
#        
    pass