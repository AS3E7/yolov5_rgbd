# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 09:24:20 2021

@author: Administrator
"""
import os

import DataPreprocess
from io_hsf import io_hsf

from PointCloudFunsC import PointCloudFunsC
from HSFFuns import ReadOneFrameDepthFile, GenImageFromDepth
from CloudPointFuns import SavePt3D2Ply

class HSFFileImageProcess():
    """
    功能：HSF文件相关图像处理
    """
    def __init__(self, SrcDataInfo, DestDataInfo, Params):
        self.SrcDataInfo = SrcDataInfo
        self.DestDataInfo = DestDataInfo
        self.Params = Params
        
    def GenImageFromHSFFile(self):
        """
        功能：从HSF文件中获取图像数据
        """
        SrcDataInfo = self.SrcDataInfo
        DestDataInfo = self.DestDataInfo
        Params = self.Params
        
        # 深度数据深度值范围
        TransDepthRange = Params['ColormapDepthRange'] # 深度数据深度值范围
        # 原始文件地址
        SrcRGBHSFFileDir = SrcDataInfo['RGBHSFDir']
        SrcDepthHSFFileDir = SrcDataInfo['DepthHSFDir']
        # 保存文件地址
        SaveDestColorFlag = Params['SaveColorImage']
        SaveDestDepthFlag = Params['SaveDepthColormapImage']
        SaveDestSrcDepthFlag = Params['SaveDepthImage']
        SaveDestDepthEachFrameFlag = Params['SaveDepthFile']
        
        SaveDestDepthHSFFileDir = DestDataInfo['DepthHSFDir']
        SaveDestRGBHSFFileDir = DestDataInfo['RGBHSFDir']
        # 选择标注数据 
        SelectFrmIdxMin = Params['FrmMin']
        SelectFrmIdxMax = Params['FrmMax']
        SelectFrmIdxStep = Params['FrmStep']
        SelectSenorName = Params['SenorName'] # 选择传感器名称
        SelectTimeAll = Params['FileTime']
    
        SelectFrame = range(SelectFrmIdxMin,SelectFrmIdxMax,SelectFrmIdxStep) # 选择帧数，Frame序号从 0 开始
    
        # 遍历 HSF 文件
        for i_sensor_name in SelectSenorName:
            CurSensorName = i_sensor_name
            for SelectTime in SelectTimeAll:
                print('CurSensorName = {}, SelectTime = {}'.format(CurSensorName, SelectTime))
                DataInfoIn = io_hsf.DataInfo()
                
                DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, SelectTime + '.HSF') # color data
                DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, SelectTime + '.HSF') # depth data
                
                # DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, 'Color-' + SelectTime + '.HSF') # color data
                # DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, 'Depth-' + SelectTime + '.HSF') # depth data
                if not os.path.exists(DataInfoIn.DepthHSFFileName):
                    DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, 'Color' + SelectTime + '.HSF') # color data
                    DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, 'Depth' + SelectTime + '.HSF') # depth data
                if not os.path.exists(DataInfoIn.DepthHSFFileName):
                    print('depth file not exist {}'.format(DataInfoIn.ColorHSFFileName))
                
                DataInfoIn.SaveColor = SaveDestColorFlag # 是否保存Color数据,[.png]
                DataInfoIn.SaveDepth = SaveDestDepthFlag # 是否保存Depth数据，配色图片,[.png]
                DataInfoIn.SaveSrcDepth = SaveDestSrcDepthFlag # 是否保存Depth数据，灰度图片，【.png】
                DataInfoIn.SaveDepthEachFrame = SaveDestDepthEachFrameFlag # 是否保存Depth每帧数据，保存为depth 文件,[.depth]
                DataInfoIn.FrameRange = SelectFrame # 选择帧数
                DataInfoIn.SaveFolder = os.path.join(SaveDestDepthHSFFileDir, CurSensorName, SelectTime) # 图片保存地址       
                DataInfoIn.PrefixName = CurSensorName # 图片文件名前缀，用于区别不同的传感器，便于查找原始HSF文件
                # generate images
                DataPreprocess.GenMultiFrameFromHSF(DataInfoIn, TransDepthRange)
    
        return 0


if __name__ == '__main__':
    print('start.')
    
    TestCase = 4 # TestCase = 1, ZT dataset, 监仓检测数据
                 # TestCase = 2, Fall dataset, 跌倒检测数据
                 # TestCase = 3, EK dataset, 二看，监仓检测数据
                 # TestCase = 4, EK dataset, 二看，监仓检测数据, 可生成点云
    
    
    if TestCase == 3: # EK dataset, 处理 HSF 文件相关图像处理
        DepthWidth = 512
        DepthHeight = 424
            
        # 输入数据信息
        SrcDataInfo = dict()
        SrcDataInfo['RGBHSFDir'] = r'Y:\ToYuJinyong\FromXbiao\Data\HSF\229'
        SrcDataInfo['DepthHSFDir'] = r'Y:\ToYuJinyong\FromXbiao\Data\HSF\229'
        # 输出数据信息
        DestDataInfo = dict()
        DestDataInfo['RGBHSFDir'] = r'Y:\ToYuJinyong\FromXbiao\Data\Depth\229'
        DestDataInfo['DepthHSFDir'] = r'Y:\ToYuJinyong\FromXbiao\Data\Depth\229'
        
        # 参数设置
        Params = dict()
        Params['SaveColorImage'] = 0 # 暂时不开放RGB 数据 
        Params['SaveDepthImage'] = 0
        Params['SaveDepthColormapImage'] = 1 
        Params['SaveDepthFile'] = 1 
        
        Params['ColormapDepthRange'] = [1000, 5000]
        Params['FrmMin'] = 0
        Params['FrmMax'] = 10000 # 10000
        Params['FrmStep'] = 10 # 10/40
        
        # Params['SenorName'] = ['243', '244', '245', '246', '247'，'248', '249', '250']
        Params['SenorName'] = ['250']
        Params['FileTime'] = [
            # 'Depth2021-11-05-163000'#, 243
            
            # 'Depth2021-11-05-163000'#, 244
            
            # 'Depth2021-11-05-163000'# 245

            # 'Depth2021-11-05-163000', # 246
                        
            # 'Depth2021-11-05-163000',  # 247
            
            # 'Depth2021-11-05-163000'  # 248
            
            # 'Depth2021-11-05-163000',  # 249
            
            'Depth2021-11-05-163000'   # 250
            
            ]
        
        
        CurHSFFileImageProcess = HSFFileImageProcess(SrcDataInfo, DestDataInfo, Params)
        # 获取图像
        CurHSFFileImageProcess.GenImageFromHSFFile()
        
        
    elif TestCase == 4: # EK dataset, 处理 HSF 文件相关图像处理, 可生成点云
            DepthWidth = 512
            DepthHeight = 424
                
            # 输入数据信息
            SrcDataInfo = dict()
            SrcDataInfo['RGBHSFDir'] = r'W:\ToYuJinyong\FromXbiao\Data\HSF\229'
            SrcDataInfo['DepthHSFDir'] = r'W:\ToYuJinyong\FromXbiao\Data\HSF\229'
            # 输出数据信息
            DestDataInfo = dict()
            DestDataInfo['RGBHSFDir'] = r'W:\ToYuJinyong\FromXbiao\Data\Depth\229'
            DestDataInfo['DepthHSFDir'] = r'W:\ToYuJinyong\FromXbiao\Data\Depth\229'
            DestDataInfo['PlyHSFDir'] = r'W:\ToYuJinyong\FromXbiao\Data\Ply\229'
            
            # 参数设置
            Params = dict()
            Params['SaveColorImage'] = 0 # 暂时不开放RGB 数据 
            Params['SaveDepthImage'] = 0
            Params['SaveDepthColormapImage'] = 1 
            Params['SaveDepthFile'] = 1 
            
            Params['ColormapDepthRange'] = [1000, 5000]
            Params['FrmMin'] = 0
            Params['FrmMax'] = 20 # 10000
            Params['FrmStep'] = 10 # 10/40
            
            # Params['SenorName'] = ['243', '244', '245', '246', '247'，'248', '249', '250']
            Params['SenorName'] = ['250']
            Params['FileTime'] = [
                # 'Depth2021-11-05-163000'#, 243
                
                # 'Depth2021-11-05-163000'#, 244
                
                # 'Depth2021-11-05-163000'# 245

                # 'Depth2021-11-05-163000', # 246
                            
                # 'Depth2021-11-05-163000',  # 247
                
                # 'Depth2021-11-05-163000'  # 248
                
                # 'Depth2021-11-05-163000',  # 249
                
                'Depth2021-11-05-163000'   # 250
                
                ]
            
            # 获取图像
            if 1:
                CurHSFFileImageProcess = HSFFileImageProcess(SrcDataInfo, DestDataInfo, Params)
                # 获取图像
                CurHSFFileImageProcess.GenImageFromHSFFile()
        
            # 获取点云信息
            if 1:
                if DepthWidth == 320:
                    ycenterF= 120.0
                    xcenterF= 160.0 # 160
                    VfoclenInPixelsF= 227 # 227
                    HfoclenInPixelsF= 227 # 227
                    CurDepth2PointCloud = PointCloudFunsC.Depth2PointCloud(DepthWidth, DepthHeight, xcenterF, ycenterF, HfoclenInPixelsF, VfoclenInPixelsF)
                elif DepthWidth == 512:
                    ycenterF= 202.546464
                    xcenterF= 256.685317
                    VfoclenInPixelsF= 368.901024
                    HfoclenInPixelsF= 368.874187
                    CurDepth2PointCloud = PointCloudFunsC.Depth2PointCloud(DepthWidth, DepthHeight, xcenterF, ycenterF, HfoclenInPixelsF, VfoclenInPixelsF)
            
                # 遍历文件
                SrcRGBHSFFileDir = SrcDataInfo['RGBHSFDir']
                SrcDepthHSFFileDir = SrcDataInfo['DepthHSFDir']
                SaveDestDepthHSFFileDir = DestDataInfo['RGBHSFDir']
                SaveDestPlyHSFFileDir = DestDataInfo['PlyHSFDir'] 
                SelectSenorName = Params['SenorName']
                SelectTimeAll = Params['FileTime']
                for i_sensor_name in SelectSenorName:
                    CurSensorName = i_sensor_name
                    for SelectTime in SelectTimeAll:
                        print('CurSensorName = {}, SelectTime = {}'.format(CurSensorName, SelectTime))
                        DataInfoIn = io_hsf.DataInfo()
                        DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, SelectTime + '.HSF') # color data
                        DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, SelectTime + '.HSF') # depth data
                        
                        # DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, 'Color-' + SelectTime + '.HSF') # color data
                        # DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, 'Depth-' + SelectTime + '.HSF') # depth data
                        if not os.path.exists(DataInfoIn.DepthHSFFileName):
                            DataInfoIn.ColorHSFFileName = os.path.join(SrcRGBHSFFileDir, CurSensorName, 'Color' + SelectTime + '.HSF') # color data
                            DataInfoIn.DepthHSFFileName = os.path.join(SrcDepthHSFFileDir, CurSensorName, 'Depth' + SelectTime + '.HSF') # depth data
                        if not os.path.exists(DataInfoIn.DepthHSFFileName):
                            print('depth file not exist {}'.format(DataInfoIn.ColorHSFFileName))
                        # save ply
                        CurSaveFolderName = os.path.join(SaveDestDepthHSFFileDir, CurSensorName, SelectTime)
                        CurSaveDepthFolderName = os.path.join(CurSaveFolderName, 'Gray')
                        CurSavePlyFolderName = os.path.join(SaveDestPlyHSFFileDir, CurSensorName, SelectTime)
                        if not os.path.exists(CurSavePlyFolderName):
                            os.makedirs(CurSavePlyFolderName)
                        CurSaveDepthFolderFileList = os.listdir(CurSaveDepthFolderName)
                        for CurdepthFileName in CurSaveDepthFolderFileList:
                            CurdepthFileFullName = os.path.join(CurSaveDepthFolderName, CurdepthFileName)
                            CurPlyFileFullName = os.path.join(CurSavePlyFolderName, CurdepthFileName.replace('.depth', '.ply'))
                            CurFrameDepth = ReadOneFrameDepthFile(CurdepthFileFullName, DepthWidth, DepthHeight) # ReadOneFrameDepthFile
                            CurFramePts = CurDepth2PointCloud.Transformation(CurFrameDepth) # depth2Pts, [Nx3]
                            SavePt3D2Ply(CurPlyFileFullName,CurFramePts)# save ply file

        
        