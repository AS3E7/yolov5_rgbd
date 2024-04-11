# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:22:01 2021

@author: Administrator
"""

import os
import numpy as np
import Depth2Pts
import HSFFuns
from CloudPointFuns import Rot3D
from PointCloudFunsC import PointCloudFunsC

def depthFileToColormapFile(SrcDepthFolderName, DestColormapFolderName, FilePostfixGroupName=None, DepthWidth = 512, DepthHeight = 424, CalibHFolderName=None, TransDepthRange=[-250, 2500], TransFormat='Colormap', Depth2PtsMethod=1):
    """
    功能：对depth文件转为Colormap图像文件
    """
    # 点云转换方法： Depth2PtsMethod=1，python方法；  Depth2PtsMethod=2，C++函数方法；  
    if Depth2PtsMethod == 2:
        if DepthWidth == 640:
            ycenterF= 240.0
            xcenterF= 320.0
            VfoclenInPixelsF= 576 # 576
            HfoclenInPixelsF= 576 # 576
            CurDepth2PointCloud = PointCloudFunsC.Depth2PointCloud(DepthWidth, DepthHeight, xcenterF, ycenterF, HfoclenInPixelsF, VfoclenInPixelsF)
        elif DepthWidth == 320:
            ycenterF= 120.0
            xcenterF= 160.0 # 160
            VfoclenInPixelsF= 576/2 # 576
            HfoclenInPixelsF= 576/2 # 576
            CurDepth2PointCloud = PointCloudFunsC.Depth2PointCloud(DepthWidth, DepthHeight, xcenterF, ycenterF, HfoclenInPixelsF, VfoclenInPixelsF)

        else:
            print('DepthWidth error ...')
     
    if not FilePostfixGroupName is None:
        SrcFolderNamePostfixName = FilePostfixGroupName[0]
        DestFolderNamePostfixName = FilePostfixGroupName[1]
    else:
        SrcFolderNamePostfixName = ''
        DestFolderNamePostfixName = ''
    # 遍历文件名
    for rt, dirs, files in os.walk(SrcDepthFolderName):
        for f in files:
            if CalibHFolderName == None: # 不使用 CalibH 
                if len(SrcFolderNamePostfixName)==0:
                    CurdepthFileFullName = os.path.join(SrcDepthFolderName, f)
                    DestFileName = os.path.join(DestColormapFolderName, f)
                    # read depth file
                    CurFrameDepth = HSFFuns.ReadOneFrameDepthFile(CurdepthFileFullName, DepthWidth, DepthHeight)
                    # generate and save colormap
                    # depthToColormap(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
                    depthToColormapNoCalib(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
                    
                if len(SrcFolderNamePostfixName)>0 and f.endswith(SrcFolderNamePostfixName):
                    CurdepthFileFullName = os.path.join(SrcDepthFolderName, f)
                    DestFileName = os.path.join(DestColormapFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName))
                    # read depth file
                    CurFrameDepth = HSFFuns.ReadOneFrameDepthFile(CurdepthFileFullName, DepthWidth, DepthHeight)
                    # generate and save colormap
                    depthToColormap(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
                    # depthToColormapNoCalib(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
                    
            else: # 使用 CalibH
                # 读取params 参数信息
                ParamsData = dict()
                ParamsFile = os.listdir(CalibHFolderName)
                for i_file in ParamsFile:
                    # 读取文件
                    CurParamsFile = os.path.join(CalibHFolderName, i_file)
                    CurParamsData = HSFFuns.ReadCalibParamsInfo(CurParamsFile)
                    ParamsData[i_file.split('_')[-1].split('.')[0]] = CurParamsData

                if len(SrcFolderNamePostfixName)==0:
                    CurdepthFileFullName = os.path.join(SrcDepthFolderName, f)
                    DestFileName = os.path.join(DestColormapFolderName, f)
                    # read depth file
                    CurFrameDepth = HSFFuns.ReadOneFrameDepthFile(CurdepthFileFullName, DepthWidth, DepthHeight)
                    # cur sensor name
                    CurSensorName = f.split('_')[0]
                    CurSensorH = ParamsData[CurSensorName]
                    if Depth2PtsMethod == 1:
                        CurFramePts = depth2Pts(CurFrameDepth, DepthWidth=DepthWidth, DepthHeight=DepthHeight)
                        CurFramePtsNew = Rot3D(CurSensorH, CurFramePts.transpose())
                        CurFramePtsNewZ = CurFramePtsNew[2,:]*1000
                        CurFrameDepth = CurFramePtsNewZ.reshape([DepthWidth, DepthHeight]).transpose()
                    elif Depth2PtsMethod == 2:
                        CurFramePts = CurDepth2PointCloud.Transformation(CurFrameDepth) # depth2Pts, [Nx3]
                        CurFramePtsNew = Rot3D(CurSensorH, CurFramePts.transpose())
                        CurFramePtsNewZ = CurFramePtsNew[2,:]*1000
                        CurFrameDepth = CurFramePtsNewZ.reshape([DepthHeight, DepthWidth])
                    
                    # generate and save colormap
                    depthToColormap(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
                if len(SrcFolderNamePostfixName)>0 and f.endswith(SrcFolderNamePostfixName):
                    CurdepthFileFullName = os.path.join(SrcDepthFolderName, f)
                    DestFileName = os.path.join(DestColormapFolderName, f.replace(SrcFolderNamePostfixName, DestFolderNamePostfixName))
                    # read depth file
                    CurFrameDepth = HSFFuns.ReadOneFrameDepthFile(CurdepthFileFullName, DepthWidth, DepthHeight)
                    # cur sensor name
                    CurSensorName = f.split('_')[0]
                    CurSensorH = ParamsData[CurSensorName]
                    if Depth2PtsMethod == 1:
                        CurFramePts = depth2Pts(CurFrameDepth, DepthWidth=DepthWidth, DepthHeight=DepthHeight)
                        CurFramePtsNew = Rot3D(CurSensorH, CurFramePts.transpose())
                        CurFramePtsNewZ = CurFramePtsNew[2,:]*1000
                        CurFrameDepth = CurFramePtsNewZ.reshape([DepthWidth, DepthHeight]).transpose()
                    elif Depth2PtsMethod == 2:
                        CurFramePts = CurDepth2PointCloud.Transformation(CurFrameDepth) # depth2Pts, [Nx3]
                        CurFramePtsNew = Rot3D(CurSensorH, CurFramePts.transpose())
                        CurFramePtsNewZ = CurFramePtsNew[2,:]*1000
                        CurFrameDepth = CurFramePtsNewZ.reshape([DepthHeight, DepthWidth])
                    
                    # generate and save colormap
                    depthToColormap(CurFrameDepth, DestFileName, TransDepthRange=TransDepthRange, TransFormat=TransFormat)
    
    return 0

def depthToColormap(CurFrameDepth, SaveColormapFileFullName, DepthWidth = 512, DepthHeight = 424, TransDepthRange=[-250, 2500], TransFormat='Colormap'):
    """
    功能：depth转Colormap
    """
    # TransDepthRangeInput
    # TransDepthRange = [-250, 2500] # [unit: mm]
    TransDepthRangeInput = TransDepthRange
    
    DepthData = -CurFrameDepth
    # TransDepthRange
    TransDepthRange = []
    TransDepthRange.append(-TransDepthRangeInput[1])
    TransDepthRange.append(-TransDepthRangeInput[0])
    # save as colormap
    # HSFFuns.GenImageFromDepth(DepthData, TransDepthRange, SaveColormapFileFullName, TransFormat='Colormap') # [424,512,3]
    HSFFuns.GenImageFromDepth(DepthData, TransDepthRange, SaveColormapFileFullName, TransFormat=TransFormat) # [424,512,3]
    return 0

def depthToColormapNoCalib(CurFrameDepth, SaveColormapFileFullName, DepthWidth = 512, DepthHeight = 424, TransDepthRange=[-250, 2500], TransFormat='Colormap'):
    """
    功能：depth转Colormap
    """
    # TransDepthRangeInput
    TransDepthRangeInput = TransDepthRange
    
    DepthData = CurFrameDepth
    # TransDepthRange
    TransDepthRange = []
    TransDepthRange.append(TransDepthRangeInput[0])
    TransDepthRange.append(TransDepthRangeInput[1])
    # save as colormap
    # HSFFuns.GenImageFromDepth(DepthData, TransDepthRange, SaveColormapFileFullName, TransFormat='Colormap') # [424,512,3]
    HSFFuns.GenImageFromDepth(DepthData, TransDepthRange, SaveColormapFileFullName, TransFormat=TransFormat) # [424,512,3]
    return 0

def depth2Pts(Depth, DepthWidth=512, DepthHeight=424):
    """
    功能：depth 转到 点云 pts
    """
    # param
    if DepthWidth == 512: # [512 424]
        ycenterF= 202.546464
        xcenterF= 256.685317
        VfoclenInPixelsF= 368.901024
        HfoclenInPixelsF= 368.874187
    elif DepthWidth == 640: # [640 480]
        ycenterF= 240.0
        xcenterF= 320.0
        VfoclenInPixelsF= 576 # 576
        HfoclenInPixelsF= 576 # 576
    else:
        ycenterF= 202.546464
        xcenterF= 256.685317
        VfoclenInPixelsF= 368.901024
        HfoclenInPixelsF= 368.874187
        
    # param
    Param = Depth2Pts.SensorInnerParam()
    Param.cx = xcenterF
    Param.cy = ycenterF
    Param.fx = HfoclenInPixelsF
    Param.fy = VfoclenInPixelsF
    SensorInnerParam = Param
    Pts = Depth2Pts.TransDepth2Pts(Depth, DepthWidth, DepthHeight, SensorInnerParam) # [217088 x 3]
    return Pts



if __name__ == '__main__':
    print('Start')
    
    TestCase = 1
    
    if TestCase == 1:
        # 目标 depth 文件夹地址
        DestdepthFolderGroupName = r'Y:\ToYuJinyong\FromXbiao\JCDete\Data\SelectData_20211123\depth'
        # 目标 Colormap 文件夹地址
        DestColormapFolderGroupName = r'Y:\ToYuJinyong\FromXbiao\JCDete\Data\SelectData_20211123\Colormap'
        # CalibHFolderName
        Selectdepth2ColormapParamsFolderName = r'Y:\ToYuJinyong\FromXbiao\Data\HSF\Calib'
        
        # depthFileToColormapFile
        depthFileToColormapFile(DestdepthFolderGroupName, DestColormapFolderGroupName, FilePostfixGroupName=['.depth', '.png'], CalibHFolderName=Selectdepth2ColormapParamsFolderName)
    
    
    print('End')

