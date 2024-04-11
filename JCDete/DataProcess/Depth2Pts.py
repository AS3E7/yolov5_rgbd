# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:55:50 2019

@author: HYD
"""

import numpy as np
import os

import HSFFuns
import CloudPointFuns

from PointCloudFunsC import PointCloudFunsC

class SensorInnerParam():
    """
    功能：传感器内部参数结构体
        [fctor, cx, cy, fx, fy, k1, k2, k3, p1,p2]
    """
    def __init__(self):
        self.cx = 0
        self.cy = 0
        self.fx = 0
        self.fy = 0
        self.fctor = 1000 # [mm]

def TransDepth2Pts(Depth, ImageWidth, ImageHeight, Param):
    """
    功能：深度数据转换为点云数据
    输入：
        Depth: 深度数据, [424 x 512]
        ImageWidth, ImageHeight: 深度图像尺寸
        Param: 转换参数
    输出：
        Pts: 点云数据, [217088 x 3]
    """
    # x
    xv = np.linspace(0,ImageWidth-1,ImageWidth)
    xv = xv - Param.cx
    xv = np.tile(xv, (ImageHeight, 1))
    
    # y
    yv = np.linspace(0,ImageHeight-1,ImageHeight)
    yv = yv - Param.cy
    yv = np.tile(yv, (ImageWidth, 1))
    yv = yv.transpose()
    
    # z
    zw = Depth.transpose()
    
    # trans
    xwNew = xv.transpose()
    ywNew = yv.transpose()
    
    xwNew = (zw*xwNew)/Param.fx
    ywNew = (zw*ywNew)/Param.fy
    zwNew = zw
    ywNew = -ywNew
    
    xwNew = np.reshape(xwNew,(xwNew.shape[0]*xwNew.shape[1]))
    ywNew = np.reshape(ywNew,(ywNew.shape[0]*ywNew.shape[1]))
    zwNew = np.reshape(zwNew,(zwNew.shape[0]*zwNew.shape[1]))
    
    Pts = np.stack((xwNew, ywNew, zwNew),1)
    Pts = Pts/Param.fctor
    
    return Pts
    
def TransMultiDepth2Pts(DepthFolderName, SavePlyFolderName, SensorInnerParam = None, DepthWidth = 512, DepthHeight = 424, Depth2PtsMethod=1):
    """
    功能：深度数据转换为点云数据
    输入：
        DepthFolderName: 深度数据文件夹
        SavePlyFolderName: ply保存文件夹
        SensorInnerParam: 转换参数
        Depth2PtsMethod: 
            Depth2PtsMethod = 1,使用 python函数 对depth转换为ply文件
            Depth2PtsMethod = 2,使用 C 对depth转换为ply文件
    输出：
        ply 文件
    """
    # SensorInnerParam
    if SensorInnerParam == None:
        # param
        ycenterF= 202.546464
        xcenterF= 256.685317
        VfoclenInPixelsF= 368.901024
        HfoclenInPixelsF= 368.874187
        # param
        Param = SensorInnerParam()
        Param.cx = xcenterF
        Param.cy = ycenterF
        Param.fx = HfoclenInPixelsF
        Param.fy = VfoclenInPixelsF
        SensorInnerParam = Param
        
    # 初始化Depth2PtsMethod=2 中转换函数的类
    if Depth2PtsMethod == 2:
        CurDepth2PointCloud = PointCloudFunsC.Depth2PointCloud(DepthWidth, DepthHeight, SensorInnerParam.cx, SensorInnerParam.cy, SensorInnerParam.fx, SensorInnerParam.fy)
        
    # loop label file
    for root, dirs, files in os.walk(DepthFolderName):
        for image_file in files:
            if image_file.endswith('.depth'):
                cur_depth_file_name = image_file
            else:
                continue
            print('  {}'.format(cur_depth_file_name))
            # cur save file name
            cur_save_file_name = os.path.join(SavePlyFolderName, cur_depth_file_name.replace('.depth','.ply'))
            # read depth
            FileName = os.path.join(DepthFolderName, cur_depth_file_name)
            Depth = HSFFuns.ReadOneFrameDepthFile(FileName, DepthWidth, DepthHeight)
            # depth2pts
            if Depth2PtsMethod == 1:
                Pts = TransDepth2Pts(Depth, DepthWidth, DepthHeight, SensorInnerParam) # [217088 x 3]
            elif Depth2PtsMethod == 2:
                Pts = CurDepth2PointCloud.Transformation(Depth) # [Nx3]
            # save pts as ply file
            CloudPointFuns.SavePt3D2Ply(cur_save_file_name, Pts)
            
            
if __name__ == '__main__':
    print('Start.')
            
    TestDepth2PtsFlag = 1 # 使用 python函数 对depth转换为ply文件
                          # 使用 C函数 对depth转换为ply文件，JZ
    
    if TestDepth2PtsFlag == 1: # 使用 python函数 对depth转换为ply文件
        # SrcDepthFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Data\NanShanDrugTreatment\CalibMultiSensorToolCode_GenPly\Data\NewLGData_A102_GenPly'
        # DestPlyFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Data\NanShanDrugTreatment\CalibMultiSensorToolCode_GenPly\Data\NewLGData_A102_GenPly'
        
        # SrcDepthFolderName = r'D:\xiongbiao\Data\HumanDete\ZT\RGBD_20210720\SelectHardCaseImage\depth'
        # DestPlyFolderName = r'D:\xiongbiao\Data\HumanDete\ZT\RGBD_20210720\SelectHardCaseImage\Ply'
        
        
        SrcDepthFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\SelectHardCaseImage\Calib\CalibPlyData\249'
        DestPlyFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\SelectHardCaseImage\Calib\CalibPlyData\249'
        

        # SensorInnerParam
        ycenterF= 202.546464
        xcenterF= 256.685317
        VfoclenInPixelsF= 368.901024
        HfoclenInPixelsF= 368.874187
        # param
        Param = SensorInnerParam()
        Param.cx = xcenterF
        Param.cy = ycenterF
        Param.fx = HfoclenInPixelsF
        Param.fy = VfoclenInPixelsF
        SensorInnerParam = Param
        # sensor size
        DepthWidth = 512
        DepthHeight = 424
        # TransMultiDepth2Pts
        CurDepthFileFullName = SrcDepthFolderName
        CurSavePlyFileFullName = DestPlyFolderName
        TransMultiDepth2Pts(CurDepthFileFullName, CurSavePlyFileFullName, SensorInnerParam, DepthWidth, DepthHeight)
        
        # 遍历文件名
#        for rt, dirs, files in os.walk(SrcDepthFolderName):
#            for f in files:
#                if f.endswith('.depth'):
#                    print(f)
#                    CurDepthFileFullName = os.path.join(rt, f)
#                    CurSavePlyFileFullName = os.path.join(rt, f.replace('.depth', '.ply'))
#                    # depth to pts
#                    TransMultiDepth2Pts(CurDepthFileFullName, CurSavePlyFileFullName, SensorInnerParam, DepthWidth, DepthHeight)
                
            
    elif TestDepth2PtsFlag == 2: # 使用 C函数 对depth转换为ply文件
        Depth2PtsMethod = 2
        
        # SrcDepthFolderName = r'D:\xiongbiao\Code\LGPoseDete\RGBDHumanAnalysis\Train\data\test_data\test_ply\15\depth'
        # DestPlyFolderName = r'D:\xiongbiao\Code\LGPoseDete\RGBDHumanAnalysis\Train\data\test_data\test_ply\15\Ply_C'
        
        SrcDepthFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\SelectHardCaseImage\Calib\CalibPlyData\245C'
        DestPlyFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211103\SelectHardCaseImage\Calib\CalibPlyData\245C'
        

        # SensorInnerParam
        ycenterF= 202.546464
        xcenterF= 256.685317
        VfoclenInPixelsF= 368.901024
        HfoclenInPixelsF= 368.874187
        # param
        Param = SensorInnerParam()
        Param.cx = xcenterF
        Param.cy = ycenterF
        Param.fx = HfoclenInPixelsF
        Param.fy = VfoclenInPixelsF
        SensorInnerParam = Param
        # sensor size
        DepthWidth = 512
        DepthHeight = 424
        # TransMultiDepth2Pts
        TransMultiDepth2Pts(SrcDepthFolderName, DestPlyFolderName, SensorInnerParam, DepthWidth, DepthHeight, Depth2PtsMethod=Depth2PtsMethod)


            
    print('End.')
            