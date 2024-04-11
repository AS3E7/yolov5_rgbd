# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:55:50 2019

@author: HYD
"""

import numpy as np
import os

#import HSFFuns
import CloudPointFuns

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
    
def TransMultiDepth2Pts(DepthFolderName, SavePlyFolderName, SensorInnerParam = None, DepthWidth = 512, DepthHeight = 424):
    """
    功能：深度数据转换为点云数据
    输入：
        DepthFolderName: 深度数据文件夹
        SavePlyFolderName: ply保存文件夹
        SensorInnerParam: 转换参数
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
#    # loop label file
#    for root, dirs, files in os.walk(DepthFolderName):
#        for image_file in files:
#            if image_file.endswith('.depth'):
#                cur_depth_file_name = image_file
#            else:
#                continue
#            print('  {}'.format(cur_depth_file_name))
#            # cur save file name
#            cur_save_file_name = os.path.join(SavePlyFolderName, cur_depth_file_name.replace('.depth','.ply'))
#            # read depth
#            FileName = os.path.join(DepthFolderName, cur_depth_file_name)
#            Depth = HSFFuns.ReadOneFrameDepthFile(FileName, DepthWidth, DepthHeight)
#            # depth2pts
#            Pts = TransDepth2Pts(Depth, DepthWidth, DepthHeight, SensorInnerParam) # [217088 x 3]
#            # save pts as ply file
#            CloudPointFuns.SavePt3D2Ply(cur_save_file_name, Pts)
            
            
if __name__ == '__main__':
    print('Test Depth2Pts Function.')
            
    TestDepth2PtsFlag = 1 # 测试深度数据转换点云函数
    
    # sensor info
    DepthWidth = 512
    DepthHeight = 424
    # test
    if TestDepth2PtsFlag == 1:
        SelectSensorName = '5'
        
        if SelectSensorName == '5':
            DepthFileName = r'X:\PlyData\LGDAT_2019\5\Depth2019-09-16-150000\Depth2019-09-16-150000_1568617200177_00002.depth'
            SavePlyFileName = r'X:\DepthData\LGDAT_2019\LGTest\Select\20200506\BG\5\bg.ply'
        elif SelectSensorName == '6':
            DepthFileName = r'X:\PlyData\LGDAT_2019\6\Depth2019-09-16-150000\Depth2019-09-16-150000_1568617200252_00002.depth'
            SavePlyFileName = r'X:\DepthData\LGDAT_2019\LGTest\Select\20200506\BG\6\bg.ply'  
        elif SelectSensorName == '7':
            DepthFileName = r'X:\PlyData\LGDAT_2019\7\Depth2019-09-16-150000\Depth2019-09-16-150000_1568617234498_00075.depth'
            SavePlyFileName = r'X:\DepthData\LGDAT_2019\LGTest\Select\20200506\BG\7\bg.ply'  
        elif SelectSensorName == '8':
            DepthFileName = r'X:\PlyData\LGDAT_2019\8\Depth2019-09-16-150000\Depth2019-09-16-150000_1568617200564_00002.depth'
            SavePlyFileName = r'X:\DepthData\LGDAT_2019\LGTest\Select\20200506\BG\8\bg.ply'  
            
        SavePlyName = os.path.basename(DepthFileName)
        SavePlyFileName = os.path.join(SavePlyFileName)
        # params, [离线数据深度转点云默认参数 <-- 周果]
        ycenterF= 211.190796
        xcenterF= 256.133209
        VfoclenInPixelsF= 366.382507
        HfoclenInPixelsF= 366.382507

#        ycenterF= 256.133209
#        xcenterF= 211.190796
#        VfoclenInPixelsF= 366.382507
#        HfoclenInPixelsF= 366.382507
        
        # param
        Param = SensorInnerParam()
        Param.cx = xcenterF
        Param.cy = ycenterF
        Param.fx = HfoclenInPixelsF
        Param.fy = VfoclenInPixelsF
        SensorInnerParam = Param
        # read depth data
        with open(DepthFileName, 'rb') as fid:
            data_array = np.fromfile(fid, np.int16)
        Depth = np.reshape(data_array, [DepthHeight, DepthWidth])
#        Depth = np.reshape(data_array, [DepthWidth, DepthHeight]).transpose()
        # depth2pts
        Pts = TransDepth2Pts(Depth, DepthWidth, DepthHeight, SensorInnerParam) # [217088 x 3]
        # 转换坐标（横纵方向）
        PtsTransIdxSrc = np.array(range(0,DepthWidth*DepthHeight))
        PtsTransIdx = np.reshape(PtsTransIdxSrc, [DepthWidth, DepthHeight]).transpose().flatten()
        Pts = Pts[PtsTransIdx,:]
        # save pts
        CloudPointFuns.SavePt3D2Ply(SavePlyFileName, Pts, 'XYZ')
        
#        import matplotlib.pyplot as plt
#        plt.imshow(Depth)
        
    print('End.')
        
        