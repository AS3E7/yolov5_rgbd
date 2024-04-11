# -*- encoding: utf-8 -*-
# cython: language_level=3
'''
Time    : 2021/05/14 16:05
Author  : JiangZhuang
File    : PreprocessC.py
Software: Visual Studio Code
'''

import os
import sys
import ctypes
import numpy as np

from config import one_sensor_params

this_file_path = os.path.abspath(__file__)
if sys.platform == "win32":
    # _PreprocessC_lib_filename = 'PreprocessC_Python.dll'
    _PreprocessC_lib_filename = 'Python_c++.dll' # PKX
    
    _PreprocessC_lib_path = this_file_path[:-len(this_file_path.split('\\')[-1])] + _PreprocessC_lib_filename              
    _lib = ctypes.CDLL(_PreprocessC_lib_path)
#    _lib = ctypes.windll.LoadLibrary(_PreprocessC_lib_path)
    
elif sys.platform == "linux":
    try:
        _PreprocessC_lib_filename = 'PreprocessC_Python_arm64.so'
        print('linux_arm64')
    except OSError:
        _PreprocessC_lib_filename = 'PreprocessC_Python_x64.so'
        print('linux_x64')
    _PreprocessC_lib_path = this_file_path[:-len(this_file_path.split('/')[-1])] + _PreprocessC_lib_filename                                      
    _lib = ctypes.CDLL(_PreprocessC_lib_path)

depth_image_width = one_sensor_params['depth_width'] # 512 / 576
depth_image_height = one_sensor_params['depth_height'] # 424 / 640

#_now_Pts_Rot3D = np.random.random([depth_image_height*depth_image_width, 3]).astype(np.double) # [576*640, 3]
#_now_Pts_CalImageSubBG = np.random.random([depth_image_height*depth_image_width, 3]).astype(np.double)
# _now_Pts_ImageTransCalibHeight = np.random.random([depth_image_height*depth_image_width, 3]).astype(np.ubyte)


class _Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
    ]


_Point = ctypes.POINTER(_Point)


class _Bgpoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
        ("var", ctypes.c_double)
    ]


_Bgpoint = ctypes.POINTER(_Bgpoint)


class _Rgb(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_ubyte),
        ("g", ctypes.c_ubyte),
        ("b", ctypes.c_ubyte)
    ]


_Rgb = ctypes.POINTER(_Rgb)

#_c_Rot3D = _lib.c_Rot3D_d
# _c_Rot3D.restype = None
# _c_Rot3D.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(_Point), ctypes.POINTER(_Point), ctypes.c_int, ctypes.c_int)
#_c_CalImageSubBG = _lib.c_CalImageSubBG
#_c_CalImageSubBG.restype = None
#_c_CalImageSubBG.argtypes = (ctypes.POINTER(_Point), ctypes.POINTER(_Bgpoint), ctypes.c_int, ctypes.c_int, ctypes.POINTER(_Point), ctypes.c_int)
# _c_ImageTrans_CalibHeight = _lib.c_ImageTrans_CalibHeight
# _c_ImageTrans_CalibHeight.restype = None
# _c_ImageTrans_CalibHeight.argtypes = (ctypes.POINTER(_Point), ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(_Rgb), ctypes.c_int)


_c_Rot3D = _lib.c_Rot3D_d
_now_Pts_Rot3D = [0 for x in range(depth_image_height*depth_image_width*3)]
_now_Pts_Rot3D = (ctypes.c_double * len(_now_Pts_Rot3D))(*_now_Pts_Rot3D)

_c_CalImageSubBG = _lib.c_CalImageSubBG_d
_now_Pts_CalImageSubBG = [0 for x in range(depth_image_height*depth_image_width*3)]
_now_Pts_CalImageSubBG = (ctypes.c_double * len(_now_Pts_CalImageSubBG))(*_now_Pts_CalImageSubBG)

_c_ImageTrans_CalibHeight = _lib.c_ImageTrans_CalibHeight_d
_now_Pts_ImageTransCalibHeight = [0 for x in range(depth_image_height*depth_image_width*3)]
_now_Pts_ImageTransCalibHeight = (ctypes.c_ubyte * len(_now_Pts_ImageTransCalibHeight))(*_now_Pts_ImageTransCalibHeight)

def Rot3D(H: np.ndarray, Pts: np.ndarray, Mode: int=1) -> np.ndarray:
    '''
    input:
        H(array): 4x4旋转矩阵
        Pts(array): Nx3点云数据

    output:
        now_Pts_Rot3D(array): Nx3旋转后的点云数据
    '''
    if Pts.shape[1] != 3:
        Pts = Pts.T
    if Pts.shape[1] != 3:
        raise ValueError('Pts.shape != [Nx3 or 3xN]')

    Pts[0,:] = [9,9,9] # [N x 3]
    # _c_Rot3D(H.reshape([1, 16])[0].ctypes.data_as(ctypes.POINTER(ctypes.c_double)), Pts.ctypes.data_as(ctypes.POINTER(_Point)), _now_Pts_Rot3D.ctypes.data_as(ctypes.POINTER(_Point)), Mode, Pts.shape[0])
    
    _c_Rot3D(H.astype(np.double).ctypes, Pts.astype(np.double).ctypes, _now_Pts_Rot3D, Mode, Pts.shape[0])
    result = np.array(_now_Pts_Rot3D, dtype=np.double).reshape([Pts.shape[0], Pts.shape[1]])
    return result


def CalImageSubBG(SrcPts: np.ndarray, BGInfo: np.ndarray, Mode: int=1, FGDistK: int = 1) -> np.ndarray:
    '''
    input:
        SrcPts(array): Nx3原始点云数据
        BGInfo(array): Nx4背景点云数据
        *FGDistK(int): 距离方差系数

    output:
        now_Pts_CalImageSubBG(array): Nx3前景点云数据
    '''
    if SrcPts.shape[1] != 3:
        SrcPts = SrcPts.T
    if SrcPts.shape[1] != 3:
        raise ValueError('SrcPts.shape != [Nx3 or 3xN]')
    if BGInfo.shape[1] != 4:
        BGInfo = BGInfo.T
    if BGInfo.shape[1] != 4:
        raise ValueError('BGInfo.shape != [Nx4 or 4xN]')

    SrcPts[0,:] = [9,9,9] # [N x 3]
    BGInfo[0,:] = [9,9,9,9] # [N x 4]
#    _c_CalImageSubBG(SrcPts.ctypes.data_as(ctypes.POINTER(_Point)), BGInfo.ctypes.data_as(ctypes.POINTER(_Bgpoint)), Mode, FGDistK, _now_Pts_CalImageSubBG.ctypes.data_as(ctypes.POINTER(_Point)), SrcPts.shape[0])
    
    _c_CalImageSubBG(SrcPts.astype(np.double).ctypes, BGInfo.astype(np.double).ctypes, Mode, FGDistK, _now_Pts_CalImageSubBG, SrcPts.shape[0])
    result = np.array(_now_Pts_CalImageSubBG, dtype=np.double).reshape([SrcPts.shape[0], SrcPts.shape[1]])

    return result


def ImageTransCalibHeight(Pts: np.ndarray, HeightMin: int, HeightMax: int, Mode: int=1, ImageHeigh: int = 424, ImageWidth: int = 512) -> np.ndarray:
    '''
    input:
        Pts(array): Nx3点云数据
        HeightMin(int): 高度最小值
        HeightMax(int): 高度最大值
        *ImageHeigh(int): 生成图像高度
        *ImageWidth（int): 生成图像宽度

    output:
        now_image(array): 转换后的图片
    '''
    if Pts.shape[1] != 3:
        Pts = Pts.T
    if Pts.shape[1] != 3:
        raise ValueError('Pts.shape != [Nx3 or 3xN]')
    if HeightMin > HeightMax:
        raise ValueError('HeightMin > HeightMax')
        
    Pts[0,:] = [9,9,9] # [N x 3]
    print('Pts size = ', Pts.shape)
    print(HeightMin, HeightMax)
    
    # _c_ImageTrans_CalibHeight(Pts.ctypes.data_as(ctypes.POINTER(_Point)), HeightMin, HeightMax, Mode, _now_Pts_ImageTransCalibHeight.ctypes.data_as(ctypes.POINTER(_Rgb)), depth_image_width*depth_image_height)
    # now_image = _now_Pts_ImageTransCalibHeight.reshape([ImageHeigh, ImageWidth, 3])
    # return now_image
    
    _c_ImageTrans_CalibHeight(Pts.astype(np.double).ctypes, ctypes.c_double(HeightMin), ctypes.c_double(HeightMax), Mode, _now_Pts_ImageTransCalibHeight, depth_image_width*depth_image_height)
    result = np.array(_now_Pts_ImageTransCalibHeight, dtype=np.ubyte).reshape([ImageHeigh, ImageWidth, 3])

    return result


if __name__ == "__main__":
    import cv2
    from Python_src import CalBGInfo
    import PointCloudFunsC

    H = np.array([[0.999906, 0.000240306, -0.0136735, 0],
                  [5.42152e-20, -0.999846, -0.0175718, 0],
                  [-0.0136756, 0.0175702, -0.999752, 3.59298],
                  [0, 0, 0, 1]], dtype=np.double)
    Pts = PointCloudFunsC.ReadPointCloudFromFile('Depth2019-02-21-105212_1516.ply').astype(np.double)

    Pts_ground = Rot3D(H, Pts)
    print(Pts_ground)

    bgPts = PointCloudFunsC.ReadPointCloudFromFile('bg_K2.ply').astype(np.double)
    bgPts = CalBGInfo(bgPts.T)
    Pts_foreground = CalImageSubBG(Pts_ground, bgPts)
    print(Pts_foreground)

    HeightMin = -250.0
    HeightMax = 2500.0
    image = ImageTransCalibHeight(Pts_ground, HeightMin, HeightMax)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
