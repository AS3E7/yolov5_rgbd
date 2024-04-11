# -*- coding: utf-8 -*-
# cython: language_level=3
"""
author: JiangZhuang
build: 2020/08/03
"""

import numpy as np
import PointCloudFunsC
import time

from CloudPointFuns import SavePt3D2Ply

if __name__ == '__main__':
    # depth = np.fromfile("depth1.bin", dtype=np.uint16)
    # depth = np.reshape(depth, [480, 640])
    depth = np.random.randint(0, 8000, [480, 640], dtype=np.uint16)

    #-----------点云转换函数调用------------------
    Depth2Pts = PointCloudFunsC.Depth2PointCloud(depth.shape[0], depth.shape[1], 310.46990967, 233.02980042, 577.58898926, 577.58898926)
    begin_time = time.time()
    Pts = Depth2Pts.Transformation(depth)
    print("Transformation Point Used Time: %.06f" % (time.time() - begin_time))

    # -----------点云保存函数调用------------------
    begin_time = time.time()
    PointCloudFunsC.SavePointCloudToPly("test.ply", Pts)
    print("Save Point Cloud Used Time: %.06f" % (time.time() - begin_time))

    #-----------点云读取函数调用
    begin_time = time.time()
    PointCloud = PointCloudFunsC.ReadPointCloudFromPly("test.ply")
    print("Read Point Cloud Used Time: %.06f" % (time.time() - begin_time))
    
    #-----------点云读取函数调用 SavePt3D2Ply
    begin_time = time.time()
    SavePt3D2Ply('test2.ply', Pts, 'XYZ')
    print("SavePt3D2Ply Save Point Cloud Used Time: %.06f" % (time.time() - begin_time))

