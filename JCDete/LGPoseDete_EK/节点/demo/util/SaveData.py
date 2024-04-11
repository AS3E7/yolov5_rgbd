# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 17:39:57 2020

@author: HYD
"""

import numpy as np
import os
import time

class LogCategory():
    """
    保存各类告警记录，告警类别序号
        0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4=打印各类告警检测信息
    """
    def __init__(self):
        self.BasicProceInfo = 0 # 打印基本流程信息
        self.FuncInputOutputInfo = 0 # 打印接口输入输出信息
        self.DeteObjInfo = 0 # 打印目标检测信息
        self.CluObjInfo = 0 # 打印聚类检测信息
        self.AlarmInfo = 0 # 打印各类告警检测信息
        
class AlarmCategory():
    """
    各功能告警序号：
    其中：#告警id	"warningTypeModel":[
        [1,"未在指定时间休息"], -
        [2,"未在制定区域监督"], +
        [4,"厕所区域异常"], +
        [8,"窗户区域异常"], +
        [16,"高度异常"], +
        [32,"非休息时间休息"], -
        [64,"进入三角区域"], -
        [128,"内务不整"], -
        [512,"单人留仓"], +
        [1024,"吊拉窗户"], +
        [2048,"搭人梯"], +
        [4096,"站在被子上做板报"] +
        ]
    """
    def __init__(self):
        self.BED = 0 
        self.INTERNALSUPERVISOR = 0 
        self.TOILET = 0 
        self.WINDOW = 0 
        self.ABOVEHEIGHTTHOD = 0
        self.WRONGTIMELIE = 0 
        self.INTRIANGLEREGION = 0 
        self.HOUSEKEEP = 0 
        self.ALONESTAY = 0 
        self.HAULWINDOW = 0 
        self.BUILDLADDER = 0 
        self.STANDQUILT = 0 
        
def LogSaveCategoryIndexFun():
    """
    功能：log 信息保存类别序号
    """
    LogSaveCategoryIndex = LogCategory()
    LogSaveCategoryIndex.BasicProceInfo = 0 # 打印基本流程信息
    LogSaveCategoryIndex.FuncInputOutputInfo = 1 # 打印接口输入输出信息
    LogSaveCategoryIndex.DeteObjInfo = 2 # 打印目标检测信息
    LogSaveCategoryIndex.CluObjInfo = 3 # 打印聚类检测信息
    LogSaveCategoryIndex.AlarmInfo = 4 # 打印各类告警检测信息
    return LogSaveCategoryIndex
    
def AlarmCategoryIndexFun():
    """
    功能：log 信息保存类别序号
    """
    AlarmCategoryIndex = AlarmCategory()
    AlarmCategoryIndex.BED = 0 
    AlarmCategoryIndex.INTERNALSUPERVISOR = 1 
    AlarmCategoryIndex.TOILET = 2 
    AlarmCategoryIndex.WINDOW = 3 
    AlarmCategoryIndex.ABOVEHEIGHTTHOD = 4
    AlarmCategoryIndex.WRONGTIMELIE = 5 
    AlarmCategoryIndex.INTRIANGLEREGION = 6 
    AlarmCategoryIndex.HOUSEKEEP = 7 
    AlarmCategoryIndex.ALONESTAY = 8 
    AlarmCategoryIndex.HAULWINDOW = 9 
    AlarmCategoryIndex.BUILDLADDER = 10 
    AlarmCategoryIndex.STANDQUILT = 11 
    return AlarmCategoryIndex
        

def SavePtsAsDepth(SrcPts, SaveFileName):
    """
    功能：点云数据保存为depth数据文件
    输入： 
        SrcPts: [3 x N]
    """
    DepthScale = 1000 # mm-->m
    # pts to depth
    CurDepth = SrcPts[2, :] # Z 方向
    CurDepth = CurDepth * DepthScale
    CurDepth = CurDepth.astype(np.int16)
    # save file
    fpDepth = open(SaveFileName, 'wb')
    fpDepth.write(CurDepth)
    fpDepth.close()

    return 0

def ReadDepthFile(FileName, DepthWidth = 512, DepthHeight = 424):
    """
    功能：读取depth 文件
    """
    # ImgSize
    ImgSize = DepthWidth * DepthHeight
    DataPtsChannelSize = 1
    # read depth file
    fp = open(FileName)
    CurDepth = np.fromfile(fp, np.int16, count=ImgSize * DataPtsChannelSize)
    fp.close()
        
    return CurDepth

def DeleteOutLimitFiles(FolderName, MaxLimitSize, PostfixName = None, DeleteMaxFilesRatio = 0.1):
    """
    功能：删除超出大小范围的数据
    输入：
        MaxLimitSize：最大文件夹大小
        DeleteFilesRatio：每次删除最大容量文件比例
    """
    # 计算文件夹大小
    if PostfixName == None:
        FolderNameFilesSize = FileSize(FolderName)
    elif PostfixName == 'depth':
        CurFolderFileNum = len(os.listdir(FolderName))
        FolderNameFilesSize = 424 * 1024 * CurFolderFileNum # Byte, 424kb
    elif PostfixName == 'ply':
        CurFolderFileNum = len(os.listdir(FolderName))
        FolderNameFilesSize = 3.9 * 1024 * 1024 * CurFolderFileNum # Byte, 3.9MB
    else:
        FolderNameFilesSize = FileSize(FolderName)
        
    # 判断文件是否超出范围
    if FolderNameFilesSize > MaxLimitSize:
        # 文件列表
        DirList_Src = os.listdir(FolderName) # 默认按文件名排序
        # os.path.getmtime() 函数是获取文件最后修改时间; os.path.getctime() 函数是获取文件最后创建时间
        DirList = sorted(DirList_Src,key=lambda x: os.path.getmtime(os.path.join(FolderName, x))) # 按修改时间排序
#        DirList = DirList_Src
        
        # 删除部分文件
        #   按文件个数删除文件
        OutLimitFileNum = len(DirList) - int(len(DirList)*MaxLimitSize/FolderNameFilesSize) # 超过文件夹大小的数据个数，【默认各文件大小相近】
        LimitFileDeleteRatioNum = max(1, int(DeleteMaxFilesRatio*len(DirList)*MaxLimitSize/FolderNameFilesSize))
        DeleteFilesNum = OutLimitFileNum + LimitFileDeleteRatioNum
        for i_file in range(DeleteFilesNum):
            CurDeleteFileName = os.path.join(FolderName, DirList[i_file])
            os.remove(CurDeleteFileName) # 删除文件
        
    return 0
    


def FileSize(path):
    """
    功能：计算文件夹大小(单位 Byte)
    """   
    size = 0
    for root, dirs, files in os.walk(path):
        #目录下文件大小累加
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files]) 
    return size


    
if __name__ == '__main__':
    print('Start.')
    
    TestDeleteOutLimitFilesFlag = 1
    
    if TestDeleteOutLimitFilesFlag ==1:
#        TestFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\result/depth'

#        TestFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\result/ply'
        TestFolderName = r'Z:\PlyData\NSDAT\171\Depth2018-12-06-143000'

        MaxLimitSize = 25000000*1024*1024 # 单位：Byte
        
        t1 = time.time()
#        DeleteOutLimitFiles(TestFolderName, MaxLimitSize, PostfixName = 'depth')
        
        DeleteOutLimitFiles(TestFolderName, MaxLimitSize, PostfixName = 'ply')
        t2 = time.time()
        print('time = {}'.format(t2-t1))
    
    
    print('End.')
    
