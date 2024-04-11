# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 12:56:36 2017

@author: xbiao
"""
import numpy as np
from ctypes import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as sio
import time
import ctypes
import os
from matplotlib import path
import logging as lgmsg

from util import CloudPointFuns, PolygonFuns
from util.SaveData import LogSaveCategoryIndexFun

from config import online_offline_type, log_info

# log 信息
ConfigDebugInfo = log_info['debug_info']
ConfigDebugTypeIndex = LogSaveCategoryIndexFun()

# --------------------------------------------------------------------
# 初始化聚类信息
#       CreatDetecter(SensorConfigFileName)
# --------------------------------------------------------------------
PT_LABEL = CloudPointFuns.LoadPointLabelIndex()
KL_EPS = 1.0e-10      # constant in KL analysis

# 聚类文件地址
if online_offline_type == 'OnLine':
    OnLineLibMeanShiftDir = "demo/util/meanshift/demo/LibMeanShift.dll"
    OffLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll"
else:
    OnLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll" # "demo/util/meanshift/demo/LibMeanShift.dll"
    OffLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll"

# Parameter setting for clustering
class TCluParam(Structure):
    _fields_ = [("bx1", c_double),
                ("by1", c_double),
                ("bx2", c_double),
                ("by2", c_double),
                ("GridSize", c_double),
                ("nKernelSize", c_int)]
                
# Prepare output CluTbl
class TCluTbl(Structure):
    _fields_ = [("nCluNum", c_int),
                ("pClu", POINTER(c_void_p))]
                

def MeanShiftCluster2D_LoadIndex():
#    CLUtype=np.dtype([('CluPropNum','i'),('iNUM','i'),('iMX','i'),('iMY','i'),
#                      ('iANG','i'), ('iL1','i'),('iL2','i'),('iMZ','i'),
#                      ('iLX','i'),('iLY','i'),('iLZ','i'),('iLABEL','i'),
#                      ('iMX2','i'),('iMY2','i'),('iMZ2','i')])
#    CLU=np.array([(14,1,2,3,4,5,6,7,8,9,10,11,12,13,14)],dtype=CLUtype)

#    class CLUtype(Structure):
#        _fields_ = [("CluPropNum", c_int),
#                    ("iNUM", c_int),
#                    ("iMX", c_int),
#                    ("iMY", c_int),
#                    ("iANG", c_int),
#                    ("iL1", c_int),
#                    ("iL2", c_int),
#                    ("iMZ", c_int),
#                    ("iLX", c_int),
#                    ("iLY", c_int),
#                    ("iLZ", c_int),
#                    ("iLABEL", c_int),
#                    ("iMX2", c_int),
#                    ("iMY2", c_int),
#                    ("iMZ2", c_int)]

    class CLUtype():
        def __init__(self):
            self.CluPropNum = 0
            self.iNUM = 0
            self.iMX = 0
            self.iMY = 0
            self.iANG = 0
            self.iL1 = 0
            self.iL2 = 0
            self.iMZ = 0
            self.iLX = 0
            self.iLY = 0
            self.iLZ = 0
            self.iLABEL = 0 
            self.iMX2 = 0
            self.iMY2 = 0
            self.iMZ2 = 0
            
    CLU = CLUtype()
    CLU.CluPropNum = 14
    CLU.iNUM = 1
    CLU.iMX = 2
    CLU.iMY = 3
    CLU.iANG = 4
    CLU.iL1 = 5
    CLU.iL2 = 6
    CLU.iMZ = 7
    CLU.iLX = 8
    CLU.iLY = 9
    CLU.iLZ = 10
    CLU.iLABEL = 11
    CLU.iMX2 = 12
    CLU.iMY2 = 13
    CLU.iMZ2 = 14
    
    return CLU

def CAL_CalcCluAxisbyKL(xx, yy, KL_EPS):
    # Calculate cluster axis by KL analysis
    # Please note that (xx, yy) have been zero-mean normalized (or shifted to
    # the specified center)
    
    tmpx = xx
    tmpy = yy
    N = len(tmpx)
    
    cxx = np.sum(tmpx * tmpx) / N + KL_EPS
    cxy = np.sum(tmpx * tmpy) / N + KL_EPS
    cyy = np.sum(tmpy * tmpy) / N + KL_EPS
    
    tmp1 = (cxx + cyy) / 2.0
    tmp2 = np.sqrt((cxx - cyy) * (cxx - cyy) / 4.0 + cxy * cxy)
    m_len1 = tmp1 + tmp2 + KL_EPS
    m_len2 = tmp1 - tmp2 + KL_EPS
    
    if np.abs(cxy) < KL_EPS:
        if cxx > cyy:
            m_ang = 0
        else:
            m_ang = np.pi / 2
    else:
        if np.abs(m_len1 - cxx) < 10.0 * KL_EPS:
            m_ang = np.arctan(cxy / (m_len1 - cyy))
        else:
            m_ang = np.arctan((m_len1 - cxx) / cxy)
    L1 = m_len1
    L2 = m_len2
    Ang = m_ang
    
    return (L1, L2, Ang)

def CAL_CalcCluAxis(objx, objy, objz, objw):
    # Calculate cluster axis by introducing weight
    cx = np.mean(objx)
    cy = np.mean(objy)

    x = objx - cx
    y = -(objy - cy) # This is negative for the orientation calculation (measured in the counter-clockwise direction).
    N = len(x)
    
    # Calculate normalized second central moments for the region. 1/12 is 
    # the normalized second central moment of a pixel with unit length.
    uxx = np.sum(np.power(x,2)) / N + 1e-3
    uyy = np.sum(np.power(y,2)) / N + 1e-3
    uxy = np.sum(x*y) / N

    # Calculate major axis length, minor axis length, and eccentricity.
    common = np.sqrt(np.power((uxx - uyy),2) + 4 * np.power(uxy,2))
    MajorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy + common)
    MinorAxisLength = 2 * np.sqrt(2) * np.sqrt(uxx + uyy - common)
    Eccentricity = 2 * np.sqrt(np.power(MajorAxisLength/2,2) - np.power(MinorAxisLength/2,2)) / MajorAxisLength

    # Calculate orientation.
    ## ??
    
    MajorAxis = MajorAxisLength
    MinorAxis = MinorAxisLength
    
    return (MajorAxis, MinorAxis)

def CLP_PointSetDist(pts1, pts2):
#==============================================================================
#  CLP_PointSetDist
# For each point in set1, find the nearest point in set2 and compute the distance
#   pts1: M * N1
#   pts2, M * N2
#   M: dimension of points, N1, N2: Number of points
# MinDist: N1 * 1
# MinIdx:  N1 * 1
# DistMat: N1 * N2. Only available when bUseHash = 0;
# Param
#   Param.bUseHash:  Whether to use hash table to acclerate computation
# ATTENTION:
#   There is some known problems when using Hash table in this algorithm, be careful when using
#   it.
#==============================================================================
    M  = pts1.shape[0]
    N1 = pts1.shape[1]
    N2 = pts2.shape[1]

    MinDist = np.zeros([N1, 1])
    MinIdx  = np.zeros([N1, 1])	
    DistMat = np.zeros([N1, N2])
    
    if (not pts1.any())|(not pts2.any()):
        return MinDist, MinIdx, DistMat
    if M!=pts2.shape[0]:
        print('Inconsistent dimension!')
        return MinDist, MinIdx, DistMat
        
    p = 2				# Euclidean distance
    
    if (N1 * N2 < 1e7):
        bUseHash = 0
    else:
        bUseHash = 1
    
#    if(nargin >= 3)	   ## ???
    
    if bool(~bUseHash):
        # When the number of point is not huge, compute it directly
        DistMat = np.zeros([N1, N2])
        for m in range(M):
            s1 = pts1[m, :]				# 1 * N1
            s2 = pts2[m, :]				# 1 * N2
            ms1 = np.tile(np.array([s1]).transpose(), [1,N2])		# N1 * N2
            ms2 = np.tile(s2,  [N1,1])		# N1 * N2			
            DistMat = DistMat + np.power((ms1 - ms2),p)       
        
        DistMat = np.power(DistMat,(1/p))
        MinDist = np.min(DistMat,1)
#        MinIdx = np.where(DistMat == MinDist)  ## ??
        for idx in range(N1):
            for ii in range(N2):
                if DistMat[idx,ii]==MinDist[idx]:
                    MinIdx[idx]=ii
                    break
            
    return MinDist, MinIdx, DistMat

def DispClusterInfo(CluTbl, rpx, rpy, rpz, theAxis, GateArea, CLU):
#    cmap = matplotlib.cm.jet
#    cmap = matplotlib.cm.get_cmap('jet')
#    print('plt 0')
    cmap_t = sio.loadmat('colormap64_jet.mat')
#    print('plt 01')
    cm = cmap_t['cm']
#    print('plt 1')
    hh1 = 0.50
    hh2 = 2.00
    hhstep = (hh2 - hh1) / (cm.shape[0] - 1)
#    time.sleep(3)
    if rpx.any():
        TmpPos = (rpz >= hh1) & (rpz < hh2)
        tmpx = rpx[TmpPos]
        tmpy = rpy[TmpPos]
        tmpz = rpz[TmpPos]
        hidx = np.round((tmpz - hh1) / hhstep) + 1
    else:
        tmpx = np.zeros([0, 1])
        tmpy = np.zeros([0, 1])
        tmpz = np.zeros([0, 1])
    
    TmpAxis = theAxis
    UIS_PlotAxis(TmpAxis, np.array([1,1,1]))
#    print('plt 2')
#    time.sleep(3)
    for m in range(len(GateArea)):
        CurrPoly = GateArea[m]
        plt.plot(CurrPoly[:,0],CurrPoly[:,1],"r-")
        plt.plot((CurrPoly[0,0],CurrPoly[-1,0]),(CurrPoly[0,1],CurrPoly[-1,1]),"r-")

    for ch in range(cm.shape[0]):
        hh = (ch - 1) * hhstep + hh1
        TmpPos = (np.abs(tmpz - hh) < (hhstep / 2))
        plt.plot(tmpx[TmpPos], tmpy[TmpPos], '.', color=cm[ch, :])  

#    color_z = np.zeros([tmpx.shape[0],cm.shape[1]])
#    for ch in range(cm.shape[0]):
#        hh = (ch - 1) * hhstep + hh1
#        TmpPos = (np.abs(tmpz - hh) < (hhstep / 2))
#        color_z[TmpPos] = cm[ch, :]
#    plt.scatter(tmpx, tmpy,c= color_z,marker='.')
#    print('plt 3')
    if CluTbl.any():
        CluPtNum = CluTbl[:, CLU.iNUM-1]        
        TmpPts = CluTbl[:, [CLU.iMX-1, CLU.iMY-1, CLU.iMZ-1]]
        mcx = TmpPts[:, 0]
        mcy = TmpPts[:, 1]
        CluNum = CluTbl.shape[0]
#        plt.plot(mcx, mcy, 'mo', 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'k', 'MarkerSize', 5)
        plt.plot(mcx, mcy, 'mo')
    plt.axis('equal')    
    plt.xlim(TmpAxis[0],TmpAxis[1])
    plt.ylim(TmpAxis[2],TmpAxis[3]) 

    
#    bShowColorbar = 0     ### ????

    return 0
    
def UIS_PlotAxis(theAxis, PlotColor):
    bx1 = theAxis[0]
    bx2 = theAxis[1]
    by1 = theAxis[2]
    by2 = theAxis[3]
    xx = np.array([bx1,bx1,bx2,bx2,bx1])
    yy = np.array([by2,by1,by1,by2,by2])
    
    if (not PlotColor.any()):
        PlotColor = np.array([0.0,0.0,0.0])
    
    plt.plot(xx,yy,'-',color=PlotColor)
    return 0
    
def UIS_DispColor(CluLabel):
    
    
    return 0
    

        
# --------------------------------------------------------------------
# 检测目标是否超高
#       目标+高度区域: CalcMultiBBoxInfo(BBoxData, FileName, PredInfo, AboveHeightThod)
#       高度区域: CalcScaleAreaClusterInfo(BBoxData, FileName, AboveHeightThod):
#       人员目标检测：CalcHumanClusterInfo(BBoxData, FileName)
# --------------------------------------------------------------------  
def CalcMultiBBoxInfo(BBoxData, FileName, PredInfo, AboveHeightThod):
    # 输入：多个目标检测框:BBoxData[N * 3]
    #       文件名称:FileName
    #       多个目标检测框的位置:PredInfo[M * 3]
    #       限制高度：AboveHeightThod
    # 输出：目标信息
    #       目标是否超高：NewPtsHeight[M * 1]
    CLU = MeanShiftCluster2D_LoadIndex()
    
    # 保存数据 .ply
    SaveMultiBBoxPlyFlag = 0 # 是否保存检测目标的点云数据
    
    # OnLine mode
    if len(FileName) == 0: # OnLine mode not save result
        SaveMultiBBoxPlyFlag = 0
    
    # SaveMultiBBoxPlyFlag
    if SaveMultiBBoxPlyFlag == 1:
        BaseDir = os.getcwd()
        TempSaveName = os.path.join(BaseDir, 'dete_obj' + '/' + FileName + '.ply')
        if not os.path.exists(os.path.join(BaseDir, 'dete_obj')): 
            os.mkdir(os.path.join(BaseDir, 'dete_obj'))
        CloudPointFuns.SavePt3D2Ply(TempSaveName, BBoxData, 'XYZ')
        print('SavePt3D2Ply Finished.')
        
    # 选择聚类平面
    CluterPlaneFlag = 1; # if CluterPlaneFlag == 1, 投影至xy 平面进行cluster
                         # if CluterPlaneFlag == 2, 投影至xz 平面进行cluster

    
    # 选择一定高度的有效数据
##    BBoxData = np.empty([1,3])
#    BBoxDataInValid = (BBoxData[:,2]<-500) # select AboveHeightThod near data to cluster, (init 0.7M)
#    print('BBoxDataInValid : ' + str(BBoxDataInValid))
    
    BBoxDataInValid = (BBoxData[:,2]<(AboveHeightThod-0.2)) # select AboveHeightThod near data to cluster, (init 0.7M)
    BBoxData[BBoxDataInValid,0] = 0
    BBoxData[BBoxDataInValid,1] = 0
    BBoxData[BBoxDataInValid,2] = 0
    
    # 检测数据
    MinPeopleHeight = 0.3
    MinCluPtNum = 200 # init: 400
    
    KernelSize = 15
    GridSize = 0.015
    theAxis = np.array([-11.0, 19.0, -17.0, 11.0]) # [xmin, xmax, ymin, ymax]
    # TCluParam
    MSC_Param = TCluParam()
    MSC_Param.bx1 = theAxis[0]
    MSC_Param.bx2 = theAxis[1]
    MSC_Param.by1 = theAxis[2]
    MSC_Param.by2 = theAxis[3]
    MSC_Param.GridSize = GridSize
    MSC_Param.nKernelSize = -1
    
    # TCluTbl   
    CluTbl0 = TCluTbl()
    nMaxCluNum = 10000   #  enough ???
    CluTbl0.nCluNum = nMaxCluNum
    CluPropNum = 6   # const
    CluProps = np.zeros([CluTbl0.nCluNum, CluPropNum], 'double', 'C')
    # 如果不是C连续的内存，必须强制转换
    if not CluProps.flags['C_CONTIGUOUS']:
        CluProps = np.ascontiguous(CluProps, dtype = CluProps.dtype)
    CluTbl0.pClu = cast(CluProps.ctypes.data, POINTER(c_void_p))
    
    NewPts0 = BBoxData.transpose() # [3 * N] 
    NewPtsValidHeightIdx = (NewPts0[2,:]>0) # 选择地面一定高度的数据
    NewPts = NewPts0[:,NewPtsValidHeightIdx]
    PtNum = len(NewPts[0,:])
    rpx = np.zeros([1,PtNum], 'double', 'C')
    rpy = np.zeros([1,PtNum], 'double', 'C')
    weight = np.zeros([PtNum, 1], 'double', 'C')
    rpx[0,:] = NewPts[0, :]
    rpy[0,:] = NewPts[1, :]
    rpz = NewPts[2, :]
    rpx = rpx.transpose()
    rpy = rpy.transpose()
    
    if CluterPlaneFlag == 1:
        tmz = rpz
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpy.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    elif CluterPlaneFlag == 2:
        tmz = rpy
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpz.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    
    # 选择有效的cluster 目标
    if (res>0):
        CluProps_V = np.zeros([res,CLU.CluPropNum])
        CluProps_Height = np.zeros([1,res]) # 保存每类数据高度
        CluProps_V[:,(CLU.iNUM-1,CLU.iMX-1,CLU.iMY-1,CLU.iANG-1,CLU.iL1-1,CLU.iL2-1)] = CluProps[0:res, :]

        CluLabel[CluLabel < 0] = PT_LABEL.OUTBUND
        CluPtNum = CluProps_V[:, CLU.iNUM-1] #
        mcx = CluProps_V[:, CLU.iMX-1]
        mcy = CluProps_V[:, CLU.iMY-1]
        CluNum = CluProps_V.shape[0]

        # Compute additional properties of clusters
        ObjSize = np.zeros([CluNum, 1])
        for nc in range(CluNum): ##
            pos = (CluLabel == (nc+1))  
            objx = np.array([rpx])[pos.transpose()]
            objy = np.array([rpy])[pos.transpose()]
            objz = np.array([rpz])[pos.transpose()]
            objw = np.array([weight])[pos.transpose()]
            dx = np.max(objx) - np.min(objx)
            dy = np.max(objy) - np.min(objy)

#            if (dx > 2.0) | (dy > 2.0):
#                print('obj of large size: nc = %d'%nc)

            if len(objx)==1:
                ObjSize[nc] = np.sqrt(np.var(objx) * np.var(objy) * np.var(objz))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz)
            else:
                ObjSize[nc] = np.sqrt(np.var(objx,ddof=1) * np.var(objy,ddof=1) * np.var(objz,ddof=1))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx,ddof=1)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy,ddof=1)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz,ddof=1)
            # Use weighted mean value of z, which provides a better
            # description about the height of objects
            CluProps_V[nc, CLU.iMZ-1] = np.sum(objz * objw) / np.sum(objw)
            CluProps_V[nc, CLU.iMX2-1] = np.mean(objx)
            CluProps_V[nc, CLU.iMY2-1] = np.mean(objy)
            CluProps_V[nc, CLU.iMZ2-1] = np.mean(objz)
            CluProps_Height[0][nc] = np.max(objz) # 保存每类数据高度

            # Calculate axis and orientation
            tmpx = objx - CluProps_V[nc, CLU.iMX-1]
            tmpy = objy - CluProps_V[nc, CLU.iMY-1]

            [L1, L2, Ang] = CAL_CalcCluAxisbyKL(tmpx, tmpy, KL_EPS)
            CluProps_V[nc, CLU.iANG-1] = Ang

            if True:
                [MajorAxis, MinorAxis] = CAL_CalcCluAxis(objx, objy, objz, objw)
                CluProps_V[nc, CLU.iL1-1] = MajorAxis / 2
                CluProps_V[nc, CLU.iL2-1] = MinorAxis / 2

        # Filter invalid clusters
        # Find good candidates from clusters
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  

        LabelRemap = np.array(range(0,CluNum))
        LabelRemap[~ValidPos] = PT_LABEL.UNDEFINED
        LabelRemap[CluProps_V[:, CLU.iNUM-1] <= 10] = PT_LABEL.NOISE
        numValidPos = 0
        for ii in range(ValidPos.shape[0]):
            if ValidPos[ii]==True:
                numValidPos=numValidPos+1
        LabelRemap[ValidPos] = np.array(range(1,numValidPos+1))  ## ValidPos index
        CluProps_V = CluProps_V[ValidPos, :] # 选择后的有效类别
        CluProps_Height = CluProps_Height[:,ValidPos] # 选择后的有效类别
        TmpPos = (CluLabel > 0)
        CluLabel[TmpPos] = LabelRemap[CluLabel[TmpPos]-1]
    else:
        CluProps_V = np.empty(shape=[0, CLU.CluPropNum-1])
        
    # 多个有效目标所有数据的高度
    NewPtsHeight = np.zeros([1,PredInfo.shape[0]]) # 超高目标
    for i in range(CluProps_V.shape[0]): # cluster 选择后的目标信息
        TempMinDis = 100 # 初始最小值
        k = -1 # 初始邻近目标序号
        for j in range(PredInfo.shape[0]): # 检测目标框的信息
            TempDis = np.sqrt((CluProps_V[i, CLU.iMX2-1] - PredInfo[j,0])**2 + (CluProps_V[i, CLU.iMY2-1] - PredInfo[j,1])**2 + (CluProps_V[i, CLU.iMZ2-1] - PredInfo[j,2])**2)
            if (TempMinDis>TempDis):
                k = j
                TempMinDis = TempDis
        if k>-1:
            NewPtsHeight[0,k] = 1 # 超高目标Flag

    
    return NewPtsHeight
    
def CalcScaleAreaClusterInfo(BBoxData, FileName, AboveHeightThod, ZaxisHeightScale=0.25):
    # 输入：一定区域数据:BBoxData[N * 3]
    #       文件名称:FileName
    #       限制高度：AboveHeightThod
    #       垂直方向范围：ZaxisHeightScale
    # 输出：目标信息
    #       目标是否超高：AboveHeightFlag, [1 * 4], [validflag, x, y, z ]
    
#    print('CalcScaleAreaClusterInfo Start')

#    print('BBoxData size = {}'.format(BBoxData.shape))
    
    CLU = MeanShiftCluster2D_LoadIndex()
    
    # 初始化 AboveHeightFlag
    AboveHeightFlag = np.zeros([1,4])
    
    # 保存数据 .ply
    SaveMultiBBoxPlyFlag = 0 # 是否保存检测目标的点云数据
    
    # OnLine mode
    if len(FileName) == 0: # OnLine mode not save result
        SaveMultiBBoxPlyFlag = 0
    # SaveMultiBBoxPlyFlag
    if SaveMultiBBoxPlyFlag == 1:
        if BBoxData.shape[0] > 0: # valid pts 
            BaseDir = os.getcwd()
            TempSaveName = os.path.join(BaseDir, 'dete_obj' + '/' + FileName + '.ply')
            if not os.path.exists(os.path.join(BaseDir, 'dete_obj')): 
                os.mkdir(os.path.join(BaseDir, 'dete_obj'))
            CloudPointFuns.SavePt3D2Ply(TempSaveName, BBoxData, 'XYZ')
            print('SavePt3D2Ply Finished.')
        
    # 选择聚类平面
    CluterPlaneFlag = 1; # if CluterPlaneFlag == 1, 投影至xy 平面进行cluster
                         # if CluterPlaneFlag == 2, 投影至xz 平面进行cluster
    
    # 选择一定高度的有效数据  
    BBoxDataInValid = (BBoxData[:,2]<(AboveHeightThod-ZaxisHeightScale))  # select AboveHeightThod near data to cluster, (init 0.7M)
    BBoxData[BBoxDataInValid,0] = 0
    BBoxData[BBoxDataInValid,1] = 0
    BBoxData[BBoxDataInValid,2] = 0
    BBoxDataInValid = (BBoxData[:,2]>AboveHeightThod)  # 
    BBoxData[BBoxDataInValid,0] = 0
    BBoxData[BBoxDataInValid,1] = 0
    BBoxData[BBoxDataInValid,2] = 0
    
    # 检测数据
    MinPeopleHeight = 0.3
    MinCluPtNum = 200 # init: 400
    
    KernelSize = 15
    GridSize = 0.015
    theAxis = np.array([-20.0, 20.0, -20.0, 20.0]) # [xmin, xmax, ymin, ymax], ??????
    # TCluParam
    MSC_Param = TCluParam()
    MSC_Param.bx1 = theAxis[0]
    MSC_Param.bx2 = theAxis[1]
    MSC_Param.by1 = theAxis[2]
    MSC_Param.by2 = theAxis[3]
    MSC_Param.GridSize = GridSize
    MSC_Param.nKernelSize = -1
    
    # TCluTbl   
    CluTbl0 = TCluTbl()
    nMaxCluNum = 10000   #  enough ???
    CluTbl0.nCluNum = nMaxCluNum
    CluPropNum = 6   # const
    CluProps = np.zeros([CluTbl0.nCluNum, CluPropNum], 'double', 'C')
    # 如果不是C连续的内存，必须强制转换
    if not CluProps.flags['C_CONTIGUOUS']:
        CluProps = np.ascontiguous(CluProps, dtype = CluProps.dtype)
    CluTbl0.pClu = cast(CluProps.ctypes.data, POINTER(c_void_p))
    
    NewPts0 = BBoxData.transpose() # [3 * N] 
    NewPtsValidHeightIdx = (NewPts0[2,:]>0) # 选择地面一定高度的数据
    NewPts = NewPts0[:,NewPtsValidHeightIdx]
    PtNum = len(NewPts[0,:])
    rpx = np.zeros([1,PtNum], 'double', 'C')
    rpy = np.zeros([1,PtNum], 'double', 'C')
    weight = np.zeros([PtNum, 1], 'double', 'C')
    rpx[0,:] = NewPts[0, :]
    rpy[0,:] = NewPts[1, :]
    rpz = NewPts[2, :]
    rpx = rpx.transpose()
    rpy = rpy.transpose()
    
    if CluterPlaneFlag == 1:
        tmz = rpz
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
#            LibMS = ctypes.windll.LoadLibrary("demo/LibMeanShift.dll")
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
            
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpy.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    elif CluterPlaneFlag == 2:
        tmz = rpy
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpz.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    
    # 选择有效的cluster 目标
    if (res>0):
        CluProps_V = np.zeros([res,CLU.CluPropNum])
        CluProps_Height = np.zeros([1,res]) # 保存每类数据高度
        CluProps_V[:,(CLU.iNUM-1,CLU.iMX-1,CLU.iMY-1,CLU.iANG-1,CLU.iL1-1,CLU.iL2-1)] = CluProps[0:res, :]

        CluLabel[CluLabel < 0] = PT_LABEL.OUTBUND
        CluPtNum = CluProps_V[:, CLU.iNUM-1] #
        mcx = CluProps_V[:, CLU.iMX-1]
        mcy = CluProps_V[:, CLU.iMY-1]
        CluNum = CluProps_V.shape[0]

        # Compute additional properties of clusters
        ObjSize = np.zeros([CluNum, 1])
        for nc in range(CluNum): ##
            pos = (CluLabel == (nc+1))  
            objx = np.array([rpx])[pos.transpose()]
            objy = np.array([rpy])[pos.transpose()]
            objz = np.array([rpz])[pos.transpose()]
            objw = np.array([weight])[pos.transpose()]
            dx = np.max(objx) - np.min(objx)
            dy = np.max(objy) - np.min(objy)

#            if (dx > 2.0) | (dy > 2.0):
#                print('obj of large size: nc = %d'%nc)

            if len(objx)==1:
                ObjSize[nc] = np.sqrt(np.var(objx) * np.var(objy) * np.var(objz))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz)
            else:
                ObjSize[nc] = np.sqrt(np.var(objx,ddof=1) * np.var(objy,ddof=1) * np.var(objz,ddof=1))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx,ddof=1)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy,ddof=1)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz,ddof=1)
            # Use weighted mean value of z, which provides a better
            # description about the height of objects
            CluProps_V[nc, CLU.iMZ-1] = np.sum(objz * objw) / np.sum(objw)
            CluProps_V[nc, CLU.iMX2-1] = np.mean(objx)
            CluProps_V[nc, CLU.iMY2-1] = np.mean(objy)
            CluProps_V[nc, CLU.iMZ2-1] = np.mean(objz)
            CluProps_Height[0][nc] = np.max(objz) # 保存每类数据高度

            # Calculate axis and orientation
            tmpx = objx - CluProps_V[nc, CLU.iMX-1]
            tmpy = objy - CluProps_V[nc, CLU.iMY-1]

            [L1, L2, Ang] = CAL_CalcCluAxisbyKL(tmpx, tmpy, KL_EPS)
            CluProps_V[nc, CLU.iANG-1] = Ang

            if True:
                [MajorAxis, MinorAxis] = CAL_CalcCluAxis(objx, objy, objz, objw)
                CluProps_V[nc, CLU.iL1-1] = MajorAxis / 2
                CluProps_V[nc, CLU.iL2-1] = MinorAxis / 2

        # Filter invalid clusters
        # Find good candidates from clusters
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
        # 201903
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        # 201909
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.035) & (CluProps_V[:, CLU.iLY-1] > 0.035) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        
        # 20190925
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.025) & (CluProps_V[:, CLU.iLY-1] > 0.025) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  

        # 20200413, ZT-Data
        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.020) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  

#        print('ValidPos = {}'.format(ValidPos))
#        print('CluProps iL1 = {}'.format(CluProps_V[:, CLU.iL1-1]))
#        print('CluProps iLX = {}'.format(CluProps_V[:, CLU.iLX-1]))
#        print('CluProps iLY = {}'.format(CluProps_V[:, CLU.iLY-1]))
#        print('CluProps iLZ = {}'.format(CluProps_V[:, CLU.iLZ-1]))
#        print('CluProps iNUM = {}'.format(CluProps_V[:, CLU.iNUM-1]))
#        print('CluProps iMZ = {}'.format(CluProps_V[:, CLU.iMZ-1]))
        
        LabelRemap = np.array(range(0,CluNum))
        LabelRemap[~ValidPos] = PT_LABEL.UNDEFINED
        LabelRemap[CluProps_V[:, CLU.iNUM-1] <= 10] = PT_LABEL.NOISE
        numValidPos = 0
        for ii in range(ValidPos.shape[0]):
            if ValidPos[ii]==True:
                numValidPos=numValidPos+1
        LabelRemap[ValidPos] = np.array(range(1,numValidPos+1))  ## ValidPos index
        CluProps_V = CluProps_V[ValidPos, :] # 选择后的有效类别
        CluProps_Height = CluProps_Height[:,ValidPos] # 选择后的有效类别
        TmpPos = (CluLabel > 0)
        CluLabel[TmpPos] = LabelRemap[CluLabel[TmpPos]-1]
    else:
        CluProps_V = np.empty(shape=[0, CLU.CluPropNum-1])
        
    # 判断范围数据是否有超高行为
    if CluProps_V.shape[0] > 0:
        # valid_flag
        AboveHeightFlag[0,0] = 1
        # x,y,z
        CluProps_V_PtsNum = CluProps_V[0, CLU.iNUM-1] # 选择数据点最多的聚类目标
        AboveHeightFlag[0,1] = CluProps_V[0, CLU.iMX-1] # 初始目标位置信息
        AboveHeightFlag[0,2] = CluProps_V[0, CLU.iMY-1]
        AboveHeightFlag[0,3] = CluProps_V[0, CLU.iMZ-1]
        for i in range(CluProps_V.shape[0]): # cluster 选择后的目标信息
            if CluProps_V_PtsNum < CluProps_V[i, CLU.iNUM-1]:
                AboveHeightFlag[0,1] = CluProps_V[i, CLU.iMX-1] # x
                AboveHeightFlag[0,2] = CluProps_V[i, CLU.iMY-1] # y
                AboveHeightFlag[0,3] = CluProps_V[i, CLU.iMZ-1] # z
    else:
        AboveHeightFlag[0,0] = 0

    
    return AboveHeightFlag
    
def CalcScaleAreaAboveHeightClusterInfo(BBoxData, FileName, AboveHeightThod, ZaxisHeightScale=0.25, MinCluPtNumThod = 200, InnerBoundRange=None, PtsClusterDistrSelect = 1):
    # 功能：针对超高告警，计算目标位置信息，返回多个超高目标位置
    # 输入：一定区域数据:BBoxData[N * 3]
    #       文件名称:FileName
    #       限制高度：AboveHeightThod
    #       垂直方向范围：ZaxisHeightScale
    #       目标最少点数：MinCluPtNumThod
    #       内部边界范围：InnerBoundRange, [xmin, xmax, ymin, ymax, zmin, zmax]
    # 输出：目标信息
    #       目标是否超高：AboveHeightFlag, [N * 4], [validflag, x, y, z ]
    
    # 设置超高目标分布信息
    if PtsClusterDistrSelect == 1: # 严格，【靠墙举手等物体检测】
        PtsCluDistr_iLX = 0.005
        PtsCluDistr_iLY = 0.005
        PtsCluDistr_iLZ = 0.005
        PtsCluDistr_dX = 0
        PtsCluDistr_dY = 0
    elif PtsClusterDistrSelect == 2: # 正常，【举手等物体检测】
        PtsCluDistr_iLX = 0.02
        PtsCluDistr_iLY = 0.01
        PtsCluDistr_iLZ = 0.02
        PtsCluDistr_dX = 0
        PtsCluDistr_dY = 0
    elif PtsClusterDistrSelect == 3: # 不严格，【头部等大物体检测】
        PtsCluDistr_iLX = 0.035
        PtsCluDistr_iLY = 0.035
        PtsCluDistr_iLZ = 0.035
        PtsCluDistr_dX = 0
        PtsCluDistr_dY = 0
    else:
        PtsCluDistr_iLX = 0.005
        PtsCluDistr_iLY = 0.005
        PtsCluDistr_iLZ = 0.005
        PtsCluDistr_dX = 0
        PtsCluDistr_dY = 0
        
    
#    print('CalcScaleAreaClusterInfo Start')

#    print('BBoxData size = {}'.format(BBoxData.shape))
    
    CLU = MeanShiftCluster2D_LoadIndex()
    
    # 初始化 AboveHeightFlag
    AboveHeightFlag = np.zeros([1,4])
    
    # 保存数据 .ply
    SaveMultiBBoxPlyFlag = 0 # 是否保存检测目标的点云数据
    
    # OnLine mode
    if len(FileName) == 0: # OnLine mode not save result
        SaveMultiBBoxPlyFlag = 0
    # SaveMultiBBoxPlyFlag
    if SaveMultiBBoxPlyFlag == 1:
        if BBoxData.shape[0] > 0: # valid pts 
            BaseDir = os.getcwd()
            TempSaveName = os.path.join(BaseDir, 'dete_obj' + '/' + FileName + '.ply')
            if not os.path.exists(os.path.join(BaseDir, 'dete_obj')): 
                os.mkdir(os.path.join(BaseDir, 'dete_obj'))
            CloudPointFuns.SavePt3D2Ply(TempSaveName, BBoxData, 'XYZ')
            print('SavePt3D2Ply Finished.')
        
    # 选择聚类平面
    CluterPlaneFlag = 1; # if CluterPlaneFlag == 1, 投影至xy 平面进行cluster
                         # if CluterPlaneFlag == 2, 投影至xz 平面进行cluster
    
    # 选择一定高度的有效数据  
    BBoxDataInValid = (BBoxData[:,2]<(AboveHeightThod-ZaxisHeightScale))  # select AboveHeightThod near data to cluster, (init 0.7M)
    BBoxData[BBoxDataInValid,0] = 0
    BBoxData[BBoxDataInValid,1] = 0
    BBoxData[BBoxDataInValid,2] = 0
    BBoxDataInValid = (BBoxData[:,2]>AboveHeightThod)  # 
    BBoxData[BBoxDataInValid,0] = 0
    BBoxData[BBoxDataInValid,1] = 0
    BBoxData[BBoxDataInValid,2] = 0
    
    # 检测数据
    MinPeopleHeight = 0.3
#    MinCluPtNum = 200 # init: 400
    MinCluPtNum = MinCluPtNumThod # 外部传入
#    MinCluPtNum = 250 # init: 400
    
    KernelSize = 15
    GridSize = 0.015
    theAxis = np.array([-20.0, 20.0, -20.0, 20.0]) # [xmin, xmax, ymin, ymax], ??????
    # TCluParam
    MSC_Param = TCluParam()
    MSC_Param.bx1 = theAxis[0]
    MSC_Param.bx2 = theAxis[1]
    MSC_Param.by1 = theAxis[2]
    MSC_Param.by2 = theAxis[3]
    MSC_Param.GridSize = GridSize
    MSC_Param.nKernelSize = -1
    
    # TCluTbl   
    CluTbl0 = TCluTbl()
    nMaxCluNum = 10000   #  enough ???
    CluTbl0.nCluNum = nMaxCluNum
    CluPropNum = 6   # const
    CluProps = np.zeros([CluTbl0.nCluNum, CluPropNum], 'double', 'C')
    # 如果不是C连续的内存，必须强制转换
    if not CluProps.flags['C_CONTIGUOUS']:
        CluProps = np.ascontiguous(CluProps, dtype = CluProps.dtype)
    CluTbl0.pClu = cast(CluProps.ctypes.data, POINTER(c_void_p))
    
    NewPts0 = BBoxData.transpose() # [3 * N] 
    NewPtsValidHeightIdx = (NewPts0[2,:]>0) # 选择地面一定高度的数据
    NewPts = NewPts0[:,NewPtsValidHeightIdx]
    PtNum = len(NewPts[0,:])
    rpx = np.zeros([1,PtNum], 'double', 'C')
    rpy = np.zeros([1,PtNum], 'double', 'C')
    weight = np.zeros([PtNum, 1], 'double', 'C')
    rpx[0,:] = NewPts[0, :]
    rpy[0,:] = NewPts[1, :]
    rpz = NewPts[2, :]
    rpx = rpx.transpose()
    rpy = rpy.transpose()
    
    if CluterPlaneFlag == 1:
        tmz = rpz
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
#            LibMS = ctypes.windll.LoadLibrary("demo/LibMeanShift.dll")
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
            
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpy.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    elif CluterPlaneFlag == 2:
        tmz = rpy
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpz.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    
    # 选择有效的cluster 目标
    if (res>0):
        CluProps_V = np.zeros([res,CLU.CluPropNum])
        CluProps_Height = np.zeros([1,res]) # 保存每类数据高度
        CluProps_V[:,(CLU.iNUM-1,CLU.iMX-1,CLU.iMY-1,CLU.iANG-1,CLU.iL1-1,CLU.iL2-1)] = CluProps[0:res, :]

        CluLabel[CluLabel < 0] = PT_LABEL.OUTBUND
        CluPtNum = CluProps_V[:, CLU.iNUM-1] #
        mcx = CluProps_V[:, CLU.iMX-1]
        mcy = CluProps_V[:, CLU.iMY-1]
        CluNum = CluProps_V.shape[0]
        
        # 目标聚类的范围
        CluProps_V_Range = np.zeros([CluNum,3]) # [range_x, range_y, range_z]

        # Compute additional properties of clusters
        ObjSize = np.zeros([CluNum, 1])
        for nc in range(CluNum): ##
            pos = (CluLabel == (nc+1))  
            objx = np.array([rpx])[pos.transpose()]
            objy = np.array([rpy])[pos.transpose()]
            objz = np.array([rpz])[pos.transpose()]
            objw = np.array([weight])[pos.transpose()]
            dx = np.max(objx) - np.min(objx)
            dy = np.max(objy) - np.min(objy)

#            if (dx > 2.0) | (dy > 2.0):
#                print('obj of large size: nc = %d'%nc)
            # 目标聚类范围
            CluProps_V_Range[nc, 0] = dx
            CluProps_V_Range[nc, 1] = dy

            if len(objx)==1:
                ObjSize[nc] = np.sqrt(np.var(objx) * np.var(objy) * np.var(objz))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz)
            else:
                ObjSize[nc] = np.sqrt(np.var(objx,ddof=1) * np.var(objy,ddof=1) * np.var(objz,ddof=1))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx,ddof=1)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy,ddof=1)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz,ddof=1)
            # Use weighted mean value of z, which provides a better
            # description about the height of objects
            CluProps_V[nc, CLU.iMZ-1] = np.sum(objz * objw) / np.sum(objw)
            CluProps_V[nc, CLU.iMX2-1] = np.mean(objx)
            CluProps_V[nc, CLU.iMY2-1] = np.mean(objy)
            CluProps_V[nc, CLU.iMZ2-1] = np.mean(objz)
            CluProps_Height[0][nc] = np.max(objz) # 保存每类数据高度

            # Calculate axis and orientation
            tmpx = objx - CluProps_V[nc, CLU.iMX-1]
            tmpy = objy - CluProps_V[nc, CLU.iMY-1]

            [L1, L2, Ang] = CAL_CalcCluAxisbyKL(tmpx, tmpy, KL_EPS)
            CluProps_V[nc, CLU.iANG-1] = Ang

            if True:
                [MajorAxis, MinorAxis] = CAL_CalcCluAxis(objx, objy, objz, objw)
                CluProps_V[nc, CLU.iL1-1] = MajorAxis / 2
                CluProps_V[nc, CLU.iL2-1] = MinorAxis / 2

        # Filter invalid clusters
        # Find good candidates from clusters
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
        # 201903
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        # 201909
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.035) & (CluProps_V[:, CLU.iLY-1] > 0.035) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        
        # 20190925
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.025) & (CluProps_V[:, CLU.iLY-1] > 0.025) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  

        # 20200413, ZT-Data
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.020) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  

        # 20200715, LG-Data
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.020) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.02) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        # 20200726
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.020) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.02) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        
        # 20200727
        if InnerBoundRange is None:
#            ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.020) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.02) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
#            ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.010) & (CluProps_V[:, CLU.iLY-1] > 0.010) & (CluProps_V[:, CLU.iLZ-1] > 0.01) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
            # 20200805
#            ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.005) & (CluProps_V[:, CLU.iLY-1] > 0.005) & (CluProps_V[:, CLU.iLZ-1] > 0.005) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  #
            # 20200806
            # 通过告警等级设置不同的聚类目标属性限制
            ValidPos = (CluProps_V[:, CLU.iLX-1] > PtsCluDistr_iLX) & (CluProps_V[:, CLU.iLY-1] > PtsCluDistr_iLY) & (CluProps_V[:, CLU.iLZ-1] > PtsCluDistr_iLZ) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  #

        else:
            ValidPos = np.zeros([1,CluNum]) # 初始化
            InnerBoundRange_xy = [InnerBoundRange[0][0], InnerBoundRange[1][0], InnerBoundRange[2][0], InnerBoundRange[3][0]]
            CurrPoly = PolygonFuns.TransBboxToPoints(InnerBoundRange_xy) #  [xmin, xmax, ymin, ymax, zmin, zmax] -> [x1,y1;x2,y2;x3,y3;x4,y4]
            CurrPoly = np.array(CurrPoly)
            CurrPoly = CurrPoly.reshape([int(len(CurrPoly)/2),2]) # 边界角点
            pCurrPoly = path.Path(CurrPoly)
            for nc in range(CluNum): ##
                TempData = np.array([[0.0, 0.0]])
                TempData[0,0] = CluProps_V[nc, CLU.iMX-1]
                TempData[0,1] = CluProps_V[nc, CLU.iMY-1]
                binAreaAll = pCurrPoly.contains_points(TempData) # limit xy, [2 x N]
                if binAreaAll[0]: # 是否在多边形内
                    MinCluPtNumK = 1
#                    CurValidPos = (CluProps_V[nc, CLU.iLX-1] > 0.005) & (CluProps_V[nc, CLU.iLY-1] > 0.005) & (CluProps_V[nc, CLU.iLZ-1] > 0.005) & (CluPtNum[nc] >= MinCluPtNumK * MinCluPtNum) & (CluProps_V[nc, CLU.iMZ-1] > MinPeopleHeight)  
                    CurValidPos = (CluProps_V[nc, CLU.iLX-1] > PtsCluDistr_iLX) & (CluProps_V[nc, CLU.iLY-1] > PtsCluDistr_iLY) & (CluProps_V[nc, CLU.iLZ-1] > PtsCluDistr_iLZ) & (CluPtNum[nc] >= MinCluPtNumK * MinCluPtNum) & (CluProps_V[nc, CLU.iMZ-1] > MinPeopleHeight)  

                else:
                    MinCluPtNumK = 1.5
#                    CurValidPos = (CluProps_V[nc, CLU.iLX-1] > 0.005) & (CluProps_V[nc, CLU.iLY-1] > 0.005) & (CluProps_V[nc, CLU.iLZ-1] > 0.005) & (CluPtNum[nc] >= MinCluPtNumK * MinCluPtNum) & (CluProps_V[nc, CLU.iMZ-1] > MinPeopleHeight)
                    CurValidPos = (CluProps_V[nc, CLU.iLX-1] > PtsCluDistr_iLX) & (CluProps_V[nc, CLU.iLY-1] > PtsCluDistr_iLY) & (CluProps_V[nc, CLU.iLZ-1] > PtsCluDistr_iLZ) & (CluPtNum[nc] >= MinCluPtNumK * MinCluPtNum) & (CluProps_V[nc, CLU.iMZ-1] > MinPeopleHeight)

                ValidPos[0,nc] = CurValidPos
            ValidPos = (ValidPos[0,:]==True) # bool
  
        if int(ConfigDebugInfo[ConfigDebugTypeIndex.CluObjInfo]) > 0: # 打印点云检测结果信息
            lgmsg.debug('  CalcScaleAreaAboveHeightClusterInfo ValidPos = {}'.format(ValidPos))
            lgmsg.debug('    CluProps iL1 = {}'.format(CluProps_V[:, CLU.iL1-1]))
            lgmsg.debug('    CluProps iLX = {}'.format(CluProps_V[:, CLU.iLX-1]))
            lgmsg.debug('    CluProps iLY = {}'.format(CluProps_V[:, CLU.iLY-1]))
            lgmsg.debug('    CluProps iLZ = {}'.format(CluProps_V[:, CLU.iLZ-1]))
            lgmsg.debug('    CluProps iNUM = {}'.format(CluProps_V[:, CLU.iNUM-1]))
            lgmsg.debug('    CluProps iMZ = {}'.format(CluProps_V[:, CLU.iMZ-1]))
        
        # 暂时未使用 dx/dy
#        print('CluProps dx = {}'.format(CluProps_V_Range[:, 0]))
#        print('CluProps dy = {}'.format(CluProps_V_Range[:, 1]))
        
        
        LabelRemap = np.array(range(0,CluNum))
        LabelRemap[~ValidPos] = PT_LABEL.UNDEFINED
        LabelRemap[CluProps_V[:, CLU.iNUM-1] <= 10] = PT_LABEL.NOISE
        numValidPos = 0
        for ii in range(ValidPos.shape[0]):
            if ValidPos[ii]==True:
                numValidPos=numValidPos+1
        LabelRemap[ValidPos] = np.array(range(1,numValidPos+1))  ## ValidPos index
        CluProps_V = CluProps_V[ValidPos, :] # 选择后的有效类别
        CluProps_Height = CluProps_Height[:,ValidPos] # 选择后的有效类别
        TmpPos = (CluLabel > 0)
        CluLabel[TmpPos] = LabelRemap[CluLabel[TmpPos]-1]
    else:
        CluProps_V = np.empty(shape=[0, CLU.CluPropNum-1])
        
    # 判断范围数据是否有超高行为
    if CluProps_V.shape[0] > 0: # 返回多个超高目标位置信息
        # valid_flag
        AboveHeightFlag = np.zeros([CluProps_V.shape[0], 4])
        for i in range(CluProps_V.shape[0]): # cluster 选择后的目标信息
            AboveHeightFlag[i,0] = 1
            AboveHeightFlag[i,1] = CluProps_V[i, CLU.iMX-1] # x
            AboveHeightFlag[i,2] = CluProps_V[i, CLU.iMY-1] # y
            AboveHeightFlag[i,3] = CluProps_V[i, CLU.iMZ-1] # z
    else:
        AboveHeightFlag[0,0] = 0

    return AboveHeightFlag

    
def CalcHumanClusterInfo(BBoxData, FileName):
    # 功能：人员目标检测
    # 输入：一定区域数据:BBoxData[N * 3]
    #       文件名称:FileName
    # 输出：目标信息
    #       目标位置信息：PreHumanInfo, [M * 4], [index, x, y, z ]
    
    CLU = MeanShiftCluster2D_LoadIndex()
    
    # 初始化 PreHumanInfo
    PreHumanInfo = []
    
    # 保存数据 .ply
    SaveMultiBBoxPlyFlag = 0 # 是否保存检测目标的点云数据

    # OnLine mode
    if len(FileName) == 0: # OnLine mode not save result
        SaveMultiBBoxPlyFlag = 0
    # SaveMultiBBoxPlyFlag
    if SaveMultiBBoxPlyFlag == 1:
        if BBoxData.shape[0] > 0: # valid pts 
            BaseDir = os.getcwd()
            TempSaveName = os.path.join(BaseDir, 'dete_obj' + '/' + FileName + '.ply')
            if not os.path.exists(os.path.join(BaseDir, 'dete_obj')): 
                os.mkdir(os.path.join(BaseDir, 'dete_obj'))
            CloudPointFuns.SavePt3D2Ply(TempSaveName, BBoxData, 'XYZ')
            print('SavePt3D2Ply Finished.')
        
    # 选择聚类平面
    CluterPlaneFlag = 1; # if CluterPlaneFlag == 1, 投影至xy 平面进行cluster
                         # if CluterPlaneFlag == 2, 投影至xz 平面进行cluster

    # 检测数据
    MinPeopleHeight = 0.3
    MinCluPtNum = 300 # init: 400
    
    KernelSize = 15 # init:15
    GridSize = 0.015
    theAxis = np.array([-20.0, 20.0, -20.0, 20.0]) # [xmin, xmax, ymin, ymax], ??????
    # TCluParam
    MSC_Param = TCluParam()
    MSC_Param.bx1 = theAxis[0]
    MSC_Param.bx2 = theAxis[1]
    MSC_Param.by1 = theAxis[2]
    MSC_Param.by2 = theAxis[3]
    MSC_Param.GridSize = GridSize
    MSC_Param.nKernelSize = -1
    
    # TCluTbl   
    CluTbl0 = TCluTbl()
    nMaxCluNum = 10000   #  enough ???
    CluTbl0.nCluNum = nMaxCluNum
    CluPropNum = 6   # const
    CluProps = np.zeros([CluTbl0.nCluNum, CluPropNum], 'double', 'C')
    # 如果不是C连续的内存，必须强制转换
    if not CluProps.flags['C_CONTIGUOUS']:
        CluProps = np.ascontiguous(CluProps, dtype = CluProps.dtype)
    CluTbl0.pClu = cast(CluProps.ctypes.data, POINTER(c_void_p))
    
    NewPts0 = BBoxData.transpose() # [3 * N] 
    NewPtsValidHeightIdx = (NewPts0[2,:]>0) # 选择地面一定高度的数据
    NewPts = NewPts0[:,NewPtsValidHeightIdx]
    PtNum = len(NewPts[0,:])
    rpx = np.zeros([1,PtNum], 'double', 'C')
    rpy = np.zeros([1,PtNum], 'double', 'C')
    weight = np.zeros([PtNum, 1], 'double', 'C')
    rpx[0,:] = NewPts[0, :]
    rpy[0,:] = NewPts[1, :]
    rpz = NewPts[2, :]
    rpx = rpx.transpose()
    rpy = rpy.transpose()
    
    if CluterPlaneFlag == 1:
        tmz = rpz
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpy.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    elif CluterPlaneFlag == 2:
        tmz = rpy
        weight = tmz / 5
        weight = np.power(weight , 5)   # large value will enhance local peak (sometimes lead to false detection), while a smaller value might miss true cluster
        MSC_Param.nKernelSize = KernelSize
        CluLabel = np.zeros([PtNum, 1], 'int', 'C')  ##
        CluTbl0.nCluNum = nMaxCluNum
        # LibMeanShift.dll
        if len(FileName) == 0: # OffLine mode
            LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)
        else: # online
            LibMS = ctypes.windll.LoadLibrary(OffLineLibMeanShiftDir)
        res = LibMS.MeanShiftWeightedCluster2D(
              cast(rpx.ctypes.data, POINTER(c_double)),
              cast(rpz.ctypes.data, POINTER(c_double)),
              cast(weight.ctypes.data, POINTER(c_double)),
              PtNum,
              byref(MSC_Param),
              byref(CluTbl0),
              cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    
    # 选择有效的cluster 目标
    if (res>0):
        CluProps_V = np.zeros([res,CLU.CluPropNum])
        CluProps_Height = np.zeros([1,res]) # 保存每类数据高度
        CluProps_V[:,(CLU.iNUM-1,CLU.iMX-1,CLU.iMY-1,CLU.iANG-1,CLU.iL1-1,CLU.iL2-1)] = CluProps[0:res, :]

        CluLabel[CluLabel < 0] = PT_LABEL.OUTBUND
        CluPtNum = CluProps_V[:, CLU.iNUM-1] #
        mcx = CluProps_V[:, CLU.iMX-1]
        mcy = CluProps_V[:, CLU.iMY-1]
        CluNum = CluProps_V.shape[0]

        # Compute additional properties of clusters
        ObjSize = np.zeros([CluNum, 1])
        for nc in range(CluNum): ##
            pos = (CluLabel == (nc+1))  
            objx = np.array([rpx])[pos.transpose()]
            objy = np.array([rpy])[pos.transpose()]
            objz = np.array([rpz])[pos.transpose()]
            objw = np.array([weight])[pos.transpose()]
            dx = np.max(objx) - np.min(objx)
            dy = np.max(objy) - np.min(objy)

#            if (dx > 2.0) | (dy > 2.0):
#                print('obj of large size: nc = %d'%nc)

            if len(objx)==1:
                ObjSize[nc] = np.sqrt(np.var(objx) * np.var(objy) * np.var(objz))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz)
            else:
                ObjSize[nc] = np.sqrt(np.var(objx,ddof=1) * np.var(objy,ddof=1) * np.var(objz,ddof=1))
                CluProps_V[nc, CLU.iLX-1] = np.std(objx,ddof=1)
                CluProps_V[nc, CLU.iLY-1] = np.std(objy,ddof=1)
                CluProps_V[nc, CLU.iLZ-1] = np.std(objz,ddof=1)
            # Use weighted mean value of z, which provides a better
            # description about the height of objects
            CluProps_V[nc, CLU.iMZ-1] = np.sum(objz * objw) / np.sum(objw)
            CluProps_V[nc, CLU.iMX2-1] = np.mean(objx)
            CluProps_V[nc, CLU.iMY2-1] = np.mean(objy)
            CluProps_V[nc, CLU.iMZ2-1] = np.mean(objz)
            CluProps_Height[0][nc] = np.max(objz) # 保存每类数据高度

            # Calculate axis and orientation
            tmpx = objx - CluProps_V[nc, CLU.iMX-1]
            tmpy = objy - CluProps_V[nc, CLU.iMY-1]

            [L1, L2, Ang] = CAL_CalcCluAxisbyKL(tmpx, tmpy, KL_EPS)
            CluProps_V[nc, CLU.iANG-1] = Ang

            if True:
                [MajorAxis, MinorAxis] = CAL_CalcCluAxis(objx, objy, objz, objw)
                CluProps_V[nc, CLU.iL1-1] = MajorAxis / 2
                CluProps_V[nc, CLU.iL2-1] = MinorAxis / 2

        # Filter invalid clusters
        # Find good candidates from clusters
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
#        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight) & (CluProps_Height[0,:]>AboveHeightThod)  
        # 201903
#        ValidPos = (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        # 201909
#        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.035) & (CluProps_V[:, CLU.iLY-1] > 0.035) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  
        # 20190925
        ValidPos = (CluProps_V[:, CLU.iLX-1] > 0.025) & (CluProps_V[:, CLU.iLY-1] > 0.025) & (CluProps_V[:, CLU.iLZ-1] > 0.045) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)  


#        print('CluProps iL1 = {}'.format(CluProps_V[:, CLU.iL1-1]))
#        print('CluProps iLX = {}'.format(CluProps_V[:, CLU.iLX-1]))
#        print('CluProps iLY = {}'.format(CluProps_V[:, CLU.iLY-1]))
#        print('CluProps iLZ = {}'.format(CluProps_V[:, CLU.iLZ-1]))
#        print('CluProps iNUM = {}'.format(CluProps_V[:, CLU.iNUM-1]))
#        print('CluProps iMZ = {}'.format(CluProps_V[:, CLU.iMZ-1]))
        
        LabelRemap = np.array(range(0,CluNum))
        LabelRemap[~ValidPos] = PT_LABEL.UNDEFINED
        LabelRemap[CluProps_V[:, CLU.iNUM-1] <= 10] = PT_LABEL.NOISE
        numValidPos = 0
        for ii in range(ValidPos.shape[0]):
            if ValidPos[ii]==True:
                numValidPos=numValidPos+1
        LabelRemap[ValidPos] = np.array(range(1,numValidPos+1))  ## ValidPos index
        CluProps_V = CluProps_V[ValidPos, :] # 选择后的有效类别
        CluProps_Height = CluProps_Height[:,ValidPos] # 选择后的有效类别
        TmpPos = (CluLabel > 0)
        CluLabel[TmpPos] = LabelRemap[CluLabel[TmpPos]-1]
    else:
        CluProps_V = np.empty(shape=[0, CLU.CluPropNum-1])
        
    # 判断范围数据是否有人员目标
    PreHumanInfo = -1*np.ones([CluProps_V.shape[0], 4]) # 初始化检测目标信息
    for i_valid_obj in range(CluProps_V.shape[0]):
        # x,y,z
        PreHumanInfo[i_valid_obj,0] = i_valid_obj # 序号
        PreHumanInfo[i_valid_obj,1] = CluProps_V[i_valid_obj, CLU.iMX-1] # 初始目标位置信息
        PreHumanInfo[i_valid_obj,2] = CluProps_V[i_valid_obj, CLU.iMY-1]
        PreHumanInfo[i_valid_obj,3] = CluProps_V[i_valid_obj, CLU.iMZ-1]
    PreHumanInfo = np.array(PreHumanInfo)

    return PreHumanInfo    
    