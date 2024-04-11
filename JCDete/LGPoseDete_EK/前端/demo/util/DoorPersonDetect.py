# -*- encoding: utf-8 -*-
'''
@File    :   DoorDetect.py
@Time    :   2020/10/28 11:28:19
@Author  :   Yangch 
@Version :   1.0
'''

import ctypes
from ctypes import *
import numpy as np
import datetime
from matplotlib import path

from util.meanshift import MeanShiftClusterFuns
from util.CloudPointFuns import LoadPointLabelIndex, SavePt3D2Ply

from config import online_offline_type, log_info

################ LibMeanShift.dll 文件地址 ################
if online_offline_type == 'OnLine':
    OnLineLibMeanShiftDir = "demo/util/meanshift/demo/LibMeanShift.dll"
    OffLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll"
else:
    OnLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll" # "demo/util/meanshift/demo/LibMeanShift.dll"
    OffLineLibMeanShiftDir = "util/meanshift/demo/LibMeanShift.dll"

    
def alarm_detet_door_object(SelectDeteArea, DeteAreaInnerDis, NewPts, FileName, InputMinCluPtNumThod = 200, PtsClusterDistrSelect = 1):
    """
    功能：对超高告警目标位置进行检测
    输入：
        SelectDeteArea(array) : 目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
        DeteAreaInnerDis: 边界区域内部一定区域
        NewPts(array) : 点云数据，[3 x N]
        FileName(str): 输入文件名称，
    输出：各个超高目标的位置信息
    """
    # 边缘一定区域内选择数据
    AboveHeightDeteBedInnerDis = DeteAreaInnerDis # 划定区域边缘内距离
  
    # 选择一定区域内 AboveHeight 有效的目标点
    AboveHeight_XY_SCALE = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
    AboveHeightThod = 0 # 初始化超高阈值
    ZaxisHeightScale = 0.25 # 初始化垂直方向范围
    if SelectDeteArea.shape[0] == AboveHeight_XY_SCALE.shape[0]:
        AboveHeight_XY_SCALE[0] = SelectDeteArea[0] + AboveHeightDeteBedInnerDis
        AboveHeight_XY_SCALE[1] = SelectDeteArea[1] - AboveHeightDeteBedInnerDis
        AboveHeight_XY_SCALE[2] = SelectDeteArea[2] + AboveHeightDeteBedInnerDis
        AboveHeight_XY_SCALE[3] = SelectDeteArea[3] - AboveHeightDeteBedInnerDis
        AboveHeight_XY_SCALE[4] = SelectDeteArea[4]
        AboveHeight_XY_SCALE[5] = SelectDeteArea[5]
        AboveHeightThod = SelectDeteArea[5]
        ZaxisHeightScale = SelectDeteArea[5] - SelectDeteArea[4]
    else:
        print('SelectDeteArea size not 6.')
    
    # selcet area
    TempAboveHeightPtsPos = (NewPts[0, :] > AboveHeight_XY_SCALE[0]) & (NewPts[0, :] < AboveHeight_XY_SCALE[1]) & \
                            (NewPts[1, :] > AboveHeight_XY_SCALE[2]) & (NewPts[1, :] < AboveHeight_XY_SCALE[3]) & \
                            (NewPts[2, :] > AboveHeight_XY_SCALE[4]) & (NewPts[2, :] < AboveHeight_XY_SCALE[5])  # 
    # select pts
    TempAboveHeightPts = NewPts[:, TempAboveHeightPtsPos]
    PtsTempAllArea = TempAboveHeightPts # [3 x N]
    # DoorAreaCluster
    people_num, people_loc = DoorAreaCluster(PtsTempAllArea, dist_threshold=0.4)  # input [3 x N]
    if people_num == 0:
        AboveHeightInfo = np.empty([0,4])
    else:
        AboveHeightInfo = np.concatenate((np.ones([people_num,1]), people_loc), axis=1) # [M x 3], [x,y,z]
    
    return AboveHeightInfo

def DoorAreaCluster(NewPts, dist_threshold=0.3, MinPeopleHeight=0.6, MinCluPtNum=250):
    '''
    input: NewPts:点云数据， 3 x n 格式， 数据类型：array
           dist_threshold：两个目标间（即聚类中心间）的最小距离，单位（米），数据类型：float
        
    output: people_num: 指定区域内的目标数量，数据类型：int
            people_loc：目标的空间位置，n x 3 格式([x, y, z])，数据类型:array
    '''
    # 默认参数 
    KL_EPS = 1.0e-10      # constant in KL analysis
    GridSize = 0.025
    KernelSizeSet = np.array([6])
    theAxis = np.array([-20.0, 20.0, -20.0, 20.0]) # [xmin, xmax, ymin, ymax], ??????
    
    CLU = MeanShiftClusterFuns.MeanShiftCluster2D_LoadIndex()
    PT_LABEL = LoadPointLabelIndex()
    LibMS = ctypes.windll.LoadLibrary(OnLineLibMeanShiftDir)

    class TCluParam(Structure):
        _fields_ = [("bx1", c_double),
                    ("by1", c_double),
                    ("bx2", c_double),
                    ("by2", c_double),
                    ("GridSize", c_double),
                    ("nKernelSize", c_int)]

    MSC_Param = TCluParam()
    MSC_Param.bx1 = theAxis[0]
    MSC_Param.bx2 = theAxis[1]
    MSC_Param.by1 = theAxis[2]
    MSC_Param.by2 = theAxis[3]
    MSC_Param.GridSize = GridSize
    MSC_Param.nKernelSize = -1

    # Prepare output CluTbl
    class TCluTbl(Structure):
        _fields_ = [("nCluNum", c_int),
                    ("pClu", POINTER(c_void_p))]

    CluTbl0 = TCluTbl()
    nMaxCluNum = 10000   
    CluTbl0.nCluNum = nMaxCluNum
    CluPropNum = 6   # const
    CluProps = np.zeros([CluTbl0.nCluNum, CluPropNum], 'double', 'C')
    # 如果不是C连续的内存，必须强制转换
    if not CluProps.flags['C_CONTIGUOUS']:
        CluProps = np.ascontiguous(CluProps, dtype = CluProps.dtype)
    CluTbl0.pClu = cast(CluProps.ctypes.data, POINTER(c_void_p))

    rpz = NewPts[2, :]
    PtNum = len(rpz)
    rpx = np.zeros([1,PtNum], 'double', 'C')
    rpy = np.zeros([1,PtNum], 'double', 'C')
    weight = np.zeros([PtNum, 1], 'double', 'C')

    rpx[0,:] = NewPts[0, :]
    rpy[0,:] = NewPts[1, :]
    rpx = rpx.transpose()
    rpy = rpy.transpose()
    tmz = rpz
    weight = tmz / (1 + np.power(tmz / 3.00 , 10))
    weight = np.power(weight , 30)   

    MSC_Param.nKernelSize = KernelSizeSet
    CluLabel = np.zeros([PtNum, 1], 'int', 'C')  
    CluTbl0.nCluNum = nMaxCluNum
    res = LibMS.MeanShiftWeightedCluster2D(
                cast(rpx.ctypes.data, POINTER(c_double)),
                cast(rpy.ctypes.data, POINTER(c_double)),
                cast(weight.ctypes.data, POINTER(c_double)),
                PtNum,
                byref(MSC_Param),
                byref(CluTbl0),
                cast(CluLabel.ctypes.data, POINTER(c_int))
            )
    # print('Res=', res)
    people_num = 0
    if(res > 0):
        CluProps_V = np.zeros([res,CLU.CluPropNum])
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

            if (dx > 2.0) | (dy > 2.0):
                print('obj of large size: nc = %d'%nc)

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
            # Calculate axis and orientation
            tmpx = objx - CluProps_V[nc, CLU.iMX-1]
            tmpy = objy - CluProps_V[nc, CLU.iMY-1]

            [L1, L2, Ang] = MeanShiftClusterFuns.CAL_CalcCluAxisbyKL(tmpx, tmpy, KL_EPS)
            CluProps_V[nc, CLU.iANG-1] = Ang

            if True:
                [MajorAxis, MinorAxis] = MeanShiftClusterFuns.CAL_CalcCluAxis(objx, objy, objz, objw)
                CluProps_V[nc, CLU.iL1-1] = MajorAxis / 2
                CluProps_V[nc, CLU.iL2-1] = MinorAxis / 2

        ValidPos = (CluProps_V[:, CLU.iL1-1] > 0.10) & (CluProps_V[:, CLU.iLZ-1] > 0.05) & (CluPtNum >= MinCluPtNum) & (CluProps_V[:, CLU.iMZ-1] > MinPeopleHeight)

        LabelRemap = np.array(range(0,CluNum))
        LabelRemap[~ValidPos] = PT_LABEL.UNDEFINED
        LabelRemap[CluProps_V[:, CLU.iNUM-1] <= 10] = PT_LABEL.NOISE
        numValidPos = 0
        for ii in range(ValidPos.shape[0]):
            if ValidPos[ii]==True:
                numValidPos=numValidPos+1
        LabelRemap[ValidPos] = np.array(range(1,numValidPos+1))  ## ValidPos index
        CluProps_V = CluProps_V[ValidPos, :]
        TmpPos = (CluLabel > 0)
        CluLabel[TmpPos] = LabelRemap[CluLabel[TmpPos]-1]
    
        CluPtNum = CluProps_V[:, CLU.iNUM-1]
        mcx = CluProps_V[:, CLU.iMX-1]
        mcy = CluProps_V[:, CLU.iMY-1]
        mcz = CluProps_V[:, CLU.iMZ-1]
        CluNum = CluProps_V.shape[0]
        CluTbl1 = CluProps_V
        
        people_loc = []
        Cpt1 = (CluTbl1[:, (CLU.iMX-1,CLU.iMY-1,CLU.iMZ-1)]).transpose()
        if Cpt1.shape[1] > 0:
            [MinDist, MinIdx, DistMat] = MeanShiftClusterFuns.CLP_PointSetDist(Cpt1, Cpt1)
            clu_list = Cpt1.transpose()
            people_loc = []
            excep_list=[]

            #去除距离太近的聚类中心
            # dist_threshold = 0.9     #距离阈值0.3米
            for k in range(clu_list.shape[0]):
                excep_mark = 0 #是否排除当前聚类中心
                outbound_pos = (DistMat[k] < dist_threshold)
                outbound_num = str(outbound_pos.tolist()).count('True') #距离小于阈值的聚类中心数量
                if outbound_num > 1:
                    out_index = np.where(outbound_pos == True)
                    out_index_list = np.where(out_index[0] != k) #当前聚类中心本身以外的聚类中心索引
                    for n in range(out_index_list[0].shape[0]):
                        if str(excep_list).count(str(out_index_list[0][n])) == 0:  #存在新的未去除的邻近中心
                            excep_list.append(k)  #将新的邻近中心加入排除列表
                            # excep_list.append(out_index_list[0][n])
                            excep_mark = 1  
                            break 
                if excep_mark:
                    continue
                people_loc.append(clu_list[k])
            
        people_loc = np.array(people_loc)    
        people_num = people_loc.shape[0]
        save_cluInfo_ply = False
        if save_cluInfo_ply:
            NewPts_T = NewPts.transpose()
            color_pts = np.ones(NewPts_T.shape)*200
            ply_pts = np.column_stack((NewPts_T, color_pts))
            color_center = np.ones(people_loc.shape)
            color_center[:] = [255,0,0]
            ply_center = np.column_stack((people_loc, color_center))
            save_pts = np.row_stack((ply_pts, ply_center))

            center_num = people_loc.shape[0]
            datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            save_name = 'temp/' + datetime_str + '_centerNum_' + str(center_num) + '_cluInfo.ply'
            SavePt3D2Ply(save_name, save_pts, VertexFormat='XYZ_RGB')

    else:
        people_loc = []

    return people_num, people_loc


