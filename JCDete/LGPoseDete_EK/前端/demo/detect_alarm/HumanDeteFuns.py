import numpy as np
from matplotlib import path
from util.meanshift import MeanShiftClusterFuns
from util.CloudPointFuns import SavePt3D2Ply

import time

# =============================================
# 功能：超高等目标检测（聚类目标检测）
# =============================================
def sequence_frame_object_dete(SelectDeteArea, PreAreaState, CurDeteInfo, StayDeteFrameThod, CurValidFlag, SequenceDeteFrameMin = -1, SequenceDeteFrameMax = 10):
    '''
    function: 连续帧检测一定区域内目标信息
    
    input
    ------------
    SelectDeteArea(array) : 目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
    PreAreaState(array): 前一帧的检测状态
    CurDeteInfo(array) : 目标当前位置，[1 x 4]
    StayDeteFrameThod(num): 连续检测帧数阈值
    CurValidFlag(bool): 当前帧有效性，
    
    output
    ------------
    CurAreaState(array) : 当前帧的检测状态
    '''
    # detect frame limit
    
    CurAreaState = PreAreaState
#    print('PreAreaState = {}'.format(PreAreaState))
#    print('CurAreaState = {}'.format(CurAreaState))
    # previous frame detecte state
#    if (CurValidFlag == True) and (CurDeteInfo[0,0] == 1):
    if (CurValidFlag == True) and (CurDeteInfo[0] == 1):
            
#        print('valid detect.')
        # detect center area, [x, y, z]
        DetectObjLocFlag = 2 # if DetectObjLocFlag == 1, 使用区域的中心位置座作为检测位置
                             # if DetectObjLocFlag == 2, 使用检测目标的中心位置座作为检测位置
        if DetectObjLocFlag == 1:
            # 使用区域的中心位置座作为检测位置
            CurAreaState[3] = (SelectDeteArea[0]+SelectDeteArea[1])/2
            CurAreaState[4] = (SelectDeteArea[2]+SelectDeteArea[3])/2
            CurAreaState[5] = (SelectDeteArea[4]+SelectDeteArea[5])/2
        elif DetectObjLocFlag == 2: 
            # 使用检测目标的中心位置座作为检测位置
#            CurAreaState[3] = CurDeteInfo[0,1]
#            CurAreaState[4] = CurDeteInfo[0,2]
#            CurAreaState[5] = CurDeteInfo[0,3]

            CurAreaState[3] = CurDeteInfo[1]
            CurAreaState[4] = CurDeteInfo[2]
            CurAreaState[5] = CurDeteInfo[3]
            
        # detect number frame
        CurAreaState[2] = CurAreaState[2] + 1
        CurAreaState[2] = max(CurAreaState[2], SequenceDeteFrameMin)
        CurAreaState[2] = min(CurAreaState[2], SequenceDeteFrameMax)
        # detect valid satate
        if CurAreaState[2] >= StayDeteFrameThod:
            CurAreaState[6] = 1
        else:
            CurAreaState[6] = -1
    else:
        # detect number frame
        CurAreaState[2] = CurAreaState[2] - 1
        CurAreaState[2] = max(CurAreaState[2], SequenceDeteFrameMin)
        CurAreaState[2] = min(CurAreaState[2], SequenceDeteFrameMax)
        # detect valid satate
        if CurAreaState[2] >= StayDeteFrameThod:
            CurAreaState[6] = 1
        else:
            CurAreaState[6] = -1
            
#    print('CurAreaState = {}'.format(CurAreaState))
    
    return CurAreaState

def select_area_object_dete(SelectDeteArea, DeteAreaInnerDis, NewPts, FileName):
    '''
    function: 检测一定区域内目标信息
    
    input
    ------------
    SelectDeteArea(array) : 目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
    DeteAreaInnerDis: 边界区域内部一定区域
    NewPts(array) : 点云数据，[3 x N]
    FileName(str): 输入文件名称，
    
    output
    ------------
    PredHeightInfo(array) : 返回超高目标当前位置，[1 x 4], [validflag, x, y, z ]
    '''
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
        
#    print('AboveHeight_XY_SCALE = {}'.format(AboveHeight_XY_SCALE))
#    print('SelectDeteArea = {}'.format(SelectDeteArea))
#    print('AboveHeight_XY_SCALE = {}'.format(AboveHeight_XY_SCALE.shape[0]))
#    print('SelectDeteArea = {}'.format(SelectDeteArea.shape[0]))
    
    # cluster info
    PredHeightInfo = high_limiting_control_action(AboveHeight_XY_SCALE, NewPts, FileName, AboveHeightThod, ZaxisHeightScale)
    
    return PredHeightInfo
    

def high_limiting_control_action(AboveHeight_XY_SCALE, NewPts, FileName, AboveHeightThod, ZaxisHeightScale=0.25):
    '''
    function: 通过聚类方法判断目标是否存在
    
    input
    ------------
    AboveHeight_XY_SCALE(list) : 目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
    Pts(array) : 点云数据，[3 x N]
    FileName(str): 输入文件名称，
    AboveHeightThod(array): 限制高度，常数, eg: 3.0 (米)
    ZaxisHeightScale: 垂直方向范围
    
    output
    ------------
    PredHeightInfo(array) : 返回超高目标当前位置，[1 x 4], [validflag, x, y, z ]
    '''
    
    # selcet area
    TempAboveHeightPtsPos = (NewPts[0, :] > AboveHeight_XY_SCALE[0]) & (NewPts[0, :] < AboveHeight_XY_SCALE[1]) & \
                            (NewPts[1, :] > AboveHeight_XY_SCALE[2]) & (NewPts[1, :] < AboveHeight_XY_SCALE[3]) & \
                            (NewPts[2, :] > AboveHeight_XY_SCALE[4]) & (NewPts[2, :] < AboveHeight_XY_SCALE[5])  # 
    # select pts
    TempAboveHeightPts = NewPts[:, TempAboveHeightPtsPos]
    PtsTempAllArea = TempAboveHeightPts.transpose() # [N x 3]
    
#    SavePt3D2Ply(str(time.time()) + 'new_pts.ply', PtsTempAllArea, 'XYZ')
#    print('TempAboveHeightPtsPos = {}'.format(TempAboveHeightPts.shape))
    
    # cluster info 
    PredHeightInfo = MeanShiftClusterFuns.CalcScaleAreaClusterInfo(PtsTempAllArea, FileName, AboveHeightThod, ZaxisHeightScale)  # input [N, 3]
  
    return PredHeightInfo


# =============================================
# 功能：人员目标头部信息检测
# =============================================
def human_cluster_dete(SelectDeteArea, NewPts, FileName):
    '''
    function: 通过聚类方法判断人员目标是否存在
    
    input
    ------------
    SelectDeteArea(list) : 目标区域，[x1,y1,x2,y2,x3,y3,x4,y4,z1,z2,validflag]
    Pts(array) : 点云数据，[3 x N]
    FileName(str): 输入文件名称，
    
    output
    ------------
    PredINTERNALSUPERVISORInfo(array) : 返回目标当前位置，[M x 4], [index, x, y, z ]
    '''
    # 初始化内部监管人员信息
    PredINTERNALSUPERVISORInfo = []
    
    # 获取区域内数据点
    SelectPts = np.zeros([3,1]) # [3 x N]
    for i_area in range(np.size(SelectDeteArea,0)):
        if SelectDeteArea[i_area,-1] == 1: # 目标区域是否有效
            SelectRegionXY = SelectDeteArea[i_area,:-3] # region XY plane limitation
            SelectRegionZ = SelectDeteArea[i_area,-3:-1] # region Z-axis limitation
            CurrPoly = SelectRegionXY.reshape([int(len(SelectRegionXY)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
            pCurrPoly = path.Path(CurrPoly)
            binAreaAll = pCurrPoly.contains_points(np.transpose(NewPts[0:2,])) # limit xy 
            pos = binAreaAll & (NewPts[2, :] >= SelectRegionZ[0]) & (NewPts[2, :] <= SelectRegionZ[1]) # limit z 
            CurSelectRegionPts = NewPts[:, pos] # NewPtsTrangleRegion, [3, N1]
            if CurSelectRegionPts.shape[0]>0:
                SelectPts = np.hstack((SelectPts, CurSelectRegionPts))
                
    # 聚类检测人员目标
#    print('SelectDeteArea = {}'.format(SelectDeteArea))
#    SavePt3D2Ply(str(time.time()) + 'new_pts.ply', SelectPts.transpose(), 'XYZ')
    
    if np.size(SelectPts,1)>10:
        SelectPts = SelectPts.transpose()
        PredINTERNALSUPERVISORInfo = MeanShiftClusterFuns.CalcHumanClusterInfo(SelectPts, FileName)
    PredINTERNALSUPERVISORInfo = np.array(PredINTERNALSUPERVISORInfo)
    
    return PredINTERNALSUPERVISORInfo


# =============================================
# 功能：人员超高检测
#      针对人员超高行为，单独计算
# =============================================
def alarm_detet_above_height_object(SelectDeteArea, DeteAreaInnerDis, NewPts, FileName, InputMinCluPtNumThod = 200, PtsClusterDistrSelect = 1):
    """
    功能：对超高告警目标位置进行检测
    输入：
        SelectDeteArea(array) : 目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
        DeteAreaInnerDis: 边界区域内部一定区域
        NewPts(array) : 点云数据，[3 x N]
        FileName(str): 输入文件名称，
    输出：各个超高目标的位置信息
    """
    # 是否使用靠近边界目标点数不同处理
    UsingInnerBoundRangeFlag = False
    
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
    PtsTempAllArea = TempAboveHeightPts.transpose() # [N x 3]
    
#    SavePt3D2Ply(str(time.time()) + 'new_pts.ply', PtsTempAllArea, 'XYZ')
#    print('TempAboveHeightPtsPos = {}'.format(TempAboveHeightPts.shape))
    
    # cluster info 
#    AboveHeightInfo = MeanShiftClusterFuns.CalcScaleAreaAboveHeightClusterInfo(PtsTempAllArea, FileName, AboveHeightThod, ZaxisHeightScale = ZaxisHeightScale, MinCluPtNumThod = InputMinCluPtNumThod)  # input [N, 3]
    # 区分靠近/远离墙体边界目标点属性
    if UsingInnerBoundRangeFlag: # 
        CurValidAreaXYScaleInnerDis = 0.2 # 靠近边界/远离边界
        CurValidAreaXYScale = np.zeros([6,1])
        CurValidAreaXYScale[0] = AboveHeight_XY_SCALE[0] + CurValidAreaXYScaleInnerDis
        CurValidAreaXYScale[1] = AboveHeight_XY_SCALE[1] - CurValidAreaXYScaleInnerDis
        CurValidAreaXYScale[2] = AboveHeight_XY_SCALE[2] + CurValidAreaXYScaleInnerDis
        CurValidAreaXYScale[3] = AboveHeight_XY_SCALE[3] - CurValidAreaXYScaleInnerDis
        CurValidAreaXYScale[4] = AboveHeight_XY_SCALE[4]
        CurValidAreaXYScale[5] = AboveHeight_XY_SCALE[5]
    else:
        CurValidAreaXYScale = None
    AboveHeightInfo = MeanShiftClusterFuns.CalcScaleAreaAboveHeightClusterInfo(PtsTempAllArea, FileName, AboveHeightThod, ZaxisHeightScale = ZaxisHeightScale, MinCluPtNumThod = InputMinCluPtNumThod, InnerBoundRange=CurValidAreaXYScale, PtsClusterDistrSelect = PtsClusterDistrSelect)  # input [N, 3]


    return AboveHeightInfo
    
    
    

    