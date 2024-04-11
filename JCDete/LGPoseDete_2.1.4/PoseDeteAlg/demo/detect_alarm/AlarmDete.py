# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:46:04 2020

@author: HYD
"""

import numpy as np
import os
import time
import copy
from matplotlib import path
import logging as lgmsg

from detect_bbox.Bbox_Detecter import LoadDeteParam
from util.CloudPointFuns import Rot3D, SavePt3D2Ply
from detect_alarm.HumanDeteFuns import select_area_object_dete, sequence_frame_object_dete, human_cluster_dete, alarm_detet_above_height_object
from detect_alarm.AlarmDeteTime import CalInValidTime
from detect_alarm.AlarmString import DetectInfo
from detect_alarm.AreaFuns import CalIntersectAreaFromTwoArea, CalMultiAreaEdge, CalAboveHeightValidArea
from util.PolygonFuns import TransBboxToPoints
from util.ReadConfig import CalConfigMultiAreaEdge
from util.BGProcess import CalFGSubNoise
from util.SaveData import SavePtsAsDepth, AlarmCategoryIndexFun
from util.PtsFunsC import PointCloudFunsC
from util.AreaConnected import connected
from util.DoorPersonDetect import alarm_detet_door_object

from config import one_sensor_params, online_offline_type, cfg_case, log_info, alarm_level


# 连续帧累加速率设置
AlarmContinuousFrameUpRate = 1 # 连续帧累计上升速率
AlarmContinuousFrameDnRate = 2 # 连续帧累计下降速率

# 连续帧数计算参数
StayLocFrameThod = one_sensor_params['alarm_valid_frame_num'] # init: 10 = 5
# 默认的功能检测内部参数
DeteBbox_BedHeight = one_sensor_params['alarm_bed_height'] # [uint: m], init:0.6
# 告警时间边界的前后一段时间无效
AlarmValidNearMintueTime = one_sensor_params['alarm_valid_near_time'] # [uint: min]
# 超高聚类最少点云点数据
#AboveHeightCluPtsNumThod = one_sensor_params['pt_cluster_num_thod'] 

# 超高相关设置
#   超高区域类型信息
AboveHeightAreaDefaultType = 1 # 其中：1 表示默认两个超高区域，2 表示特殊超高区域（晾晒衣服区域），3 表示风仓门洞下方区域
AboveHeightAreaSecondType = 2
AboveHeightAreaThirdType = 3
AboveHeightAreaTypeIdx = [1,1,2,3] # 超高区域类型设置
                                   #   不同应用场景设置不同
                                   
#   超高上下限设置
AboveHeightUpDownThod = [0.1, 0.5] # [下限，上限], [0.1, 0.25],[0.1, 1.1]，[0.1, 0.5]，
AboveHeightUpDownThod2 = [0.4, -0.2] # [下限，上限], 备用超高阈值，如：晾晒衣服下方阈值
#   风仓门洞下方人员检测
AloneDoorAreaHeightFlag = True # 是否添加风仓门洞下方人员检测
AloneDoorAreaHeightRange = [1.0, 1.8] # 门洞人员检测点云高度范围

#   超高交接处重叠距离设置
AboveHeightCluObjToHumanDistThod = 1.0 # 聚类目标与人员目标距离阈值，单位：m，默认2.0米，3.0米
AboveHeightHumanObjHeightThod = 1.1 # 超高对应的人员高度，单位：m，默认1.2米
AboveHeightSensorOverlapDist = 0.5 # 传感器交接处重叠范围
#AboveHeightDeteAreaPtsInnerDis = 0.2 # 边界区域往内一部分，init：0.01/0.05/0.2
AboveHeightFGMethodFlag = 2 # AboveHeightFGMethodFlag = 1，只使用检测背景；AboveHeightFGMethodFlag = 2，检测背景+深度图像去噪；
# 特殊区域超高设置
AboveHeightSecondFlag = True # 是否启用特殊区域超高设置
AboveHeightSecondMethodFlag = 3 # 方法1：使用中间设置的一个高度计算是否超高，判断是否有两个高度超高
                                # 方法2：使用目标下方一定数据，判断连通性
                                # 方法3：屏蔽特殊区域超高目标
    


# 应用场景差别
CfgCaseIdx = cfg_case

# 在线/离线文件保存地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SaveResultDir = 'demo/result'
    SaveDataDepthDir = 'demo/result/depth'
    SaveDataPlyDir = 'demo/result/ply'
else:
    SaveResultDir = 'result'
    SaveDataDepthDir = 'result/depth'
    SaveDataPlyDir = 'result/ply'

# log 信息
ConfigDebugInfo = log_info['debug_info']
ConfigSaveInputPtsFlag = log_info['save_input_pts_flag']
# 保存告警原始数据
ConfigSaveDataInfo = log_info['save_data_info']
ConfigSaveDataTypeIndex = AlarmCategoryIndexFun()
ConfigSaveDataInfo_ABOVEHEIGHT = int(ConfigSaveDataInfo[ConfigSaveDataTypeIndex.ABOVEHEIGHTTHOD]) # 是否保存超高


def AlarmDete(ImgDeteBbox, ImgDeteLabel, ImgDeteScore, Pts, ImgWidth, ImgHeight, DepToCptIdx, WorldTime, ConfigInfo, PreDete, DebugFlag = 0, SrcPts = None, PtsSubBG=None):
    """
    功能：根据图像的检测结果，计算多种告警信息
    输入：
        ImgDeteBbox: 图像目标检测的 bbox 信息
        Pts: 点云数据
        ImgWidth, ImgHeight: 图像长宽
        WorldTime: 世界时间
        PreDete: 前一帧检测结果
        SrcPts: 原始点云数据
        
        说明：            
        其中：#告警id	"warningTypeModel":[
                    [1,"未在指定时间休息"], -
                    [2,"未在指定区域监督"], +
                    [4,"厕所区域异常"], +
                    [8,"窗户区域异常"], +
                    [16,"高度异常"], +
                    [32,"非休息时间休息"], -
                    [64,"进入三角区域"], -
                    [128,"内务不整"], -
                    [256,"打架"],
                    [512,"单人留仓"], +
                    [1024,"吊拉窗户"], +
                    [2048,"搭人梯"], +
                    [4096,"站在被子上做板报"]， +
                    [8192,"多人聚集"]，
                    [16384,"禁止遮挡传感器"]，
                    ]

    输出：
        AlarmDete: 告警检测结果
    """
    
    # 使用单帧结果
    OneFrmDeteInfoFlag = True # 使用单帧结果
    OneFrmContinuousFrameNum = 0
    OneFrmContinuousFrameMin = -1
    OneFrmContinuousFrameMax = 0
    OneFrmContinuousTime = 0
    
    ##########################################################################
    # 获取配置文件信息
    ##########################################################################
    # 配置文件是否被修改，【如果文件被修改，则和连续时间戳相关的告警记录重新开始】
    ConfigInfoUpdate = ConfigInfo['isUpdate']
    # 告警等级设置
    CurConfigInfoDeteAlarmLevelSetting = ConfigInfo['ROOM_INFO']['managelevel'] # 1/2/3
    CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
    # 现选择告警等级的配置信息
    #   单人留仓
    DeteAlarmLevelInfo_alone_continuous_frame_num = CurDeteAlarmLevelInfo['alone_continuous_frame_num']
    #   超高
    DeteAlarmLevelInfo_above_height_continuous_frame_num = CurDeteAlarmLevelInfo['above_height_continuous_frame_num'] # 暂时未使用
    DeteAlarmLevelInfo_above_height_edge_inner_dist = CurDeteAlarmLevelInfo['above_height_edge_inner_dist']
    DeteAlarmLevelInfo_above_height_cluster_pts_num = CurDeteAlarmLevelInfo['above_height_cluster_pts_num']
    DeteAlarmLevelInfo_above_height_cluster_distribute_select = CurDeteAlarmLevelInfo['above_height_cluster_distribute_select'] # 超高 目标点云分布参数设置
    DeteAlarmLevelInfo_above_height__using_human_dete = CurDeteAlarmLevelInfo['above_height_using_human_dete'] # 结合人员检测信息判断超高有效性
    
    #   内部监管
    DeteAlarmLevelInfo_internal_supervisor_continuous_time = CurDeteAlarmLevelInfo['internal_supervisor_continuous_time']
    #   厕所区域异常
    DeteAlarmLevelInfo_toilet_continuous_time = CurDeteAlarmLevelInfo['toilet_continuous_time'] # 
    # 未在指定时间休息
    WrongTimeNoLieFrameThod = CurDeteAlarmLevelInfo['wrong_time_nolie_frame_num'] # 连续帧数
    # 非休息时间休息
    WrongTimeLieFrameThod = CurDeteAlarmLevelInfo['wrong_time_lie_frame_num'] # 连续帧数

    # 更新告警参数设置
    AboveHeightDeteAreaPtsInnerDis = DeteAlarmLevelInfo_above_height_edge_inner_dist # 边界区域往内一部分，init：0.01
    AboveHeightCluPtsNumThod = DeteAlarmLevelInfo_above_height_cluster_pts_num
    
    
    # 保存文件名
    FileName = '' # 暂时设置空文件名
    # 保存点云相关数据时间
    SaveDataTimeName = str(time.time())
    
    # 模型 Label 信息
    BboxLabelNames = LoadDeteParam()
    # lying label 序号
    CurModelLyingLabel = 1
    if len(BboxLabelNames) == 1: # ['person']
        CurModelLyingLabel = 1
    if len(BboxLabelNames) == 2: # ['nolying', 'lying']
        CurModelLyingLabel = 1
    if len(BboxLabelNames) == 3: # ['standing', 'sitting', 'lying']
        CurModelLyingLabel = 2
        
    # 所有传感器检测区域
    MultiSensorDeteArea = CalConfigMultiAreaEdge(ConfigInfo)
#    print('MultiSensorDeteArea = {}'.format(MultiSensorDeteArea))
    
    # 传感器转换矩阵
    CurRoomId = ConfigInfo['ROOM_INFO']['roomid']
    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
    # 传感器检测范围， SENSORS_INFO
    CurSensorDeteArea = np.array(ConfigInfo['SENSORS_INFO'][CurSensorId]['Area'])
    CurSensorDeteArea = CurSensorDeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
    # 传感器转换矩阵，H
    H = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
#    print('CurSensorDeteArea = {}'.format(CurSensorDeteArea))
    
    # 获取新的旋转点云数据
    NewPts = Rot3D(H, Pts) #  H * Pts, [3 x 217088]
    if ConfigSaveInputPtsFlag == 1:
        InputRotPtsFileName = os.path.join(SaveResultDir, 'input_pts_rot_' + str(time.time()) + '.ply')
        SavePt3D2Ply(InputRotPtsFileName, NewPts.transpose(), 'XYZ')
    
    # 根据目标框获取人员空间位置
    ImageFilpFlag = 0 # 暂时不设置数据时候翻转
    ZAxisMaxHeight = H[2,3] # 图像旋转矩阵在 Z 方向上的偏移量
    PredInfo, PredInfoSrc, HUMANINFO_Dete = GetObjInfoFromBbox(ImgDeteBbox, ImgDeteLabel, ImgDeteScore, ImageFilpFlag, NewPts, DepToCptIdx, ZAxisMaxHeight)
    PredHumanObjHeight = PredInfoSrc[:,5] # 目标检测目标的最高高度
    
    # 功能检测
    #   功能类型：
    #       类型 1：根据目标人员【不进行姿态判定】的中心点和区域判断时候告警，包括：厕所区域异常/窗户区域异常/
    #       类型 2：根据目标区域的点云数据判断告警，包括：高度异常/吊拉窗户/搭人梯/站在被子上做板报/进入三角区域
    #       类型 3：根据目标人员姿态类型，中心点和区域判断告警，如：未在指定时间休息/非休息时间休息
    #       类型 4：其他告警类型
    #       类型 5：前端不检测告警，只发送当前检测结果到服务器端融合多传感器后进行检测
    
    # 当前帧 各告警检测信息，即上一帧的检测信息
    PredBedPos = PreDete.BED # PredBedPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredBedPos[:,[1,6]] = -1
    
    PredINTERNALSUPERVISORPos = [] # PredINTERNALSUPERVISORPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    
    PredTOILETPos = PreDete.TOILET # PredTOILETPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredTOILETPos[:,[1,6]] = -1 # 初始化姿态和有效标记为 -1
    
    PredWINDOWPos = PreDete.WINDOW # PredWINDOWPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredWINDOWPos[:,[1,6]] = -1
    
#    PredAboveHeight = [] # PredAboveHeight, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredAboveHeight = PreDete.ABOVEHEIGHT # PredAboveHeight, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    
    PredWRONGTIMELIE = [] # PredWRONGTIMELIE, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredWRONGTIMELIE_Pre = PreDete.WRONGTIMELIE # 前一帧 WRONGTIMELIE 信息
    PredWRONGTIMELIE_PreValid = -1*np.ones([1,PredWRONGTIMELIE_Pre.shape[0]]) # 初始化前一帧 WRONGTIMELIE 状态

    PredINTRIANGLEREGION = [] # PredINTRIANGLEREGION, eg:[ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredTOILET_STARTTIME = PreDete.TOILET_STARTTIME
    PredWINDOW_STARTTIME = PreDete.WINDOW_STARTTIME
    PredHOUSEKEEP = PreDete.HOUSEKEEP # eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    
    PredHAULWINDOW = PreDete.HAULWINDOW # PredHAULWINDOW, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredHAULWINDOW[:,[1,6]] = -1 # 初始化姿态和有效标记为 -1
    
    PredBUILDLADDER = PreDete.BUILDLADDER # PredBUILDLADDER, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredBUILDLADDER[:,[1,6]] = -1
    
    PredSTANDQUILT = PreDete.STANDQUILT # PredSTANDQUILT, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    PredSTANDQUILT[:,[1,6]] = -1
    
    PredHUMANINFO = HUMANINFO_Dete # eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    
    #    [1,"未在指定时间休息"],
    CurAlarmTypeId = 1 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        # 设置不同使用场景
        if CfgCaseIdx == 2: # CfgCaseIdx = 2.认证版本场景
            CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
            if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
                # 配置文件设置参数
                DeteArea = np.array(CurALARM_INFO['DeteArea'])
                DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
                DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
                DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
                # 其他参数
                DeteAreaPtsInnerDis = 0.1# 往内一定区域
                DeteMinHeight = DeteBbox_BedHeight + 0.5 # 龙岗厕所区域设置高度：0.9， 为了排除桶的干扰
                NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
                AlarmFrameNumThod = 1 # 连续帧才启动有效计算，【由于站立人员易在多传感器之间走动，这是设置连续帧为2】
                Bed_MinFrameNum = -1
                Bed_MaxFrameNum = AlarmFrameNumThod+3
                # 是否只使用单帧结果
                if OneFrmDeteInfoFlag == True:
                    DeteContinuousTime = OneFrmContinuousTime
                    AlarmFrameNumThod = OneFrmContinuousFrameNum
                    Bed_MinFrameNum = OneFrmContinuousFrameMin
                    Bed_MaxFrameNum = OneFrmContinuousFrameMax
                
                # 判断是否在告警时间内
                InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
                if InPeriodTimeFlag == 1:
                    # 先判断目标区域的点云检测结果
                    PredAreaPtsLoc = np.zeros([1, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                    PredBedPos_Cur = -1 * np.ones([1, PredBedPos.shape[1]])
                    PredBedPos_Cur[0,:] = PredBedPos[0,:] # 选择第一个目标位置1

                    # 当前区域是否有未躺下人数
                    if True: # 
                        # 1,判断目标是否在当前传感器检测范围内
                        #   判断目标是否在当前传感器检测范围内
                        #      在告警范围内
                        #           躺下
                        #                不告警
                        #           不躺下
                        #                告警
                        #      否则，告警
                        #   否则，不告警
#                        print('PredInfo = ', PredInfo)
#                        print('DeteMinHeight = ', DeteMinHeight)
                        
                        NumPerson = PredInfo.shape[0] # 检测到的目标人数
                        PredAlarmValidObjLoc = []
                        PredAlarmValidObjNum = 0
                        CurPredAllObjAlarmValidFlag = 0 # 是否存在需要告警的目标
                        for nump in range(NumPerson):
                            # 判断目标位置是否在当前传感器检测范围内
                            CurPredObjValidFlag = -1 # 是否在传感器检测区域内
                            for i_sensor_area in range(CurSensorDeteArea.shape[0]):
                                
                                # 遍历告警功能检测区域
                                for i_area in range(DeteArea.shape[0]):
                                    # 计算各传感器检测的区域
                                    ValidDeteArea = CurSensorDeteArea[i_sensor_area]
                                    CurMultiAreaEdge = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                                    # CurDeteArea
                                    CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                                    if len(ValidDeteArea) > 0: # 存在交叉区域
                                        ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis) # 计算超高+边界区域的有效区域
                                        CurDeteArea[0] = ValidDeteArea[0] # xmin
                                        CurDeteArea[1] = ValidDeteArea[1] # xmax
                                        CurDeteArea[2] = ValidDeteArea[2] # ymin
                                        CurDeteArea[3] = ValidDeteArea[3] # ymax
                                        CurDeteArea[4] = 0 # zmin
                                        CurDeteArea[5] = 0 # zmax
                                    # 是否在检测范围内
                                    if PredInfo[nump][0] > CurDeteArea[0] and PredInfo[nump][0] < CurDeteArea[1] \
                                        and PredInfo[nump][1] > CurDeteArea[2] and PredInfo[nump][1] < CurDeteArea[3]:
                                        CurPredObjValidFlag = 1
                                        continue

#                                if PredInfo[nump][0] > CurSensorDeteArea[i_sensor_area][0] and PredInfo[nump][0] < CurSensorDeteArea[i_sensor_area][1] \
#                                    and PredInfo[nump][1] > CurSensorDeteArea[i_sensor_area][2] and PredInfo[nump][1] < CurSensorDeteArea[i_sensor_area][3]:
#                                    CurPredObjValidFlag = 1
#                                    continue
                                
                            # 此处修改只针对 ZT-RZ 版本，20200515
                            if CurPredObjValidFlag == 1: # 目标位置在当前传感器检测范围内
                                CurPredObjAlarmValidFlag = -1 # 是否在告警区域内
                                for i_area in range(DeteArea.shape[0]):
                                    if PredInfo[nump][0] > DeteArea[i_area][0] and PredInfo[nump][0] < DeteArea[i_area][1] \
                                        and PredInfo[nump][1] > DeteArea[i_area][2] and PredInfo[nump][1] < DeteArea[i_area][3]: # 目标是否在告警区域内 
                                        CurPredObjAlarmValidFlag = 1
                                        continue
                                if CurPredObjAlarmValidFlag == 1: #在告警区域内
                                    if ImgDeteLabel[nump] != CurModelLyingLabel: # 如果在指定区域非躺下,则告警， 未增加‘一定高度有数据点’： PredAreaPtsLoc[i_area, 0] == 1
#                                    if (ImgDeteLabel[nump] != CurModelLyingLabel) and (PredInfo[nump][2] < DeteMinHeight): # 如果在指定区域非躺下,则告警， 未增加‘一定高度有数据点’： PredAreaPtsLoc[i_area, 0] == 1
                                        PredBedPos_Cur[0,1] = ImgDeteLabel[nump]
                                        PredBedPos_Cur[0,3:6] = PredInfo[nump]
                                        PredAreaPtsLoc[0, 0] = 1
                                        PredAreaPtsLoc[0, 1:] = PredInfo[nump]
                                        CurPredAllObjAlarmValidFlag = 1 # 存在需要告警的目标
                                        # 真实的多目标人员位置
                                        PredAlarmValidObjLoc.append([PredAlarmValidObjNum, ImgDeteLabel[nump], 0, PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2], -1])
                                        PredAlarmValidObjNum = PredAlarmValidObjNum + 1
                                    else:# 如果在指定区域躺下,则不告警
                                        PredAreaPtsLoc[0, 0] = 0
                                        CurPredAllObjAlarmValidFlag = 0 # 存在需要告警的目标
                                    if (ImgDeteLabel[nump] == CurModelLyingLabel and PredInfo[nump][2] < DeteMinHeight-0.3): # 排除躺下检测不准的情况
                                        PredAreaPtsLoc[0, 0] = 0
                                        CurPredAllObjAlarmValidFlag = 0 # 存在需要告警的目标
                                else: #如果目标不在告警区域内，则告警
                                    PredBedPos_Cur[0,1] = ImgDeteLabel[nump]
                                    PredBedPos_Cur[0,3:6] = PredInfo[nump]
                                    PredAreaPtsLoc[0, 0] = 1
                                    PredAreaPtsLoc[0, 1:] = PredInfo[nump]
                                    CurPredAllObjAlarmValidFlag = 1 # 存在需要告警的目标
                                    # 真实的多目标人员位置
                                    PredAlarmValidObjLoc.append([PredAlarmValidObjNum, ImgDeteLabel[nump], 0, PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2], -1])
                                    PredAlarmValidObjNum = PredAlarmValidObjNum + 1
                            else: # 目标位置不在当前传感器检测范围内，不告警
                                PredAreaPtsLoc[0, 0] = 0
                        if NumPerson == 0: # 未检测到目标，不告警
                            PredAreaPtsLoc[0, 0] = 0
                    # 根据多帧结果判断是否告警
                    AlarmValidFlag = True
                    PredAreaPtsLoc[0, 0] = CurPredAllObjAlarmValidFlag
                    PredBedPos_Step1 = AlarmDete_Pts(PredBedPos_Cur, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = Bed_MinFrameNum, MaxFrameNum = Bed_MaxFrameNum)
                    if (PredAlarmValidObjNum > 1): # 出现多个目标，则返回所有目标位置
                        PredBedPos = np.array(PredAlarmValidObjLoc)
                        PredBedPos[:, 1:3] = PredBedPos_Step1[0, 1:3]
                        PredBedPos[:, -1] = PredBedPos_Step1[0, -1]
                    else:
                        PredBedPos = PredBedPos_Step1
                else: # 不在检测时间段内
                    # 根据多帧结果判断是否告警
                    PredBedPos_Cur = -1 * np.ones([1, PredBedPos.shape[1]])
                    PredBedPos_Cur[0,:] = PredBedPos[0,:] # 选择第一个目标位置
                    PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                    AlarmValidFlag = False
                    PredBedPos = AlarmDete_Pts(PredBedPos_Cur, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = Bed_MinFrameNum, MaxFrameNum = Bed_MaxFrameNum)
            else:
                PredBedPos[:,[1,6]] = -1 # 设置告警状态为 -1
                
        else: # 其他场景
            CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
            if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
                # 配置文件设置参数
                DeteArea = np.array(CurALARM_INFO['DeteArea'])
                DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
                DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
                DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
                # 其他参数
                DeteAreaPtsInnerDis = 0.1# 往内一定区域, 0.1
                DeteMinHeight = DeteBbox_BedHeight + 0.35 # 龙岗厕所区域设置高度：0.9， 为了排除桶的干扰，DeteBbox_BedHeight + 0.5
                NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
                AlarmFrameNumThod = StayLocFrameThod # 连续 5 帧才启动有效计算
                Bed_MinFrameNum = -1
                Bed_MaxFrameNum = AlarmFrameNumThod + 5
                # 是否只使用单帧结果
                if OneFrmDeteInfoFlag == True:
                    DeteContinuousTime = OneFrmContinuousTime
                    AlarmFrameNumThod = OneFrmContinuousFrameNum
                    Bed_MinFrameNum = OneFrmContinuousFrameMin
                    Bed_MaxFrameNum = OneFrmContinuousFrameMax
                
                # 判断是否在告警时间内
                InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
                # 判断文件是否被修改了
                if ConfigInfoUpdate == True: # 如果文件有更新，则重置告警结果和告警起始时间
                    NumBed = DeteArea.shape[0]
                    PredBedPos = -1 * np.ones([NumBed, 7]) # 设置告警状态重置
                    print('  PredBedPos re-setting.')
                else:
                    if InPeriodTimeFlag == 1:
                        # 先判断目标区域的点云检测结果
                        PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        for i_area in range(PredAreaPtsLoc.shape[0]):
                            # 计算各传感器检测的区域
                            ValidDeteArea = CalIntersectAreaFromTwoArea(DeteArea[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                            CurMultiAreaEdge = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                            # CurDeteArea
                            CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                            if len(ValidDeteArea) > 0: # 存在交叉区域
                                ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis) # 计算超高+边界区域的有效区域
                                CurDeteArea[0] = ValidDeteArea[0] # xmin
                                CurDeteArea[1] = ValidDeteArea[1] # xmax
                                CurDeteArea[2] = ValidDeteArea[2] # ymin
                                CurDeteArea[3] = ValidDeteArea[3] # ymax
                                CurDeteArea[4] = DeteMinHeight # zmin
                                CurDeteArea[5] = DeteMaxHeight # zmax
        #                    print('CurDeteArea = {}'.format(CurDeteArea))
                            CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName) #
                            if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                                PredAreaPtsLoc[i_area, :] = CurAlarmDeteHeightInfo
    
                            # 当前区域是否有未躺下人数
                            if True:
                                CurBedAreaValidFlag = 0
                                NumPerson = PredInfo.shape[0] # 检测到的目标人数
                                for nump in range(NumPerson):
                                    PredBedPos[i_area, 0] = i_area
#                                    if PredInfo[nump][0] > DeteArea[i_area][0] and PredInfo[nump][0] < DeteArea[i_area][1] \
#                                        and PredInfo[nump][1] > DeteArea[i_area][2] and PredInfo[nump][1] < DeteArea[i_area][3] \
#                                        and ImgDeteLabel[nump] != CurModelLyingLabel and PredAreaPtsLoc[i_area, 0] == 1: # 目标在区域内+目标不是lying+目标一定高度有数据点
                                    if PredInfo[nump][0] > CurDeteArea[0] and PredInfo[nump][0] < CurDeteArea[1] \
                                        and PredInfo[nump][1] > CurDeteArea[2] and PredInfo[nump][1] < CurDeteArea[3] \
                                        and ImgDeteLabel[nump] != CurModelLyingLabel and PredAreaPtsLoc[i_area, 0] == 1: # 目标在区域内+目标不是lying+目标一定高度有数据点, 使用有效区域内目标CurDeteArea
                                        
                                        PredBedPos[i_area,1] = ImgDeteLabel[nump]
                                        PredBedPos[i_area,3:6] = PredInfo[nump]
                                        # 重新定义高度区域 PredAreaPtsLoc
                                        CurBedAreaValidFlag = 1
                                        PredAreaPtsLoc[i_area, 1:] = PredInfo[nump]
                                        break;
                                # 更新当前 PredAreaPtsLoc 信息
                                PredAreaPtsLoc[i_area, 0] = CurBedAreaValidFlag
                                
                        # 根据多帧结果判断是否告警
                        AlarmValidFlag = True
                        PredBedPos = AlarmDete_Pts(PredBedPos, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = Bed_MinFrameNum, MaxFrameNum = Bed_MaxFrameNum)
        
                    else: # 不在检测时间段内
                        # 根据多帧结果判断是否告警
                        PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        AlarmValidFlag = False
                        PredBedPos = AlarmDete_Pts(PredBedPos, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = Bed_MinFrameNum, MaxFrameNum = Bed_MaxFrameNum)
            else:
                PredBedPos[:,[1,6]] = -1 # 设置告警状态为 -1
    
    #    [2,"未在制定区域监督"],
    CurAlarmTypeId = 2 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
            # 其他参数
            DeteAreaPtsInnerDis = 0.15 # 往内一定区域,init:0.01
            DeteMinHeight = 1.15 # 监管人员高度设置，init：1.25米
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = 0 # 连续 5 帧才启动有效计算
            # 内部监管使用方法
            INTERNALSUPERVISORMethodFlag = 2 # INTERNALSUPERVISORMethodFlag = 1，使用点云聚类方法
                                             # INTERNALSUPERVISORMethodFlag = 2，使用目标检测位置，非躺下人员个数
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            if InPeriodTimeFlag == 1:
                # 先判断目标区域的点云检测结果
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                AreaINTERNALSUPERVISOR = np.zeros([DeteArea.shape[0], 11]) # [x1,y1,x2,y2,x3,y3,x4,y4,z1,z2,validflag]
                for i_area in range(PredAreaPtsLoc.shape[0]):
                    ValidDeteArea = CalIntersectAreaFromTwoArea(DeteArea[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域，,【暂定单个传感器只有一个检测区域】
                    if len(ValidDeteArea) > 0: # 存在交叉区域
                        # area
                        CurMultiAreaEdge = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                        ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis) # 计算超高+边界区域的有效区域                        
                        ValidDeteArea = TransBboxToPoints(ValidDeteArea) # [xmin, xmax, ymin, ymax] -> [x1,y1,x2,y2,x3,y3,x4,y4]
                        # pts
                        AreaINTERNALSUPERVISOR[i_area,0] = ValidDeteArea[0] # x1
                        AreaINTERNALSUPERVISOR[i_area,1] = ValidDeteArea[1] # y1
                        AreaINTERNALSUPERVISOR[i_area,2] = ValidDeteArea[2] # x2
                        AreaINTERNALSUPERVISOR[i_area,3] = ValidDeteArea[3] # y2
                        AreaINTERNALSUPERVISOR[i_area,4] = ValidDeteArea[4] # x3
                        AreaINTERNALSUPERVISOR[i_area,5] = ValidDeteArea[5] # y3
                        AreaINTERNALSUPERVISOR[i_area,6] = ValidDeteArea[6] # x4
                        AreaINTERNALSUPERVISOR[i_area,7] = ValidDeteArea[7] # y4
                        AreaINTERNALSUPERVISOR[i_area,8] = DeteMinHeight # zmin
                        AreaINTERNALSUPERVISOR[i_area,9] = DeteMaxHeight # zmax
                        AreaINTERNALSUPERVISOR[i_area,10] = 1 # valid flag
                # CurINTERNALSUPERVISORHumanInfo
                if INTERNALSUPERVISORMethodFlag == 1: # 使用点云聚类方法
                    CurINTERNALSUPERVISORHumanInfo = human_cluster_dete(AreaINTERNALSUPERVISOR, NewPts, FileName) # 当前监管区域人员位置
                else: # 使用目标检测位置，非躺下人员个数
                    CurINTERNALSUPERVISORHumanInfo = []
                    CurINTERNALSUPERVISORHumanNum = 0
                    for nump in range(PredInfo.shape[0]):
                        for numa in range(AreaINTERNALSUPERVISOR.shape[0]):
                            # 多边形区域内
                            TempSensorDeteArea = AreaINTERNALSUPERVISOR[numa, :8]
                            CurrPoly = TempSensorDeteArea.reshape([int(len(TempSensorDeteArea)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                            pCurrPoly = path.Path(CurrPoly)
                            TempData = np.array([[0.0, 0.0]])
                            TempData[0,0] = PredInfo[nump][0]
                            TempData[0,1] = PredInfo[nump][1]
                            binAreaAll = pCurrPoly.contains_points(TempData) # limit xy, [2 x N]
                            # 是否在多边形内 + 非躺下
                            if binAreaAll[0] and ImgDeteLabel[nump] != CurModelLyingLabel:
                                CurINTERNALSUPERVISORHumanInfo.append([CurINTERNALSUPERVISORHumanNum, PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2]]) # [index, x, y, z]
                                CurINTERNALSUPERVISORHumanNum = CurINTERNALSUPERVISORHumanNum + 1
                    CurINTERNALSUPERVISORHumanInfo = np.array(CurINTERNALSUPERVISORHumanInfo)
                    
                # PredINTERNALSUPERVISORPos
                for i_INTERNALSUPERVISOR_obj in range(CurINTERNALSUPERVISORHumanInfo.shape[0]): 
                    # [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
                    PredINTERNALSUPERVISORPos.append([CurINTERNALSUPERVISORHumanInfo[i_INTERNALSUPERVISOR_obj,0], 1, -1, CurINTERNALSUPERVISORHumanInfo[i_INTERNALSUPERVISOR_obj,1], \
                                                          CurINTERNALSUPERVISORHumanInfo[i_INTERNALSUPERVISOR_obj,2], CurINTERNALSUPERVISORHumanInfo[i_INTERNALSUPERVISOR_obj,3], 1])
                    
                    
            else: # 不在检测时间段内
                PredINTERNALSUPERVISORPos = []
        else:
            PredINTERNALSUPERVISORPos = []
    
    #    [4,"厕所区域异常"],    
    CurAlarmTypeId = 4 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
            # 厕所区域持续时长
            if DeteContinuousTime < 0.0001: # 如果UI中未设置持续时长，则使用配置中设置
                DeteContinuousTime = DeteAlarmLevelInfo_toilet_continuous_time
            # 其他参数
            DeteAreaPtsInnerDis = 0.15# 往内一定区域
            DeteMinHeight = 0.2 # 龙岗厕所区域设置高度：0.9， 为了排除桶的干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = StayLocFrameThod # 连续 5 帧才启动有效计算
            TOILE_MinFrameNum = -1
            TOILE_MaxFrameNum = AlarmFrameNumThod+5
            
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                TOILE_MinFrameNum = OneFrmContinuousFrameMin
                TOILE_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            # 判断文件是否被修改了
            if ConfigInfoUpdate == True: # 如果文件有更新，则重置告警结果和告警起始时间
                NumTOILET = DeteArea.shape[0]
                PredTOILETPos = -1 * np.ones([NumTOILET, 7]) # 设置告警状态为 -1
                PredTOILET_STARTTIME = -1 * np.ones([NumTOILET, 2]) # 设置告警状态为 -1
                print('  PredTOILETPos re-setting.')
            else:
                if InPeriodTimeFlag == 1:
                    # 先判断目标区域的点云检测结果
                    PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                    for i_area in range(PredAreaPtsLoc.shape[0]):
                        CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                        CurDeteArea[0] = DeteArea[i_area][0] # xmin
                        CurDeteArea[1] = DeteArea[i_area][1] # xmax
                        CurDeteArea[2] = DeteArea[i_area][2] # ymin
                        CurDeteArea[3] = DeteArea[i_area][3] # ymax
                        CurDeteArea[4] = DeteMinHeight # zmin
                        CurDeteArea[5] = DeteMaxHeight # zmax
                        CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName)
                        if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                            PredAreaPtsLoc[i_area, :] = CurAlarmDeteHeightInfo
                    # 根据目标框结果检测人员信息
                    PredTOILETPos = AlarmDete_Loc(PredTOILETPos, PredInfo, ImgDeteLabel, ImgDeteScore, DeteArea, DeteMinHeight, DeteMaxHeight, PredAreaPtsLoc=PredAreaPtsLoc)
                    # 根据多帧结果判断是否告警
                    PredTOILETPos,  PredTOILET_STARTTIME = AlarmDete_SequenceFrame(PredTOILETPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredTOILET_STARTTIME, MinFrameNum = TOILE_MinFrameNum, MaxFrameNum = TOILE_MaxFrameNum)
#                    PredTOILETPos,  PredTOILET_STARTTIME = AlarmDete_SequenceFrame(PredTOILETPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredTOILET_STARTTIME, MinFrameNum = -1, MaxFrameNum = AlarmFrameNumThod+5, ContinuousFrameUpRate = AlarmContinuousFrameUpRate, ContinuousFrameDnRate = AlarmContinuousFrameDnRate)
                else: # 不在检测时间段内
                    # 根据目标框结果检测人员信息
                    PredTOILETPos = AlarmDete_Loc(PredTOILETPos, PredInfo, ImgDeteLabel, ImgDeteScore, DeteArea, DeteMinHeight, DeteMaxHeight)
                    # 设置未检测到人员
                    PredTOILETPos[:,1] = -1 # 设置告警类型为 -1，即未检测到人
                    # 根据多帧结果判断是否告警
                    PredTOILETPos,  PredTOILET_STARTTIME = AlarmDete_SequenceFrame(PredTOILETPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredTOILET_STARTTIME, MinFrameNum = TOILE_MinFrameNum, MaxFrameNum = TOILE_MaxFrameNum)
#                    PredTOILETPos,  PredTOILET_STARTTIME = AlarmDete_SequenceFrame(PredTOILETPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredTOILET_STARTTIME, MinFrameNum = -1, MaxFrameNum = AlarmFrameNumThod+5, ContinuousFrameUpRate = AlarmContinuousFrameUpRate, ContinuousFrameDnRate = AlarmContinuousFrameDnRate)
    
        else:
            PredTOILETPos[:,6] = -1 # 设置告警状态为 -1

    
    #    [8,"窗户区域异常"],
    CurAlarmTypeId = 8 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
            # 其他参数
            DeteMinHeight = 0.0 # 为了排除桶的干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = StayLocFrameThod # 连续 5 帧才启动有效计算
            WINDOW_MinFrameNum = -1
            WINDOW_MaxFrameNum = AlarmFrameNumThod+5
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                WINDOW_MinFrameNum = OneFrmContinuousFrameMin
                WINDOW_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            # 判断文件是否被修改了
            if ConfigInfoUpdate == True: # 如果文件有更新，则重置告警结果和告警起始时间
                NumWINDOW = DeteArea.shape[0]
                PredWINDOWPos = -1 * np.ones([NumWINDOW, 7]) # 设置告警状态为 -1
                PredWINDOW_STARTTIME = -1 * np.ones([NumWINDOW, 2]) # 设置告警状态为 -1
#                print('  PredWINDOWPos re-setting.')
            else:
                if InPeriodTimeFlag == 1:
                    # 根据目标框结果检测人员信息
                    PredWINDOWPos = AlarmDete_Loc(PredWINDOWPos, PredInfo, ImgDeteLabel, ImgDeteScore, DeteArea, DeteMinHeight, DeteMaxHeight)
                    # 根据多帧结果判断是否告警
                    PredWINDOWPos,  PredWINDOW_STARTTIME = AlarmDete_SequenceFrame(PredWINDOWPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredWINDOW_STARTTIME, MinFrameNum = WINDOW_MinFrameNum, MaxFrameNum = WINDOW_MaxFrameNum)
                else: # 不在检测时间段内
                    # 根据目标框结果检测人员信息
                    PredWINDOWPos = AlarmDete_Loc(PredWINDOWPos, PredInfo, ImgDeteLabel, ImgDeteScore, DeteArea, DeteMinHeight, DeteMaxHeight)
                    # 设置未检测到人员
                    PredWINDOWPos[:,1] = -1 # 设置告警类型为 -1，即未检测到人
                    # 根据多帧结果判断是否告警
                    PredWINDOWPos,  PredWINDOW_STARTTIME = AlarmDete_SequenceFrame(PredWINDOWPos, AlarmFrameNumThod, DeteContinuousTime, WorldTime, STARTTIME = PredWINDOW_STARTTIME, MinFrameNum = WINDOW_MinFrameNum, MaxFrameNum = WINDOW_MaxFrameNum)
        else:
            PredWINDOWPos[:,6] = -1 # 设置告警状态为 -1
        
    #    [16,"高度异常"],
    CurAlarmTypeId = 16 # 当前告警 id
    CurFrmAlarmFlag = False # 当前帧告警是否有效
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
#            print('CurAlarmTypeId = {}'.format(CurAlarmTypeId))
            # 配置文件设置参数
            DeteAreaSecond = None
            if AboveHeightSecondFlag == True:
                if np.array(CurALARM_INFO['DeteArea']).shape[0] > 2: # 超过两个超高目标区域
                    # 默认超高检测区域
                    AboveHeightAreaDefaultTypeIndex = [i_type for i_type in range(len(AboveHeightAreaTypeIdx)) if AboveHeightAreaTypeIdx[i_type]==AboveHeightAreaDefaultType]
                    # 特殊超高检测区域
                    AboveHeightAreaSecondTypeIndex = [i_type for i_type in range(len(AboveHeightAreaTypeIdx)) if AboveHeightAreaTypeIdx[i_type]==AboveHeightAreaSecondType]
                    # DeteArea，DeteAreaSecond
                    DeteArea_src = np.array(CurALARM_INFO['DeteArea'])
                    DeteArea = DeteArea_src[AboveHeightAreaDefaultTypeIndex,:][:,[0,2,1,3]] # 前两个区域，[xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                    DeteAreaSecond = DeteArea_src[AboveHeightAreaSecondTypeIndex,:][:,[0,2,1,3]] # 其他特殊目标区域
                else:
                    DeteArea = np.array(CurALARM_INFO['DeteArea'])
                    DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            else:
                DeteArea = np.array(CurALARM_INFO['DeteArea'])
                DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                    
#            DeteArea = np.array(CurALARM_INFO['DeteArea'])
#            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight'] + AboveHeightUpDownThod[1] # init:0.25/1.1 , 【20200707：超高阈值上限设置更高】
            DeteMinHeight = CurALARM_INFO['DeteMaxHeight'] - AboveHeightUpDownThod[0] # 超高设置范围
            
            # 其他参数
#            DeteAreaPtsInnerDis = 0.01 # 往内一定区域,init:0.1,【20200620】
            DeteAreaPtsInnerDis = AboveHeightDeteAreaPtsInnerDis # 往内一定区域,init:0.1,【20200620】
            
#            DeteMinHeight = CurALARM_INFO['DeteMaxHeight'] - AboveHeightUpDownThod[0] # 超高设置范围
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
#            AlarmFrameNumThod = 0 # 连续 5 帧才启动有效计算
            AlarmFrameNumThod = DeteAlarmLevelInfo_above_height_continuous_frame_num # 超高连续多帧告警
            
            # 超高连续帧上下限制阈值
            AboveHeight_MinFrameNum = -1
            AboveHeight_MaxFrameNum = DeteAlarmLevelInfo_above_height_continuous_frame_num + 2
            
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                AboveHeight_MinFrameNum = OneFrmContinuousFrameMin
                AboveHeight_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 计算前景数据去除噪点，暂时只对‘超高’做此处理
            if AboveHeightFGMethodFlag == 1:
                NewPtsBGValid = copy.copy(Pts)
            else:
                if PtsSubBG is None:
                    NewPtsBG = CalFGSubNoise(Pts, DepthWidth = ImgWidth, DepthHeight = ImgHeight)
                    NewPtsBGValid = Rot3D(H, NewPtsBG) #  H * Pts, [3 x 217088]
                else:
                    NewPtsBGValid = Rot3D(H, PtsSubBG) #  H * Pts, [3 x 217088]
            
            # 初始化 PredAboveHeight
            NumAboveHeightArea = DeteArea.shape[0]
#            PredAboveHeight = -1 * np.ones([NumAboveHeightArea, 7])
            AllDeteValidArea = np.zeros([6,NumAboveHeightArea]) # 多个有效的目标区域，[xmin, xmax, ymin, ymax, zmin, zmax]
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
#            print('AboveHeight InPeriodTimeFlag = {}'.format(InPeriodTimeFlag))
            if InPeriodTimeFlag == 1:
                # 先判断目标区域的点云检测结果
                PredAreaPtsLoc = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                for i_area in range(DeteArea.shape[0]):
#                    print('DeteArea[i_area] = ', DeteArea[i_area])
#                    print('CurSensorDeteArea = ', CurSensorDeteArea)
                    # 计算各传感器检测的区域
                    ValidDeteArea = CalIntersectAreaFromTwoArea(DeteArea[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
#                    print('ValidDeteArea = ', ValidDeteArea)
                    # CurDeteArea
                    CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                    if len(ValidDeteArea) > 0: # 存在交叉区域
                        CurMultiAreaEdge = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                        DeteAreaPtsInnerDis_Add = DeteAreaPtsInnerDis + AboveHeightSensorOverlapDist # 增加传感器交接处范围，解决超高告警漏报问题
                        ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis_Add) # 计算超高+边界区域的有效区域
                        CurDeteArea[0] = ValidDeteArea[0] # xmin
                        CurDeteArea[1] = ValidDeteArea[1] # xmax
                        CurDeteArea[2] = ValidDeteArea[2] # ymin
                        CurDeteArea[3] = ValidDeteArea[3] # ymax
                        CurDeteArea[4] = DeteMinHeight # zmin
                        CurDeteArea[5] = DeteMaxHeight # zmax
                    AllDeteValidArea[:,i_area] = CurDeteArea[:,0]
#                    print('CurDeteArea = {}'.format(CurDeteArea))
#                    print('NewPts size = {}'.format(NewPts.shape))
#                    CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName) #
#                    CurAlarmDeteHeightInfo = alarm_detet_above_height_object(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName) #

##                    print('AboveHeightCluPtsNumThod = ', AboveHeightCluPtsNumThod)
#                    CurAlarmDeteHeightInfo = alarm_detet_above_height_object(CurDeteArea, DeteAreaPtsInnerDis, NewPtsBGValid, FileName, InputMinCluPtNumThod = AboveHeightCluPtsNumThod) # 20200716

#                    print('AboveHeightCluPtsNumThod = ', AboveHeightCluPtsNumThod)
                    CurAlarmDeteHeightInfo = alarm_detet_above_height_object(CurDeteArea, DeteAreaPtsInnerDis, NewPtsBGValid, FileName, InputMinCluPtNumThod = AboveHeightCluPtsNumThod, PtsClusterDistrSelect = DeteAlarmLevelInfo_above_height_cluster_distribute_select) # 20200716

                    
                    if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                        PredAreaPtsLoc = np.concatenate((PredAreaPtsLoc, CurAlarmDeteHeightInfo)) # 多区域多目标累加
#                print('src PredAreaPtsLoc = {}'.format(PredAreaPtsLoc))

                # 是否存在多类型超高目标区域
                if DeteAreaSecond is not None:
                    # 判断特殊区域内是否存在目标
                    if AboveHeightSecondMethodFlag == 1: # 方法1：使用中间设置的一个高度计算是否超高，判断是否有两个高度超高
                        PredAreaPtsLocSecond = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        AllDeteValidAreaSecond = np.zeros([6,DeteAreaSecond.shape[0]])
                        # 遍历特殊超高目标区域
                        for i_area in range(DeteAreaSecond.shape[0]):
                            # 计算各传感器检测的区域
                            ValidDeteAreaSecond = CalIntersectAreaFromTwoArea(DeteAreaSecond[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                            # CurDeteArea
                            CurDeteAreaSecond = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                            if len(ValidDeteAreaSecond) > 0: # 存在交叉区域
                                CurMultiAreaEdgeSecond = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                                DeteAreaPtsInnerDis_Add_Second = DeteAreaPtsInnerDis + AboveHeightSensorOverlapDist # 增加传感器交接处范围，解决超高告警漏报问题
                                ValidDeteAreaSecond = CalAboveHeightValidArea(ValidDeteAreaSecond, CurMultiAreaEdgeSecond, DeteAreaInnerDis = DeteAreaPtsInnerDis_Add_Second) # 计算超高+边界区域的有效区域
                                CurDeteAreaSecond[0] = ValidDeteAreaSecond[0] # xmin
                                CurDeteAreaSecond[1] = ValidDeteAreaSecond[1] # xmax
                                CurDeteAreaSecond[2] = ValidDeteAreaSecond[2] # ymin
                                CurDeteAreaSecond[3] = ValidDeteAreaSecond[3] # ymax
                                CurDeteAreaSecond[4] = CurALARM_INFO['DeteMaxHeight'] - AboveHeightUpDownThod2[0] # zmin
                                CurDeteAreaSecond[5] = CurALARM_INFO['DeteMaxHeight'] + AboveHeightUpDownThod2[1] # zmax
                            AllDeteValidAreaSecond[:,i_area] = CurDeteAreaSecond[:,0]
                            CurAlarmDeteHeightInfoSecond = alarm_detet_above_height_object(CurDeteAreaSecond, DeteAreaPtsInnerDis, NewPtsBGValid, FileName, InputMinCluPtNumThod = AboveHeightCluPtsNumThod, PtsClusterDistrSelect = DeteAlarmLevelInfo_above_height_cluster_distribute_select) # 20200716
                            if CurAlarmDeteHeightInfoSecond[0,0] == 1: # 是否存在目标
                                PredAreaPtsLocSecond = np.concatenate((PredAreaPtsLocSecond, CurAlarmDeteHeightInfoSecond)) # 多区域多目标累加
                        
                        # 合并 PredAreaPtsLoc 和 PredAreaPtsLocSecond
                        PredAreaPtsLocCombine = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z] 
                        for i_obj in range(PredAreaPtsLoc.shape[0]):
                            CurPredAreaPtsLoc = PredAreaPtsLoc[i_obj,:] # [目标有效标记, 位置x, 位置y, 位置z]                                              
                            CurPredAreaPtsLocValidFlag = False
                            for j_area in range(DeteAreaSecond.shape[0]):
                                CurOneDeteAreaSecond = DeteAreaSecond[j_area,:] # [xmin, xmax, ymin, ymax]
                                if CurPredAreaPtsLoc[1] > CurOneDeteAreaSecond[0] and CurPredAreaPtsLoc[1] < CurOneDeteAreaSecond[1] and CurPredAreaPtsLoc[2] > CurOneDeteAreaSecond[2] and CurPredAreaPtsLoc[2] < CurOneDeteAreaSecond[3]:
                                    CurPredAreaPtsLocValidFlag = True
                            if CurPredAreaPtsLocValidFlag == True: # 如果目标点再特殊区域内
                                if PredAreaPtsLocSecond.shape[0] > 0: # 特殊区域存在目标点，[暂时未区分目标具体对应关系]
                                    CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                    CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                    PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save)) # 多区域多目标累加
                            else: # 如果目标点不在特殊区域内
                                CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save))
                        PredAreaPtsLoc = PredAreaPtsLocCombine
       
                    elif AboveHeightSecondMethodFlag == 2: # 方法2：使用目标下方一定数据，判断连通性
                        AboveHeightSecondMethod2_MinHeight = 1.3 # 人员最低高度
                        PredAreaPtsLocSecond = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        AllDeteValidAreaSecond = np.zeros([6,DeteAreaSecond.shape[0]])
                        # 遍历特殊超高目标区域
                        for i_area in range(DeteAreaSecond.shape[0]):
                            # 计算各传感器检测的区域
                            ValidDeteAreaSecond = CalIntersectAreaFromTwoArea(DeteAreaSecond[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                            # CurDeteArea
                            CurDeteAreaSecond = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                            if len(ValidDeteAreaSecond) > 0: # 存在交叉区域
                                CurMultiAreaEdgeSecond = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                                DeteAreaPtsInnerDis_Add_Second = DeteAreaPtsInnerDis + AboveHeightSensorOverlapDist # 增加传感器交接处范围，解决超高告警漏报问题
                                ValidDeteAreaSecond = CalAboveHeightValidArea(ValidDeteAreaSecond, CurMultiAreaEdgeSecond, DeteAreaInnerDis = DeteAreaPtsInnerDis_Add_Second) # 计算超高+边界区域的有效区域
                                CurDeteAreaSecond[0] = ValidDeteAreaSecond[0] # xmin
                                CurDeteAreaSecond[1] = ValidDeteAreaSecond[1] # xmax
                                CurDeteAreaSecond[2] = ValidDeteAreaSecond[2] # ymin
                                CurDeteAreaSecond[3] = ValidDeteAreaSecond[3] # ymax
                                CurDeteAreaSecond[4] = CurALARM_INFO['DeteMaxHeight'] - AboveHeightUpDownThod2[0] # zmin
                                CurDeteAreaSecond[5] = CurALARM_INFO['DeteMaxHeight'] + AboveHeightUpDownThod2[1] # zmax
                            AllDeteValidAreaSecond[:,i_area] = CurDeteAreaSecond[:,0]
#                            CurAlarmDeteHeightInfoSecond = alarm_detet_above_height_object(CurDeteAreaSecond, DeteAreaPtsInnerDis, NewPtsBGValid, FileName, InputMinCluPtNumThod = AboveHeightCluPtsNumThod, PtsClusterDistrSelect = DeteAlarmLevelInfo_above_height_cluster_distribute_select) # 20200716
#                            if CurAlarmDeteHeightInfoSecond[0,0] == 1: # 是否存在目标
#                                PredAreaPtsLocSecond = np.concatenate((PredAreaPtsLocSecond, CurAlarmDeteHeightInfoSecond)) # 多区域多目标累加

                        # 合并 PredAreaPtsLoc 和 PredAreaPtsLocSecond
                        PredAreaPtsLocCombine = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z] 
                        for i_obj in range(PredAreaPtsLoc.shape[0]):
                            CurPredAreaPtsLoc = PredAreaPtsLoc[i_obj,:] # [目标有效标记, 位置x, 位置y, 位置z]                                              
                            CurPredAreaPtsLocValidFlag = False
                            CurPredAreaPtsLocConnectFlag = 0
                            for j_area in range(DeteAreaSecond.shape[0]):
                                CurOneDeteAreaSecond = AllDeteValidAreaSecond[:,j_area] # 调整后的目标检测区域，[xmin, xmax, ymin, ymax, zmin, zmax]
                                if CurPredAreaPtsLoc[1] > CurOneDeteAreaSecond[0] and CurPredAreaPtsLoc[1] < CurOneDeteAreaSecond[1] and CurPredAreaPtsLoc[2] > CurOneDeteAreaSecond[2] and CurPredAreaPtsLoc[2] < CurOneDeteAreaSecond[3]:
                                    CurPredAreaPtsLocValidFlag = True
                                    # 计算是否存在超高
                                    ConnectSampCenterDist = 0.4 # 目标中心点边界半径
                                    ConnectSampCenter = CurPredAreaPtsLoc[1:] # [x,y,z]
                                    ConnectSampWinSizeZ = ConnectSampCenter[2] - AboveHeightSecondMethod2_MinHeight # z方向数据
                                    if ConnectSampWinSizeZ > 0: # 超高点需高于人员目标
                                        # 使用去噪后点云数据 NewPtsBGValid
#                                        ConnectSampPtsIdx = (NewPtsBGValid[0,:]>ConnectSampCenter[0]-ConnectSampCenterDist) & (NewPtsBGValid[0,:]<ConnectSampCenter[0]+ConnectSampCenterDist) & \
#                                                             (NewPtsBGValid[1,:]>ConnectSampCenter[1]-ConnectSampCenterDist) & (NewPtsBGValid[1,:]<ConnectSampCenter[1]+ConnectSampCenterDist) & \
#                                                              (NewPtsBGValid[2,:]>AboveHeightSecondMethod2_MinHeight) & (NewPtsBGValid[0,:]<ConnectSampCenter[2])
#                                        ConnectSampPts = NewPtsBGValid[:,ConnectSampPtsIdx] # [3 x N]
                                        # 使用原始配准后点云数据 NewPts
                                        ConnectSampPtsIdx = (NewPts[0,:]>ConnectSampCenter[0]-ConnectSampCenterDist) & (NewPts[0,:]<ConnectSampCenter[0]+ConnectSampCenterDist) & \
                                                             (NewPts[1,:]>ConnectSampCenter[1]-ConnectSampCenterDist) & (NewPts[1,:]<ConnectSampCenter[1]+ConnectSampCenterDist) & \
                                                              (NewPts[2,:]>AboveHeightSecondMethod2_MinHeight) & (NewPts[2,:]<ConnectSampCenter[2])
                                        ConnectSampPts = NewPts[:,ConnectSampPtsIdx] # [3 x N]

                                        ConnectSampWinSizeX = 2 * ConnectSampCenterDist
                                        # calculate connected
                                        CurPredAreaPtsLocConnectFlag = connected(ConnectSampPts, ConnectSampCenter, ConnectSampWinSizeX, ConnectSampWinSizeZ, GridSize2d=0.05, connect_rate=0.9)
                                        
                            if CurPredAreaPtsLocValidFlag == True: # 如果目标点再特殊区域内
                                if CurPredAreaPtsLocConnectFlag > 0: # 特殊区域存在目标点，[暂时未区分目标具体对应关系]
                                    CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                    CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                    PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save)) # 多区域多目标累加
                            else: # 如果目标点不在特殊区域内
                                CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save))
                        PredAreaPtsLoc = PredAreaPtsLocCombine
                        
                    elif AboveHeightSecondMethodFlag == 3: # 方法3：屏蔽特殊区域超高目标
                        AboveHeightSecondMethod2_MinHeight = 1.3 # 人员最低高度
                        PredAreaPtsLocSecond = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        AllDeteValidAreaSecond = np.zeros([6,DeteAreaSecond.shape[0]])
                        # 遍历特殊超高目标区域
                        for i_area in range(DeteAreaSecond.shape[0]):
                            # 计算各传感器检测的区域
                            ValidDeteAreaSecond = CalIntersectAreaFromTwoArea(DeteAreaSecond[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                            # CurDeteArea
                            CurDeteAreaSecond = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                            if len(ValidDeteAreaSecond) > 0: # 存在交叉区域
                                CurMultiAreaEdgeSecond = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
                                DeteAreaPtsInnerDis_Add_Second = DeteAreaPtsInnerDis + AboveHeightSensorOverlapDist # 增加传感器交接处范围，解决超高告警漏报问题
                                ValidDeteAreaSecond = CalAboveHeightValidArea(ValidDeteAreaSecond, CurMultiAreaEdgeSecond, DeteAreaInnerDis = DeteAreaPtsInnerDis_Add_Second) # 计算超高+边界区域的有效区域
                                CurDeteAreaSecond[0] = ValidDeteAreaSecond[0] # xmin
                                CurDeteAreaSecond[1] = ValidDeteAreaSecond[1] # xmax
                                CurDeteAreaSecond[2] = ValidDeteAreaSecond[2] # ymin
                                CurDeteAreaSecond[3] = ValidDeteAreaSecond[3] # ymax
                                CurDeteAreaSecond[4] = CurALARM_INFO['DeteMaxHeight'] - AboveHeightUpDownThod2[0] # zmin
                                CurDeteAreaSecond[5] = CurALARM_INFO['DeteMaxHeight'] + AboveHeightUpDownThod2[1] # zmax
                            AllDeteValidAreaSecond[:,i_area] = CurDeteAreaSecond[:,0]
#                            CurAlarmDeteHeightInfoSecond = alarm_detet_above_height_object(CurDeteAreaSecond, DeteAreaPtsInnerDis, NewPtsBGValid, FileName, InputMinCluPtNumThod = AboveHeightCluPtsNumThod, PtsClusterDistrSelect = DeteAlarmLevelInfo_above_height_cluster_distribute_select) # 20200716
#                            if CurAlarmDeteHeightInfoSecond[0,0] == 1: # 是否存在目标
#                                PredAreaPtsLocSecond = np.concatenate((PredAreaPtsLocSecond, CurAlarmDeteHeightInfoSecond)) # 多区域多目标累加

                        # 合并 PredAreaPtsLoc 和 PredAreaPtsLocSecond
                        PredAreaPtsLocCombine = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z] 
                        for i_obj in range(PredAreaPtsLoc.shape[0]):
                            CurPredAreaPtsLoc = PredAreaPtsLoc[i_obj,:] # [目标有效标记, 位置x, 位置y, 位置z]                                              
                            CurPredAreaPtsLocValidFlag = False
                            CurPredAreaPtsLocConnectFlag = 0
                            for j_area in range(DeteAreaSecond.shape[0]):
                                CurOneDeteAreaSecond = AllDeteValidAreaSecond[:,j_area] # 调整后的目标检测区域，[xmin, xmax, ymin, ymax, zmin, zmax]
                                if CurPredAreaPtsLoc[1] > CurOneDeteAreaSecond[0] and CurPredAreaPtsLoc[1] < CurOneDeteAreaSecond[1] and CurPredAreaPtsLoc[2] > CurOneDeteAreaSecond[2] and CurPredAreaPtsLoc[2] < CurOneDeteAreaSecond[3]:
                                    CurPredAreaPtsLocValidFlag = True
                                    # 计算是否存在超高
                                    CurPredAreaPtsLocConnectFlag = 0
                                        
                            if CurPredAreaPtsLocValidFlag == True: # 如果目标点再特殊区域内
                                if CurPredAreaPtsLocConnectFlag > 0: # 特殊区域存在目标点，[暂时未区分目标具体对应关系]
                                    CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                    CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                    PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save)) # 多区域多目标累加
                            else: # 如果目标点不在特殊区域内
                                CurPredAreaPtsLoc_Save = np.zeros([1, 4])
                                CurPredAreaPtsLoc_Save[0,:] = CurPredAreaPtsLoc
                                PredAreaPtsLocCombine = np.concatenate((PredAreaPtsLocCombine, CurPredAreaPtsLoc_Save))
                        PredAreaPtsLoc = PredAreaPtsLocCombine
                    
                # 判断当前帧超高是否有效
                if PredAreaPtsLoc.shape[0] > 0:
                    CurFrmAlarmFlag = True
                
                # 根据多帧结果判断是否告警
                AlarmValidFlag = True
                if AlarmFrameNumThod == 0: # 暂定使用 1 帧数据判断超高，【如果多帧数据需要做目标匹配过程】
#                    PredAboveHeight_AllArea = -1 * np.ones([DeteArea.shape[0], 7])
                    PredAboveHeight_AllArea = np.empty([0,7])
                    for i_area in range(DeteArea.shape[0]):
                        i_AllDeteValidArea = i_area
                        # 计算目标人数
                        PredValidObjInfo = np.empty([0, 3]) # 有效的目标个数
                        for nump in range(PredInfo.shape[0]):
                            # 判断目标位置是否在当前传感器检测范围内
                            CurPredObjValidFlag = -1 # 是否在传感器检测区域内
                            for i_AllDeteValidArea in range(AllDeteValidArea.shape[1]):
                                if PredInfo[nump][0] > AllDeteValidArea[0,i_AllDeteValidArea] and PredInfo[nump][0] < AllDeteValidArea[1,i_AllDeteValidArea] and PredInfo[nump][1] > AllDeteValidArea[2,i_AllDeteValidArea] and PredInfo[nump][1] < AllDeteValidArea[3,i_AllDeteValidArea]:
                                    CurPredObjValidFlag = 1
                            # 选择最近的超高点
                            if CurPredObjValidFlag == 1:
                                CurPredInfo = np.zeros([1,3])
                                CurPredInfo[0,:] = PredInfo[nump]
                                PredValidObjInfo = np.concatenate([PredValidObjInfo, CurPredInfo])
                        # 计算聚类目标是否在当前范围内
                        CluValidObjInfo = np.empty([0, 4]) # 有效的目标个数
                        for num_clu in range(PredAreaPtsLoc.shape[0]):
                            i_AllDeteValidArea = i_area
                            if PredAreaPtsLoc[num_clu][1] > AllDeteValidArea[0,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][1] < AllDeteValidArea[1,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][2] > AllDeteValidArea[2,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][2] < AllDeteValidArea[3,i_AllDeteValidArea]:
                                CurCluValidObjInfo = np.zeros([1,4])
                                CurCluValidObjInfo[0,1:4] = PredAreaPtsLoc[num_clu][1:4]
                                CurCluValidObjInfo[0,0] = 1
                                CluValidObjInfo = np.concatenate([CluValidObjInfo, CurCluValidObjInfo])
                        
                        # 根据人员和超高匹配关系，挑选目标, [人员有一定高度 + 人员距离超高目标一定范围内，才认为超高目标有效；]
                        if DeteAlarmLevelInfo_above_height__using_human_dete == True:
                            if CluValidObjInfo.shape[0] > 0: # 有超高目标
                                if PredValidObjInfo.shape[0] > 0: # 有人员检测目标
                                    CluValidObjInfo_Valid = [False] * CluValidObjInfo.shape[0] # 有效目标个数
                                    for i_CluValidObj in range(CluValidObjInfo.shape[0]): # 遍历超高目标
                                        CurCluValidObjInfo = CluValidObjInfo[i_CluValidObj,:] # 当前超高目标
                                        for i_PredValidObj in range(PredValidObjInfo.shape[0]): # 遍历人员检测目标
                                            CurPredValidObjInfo = PredValidObjInfo[i_PredValidObj, :] # 当前人员检测目标
                                            CurValidObjAndAlarmDis = np.sqrt((CurPredValidObjInfo[0]-CurCluValidObjInfo[1])**2 + (CurPredValidObjInfo[1]-CurCluValidObjInfo[2])**2) # 聚类目标与人员的距离
                                            if (CurValidObjAndAlarmDis < AboveHeightCluObjToHumanDistThod) and (PredHumanObjHeight[i_PredValidObj] > AboveHeightHumanObjHeightThod): # 人员距离超高点小于阈值 + 人员高度大于阈值
                                                CluValidObjInfo_Valid[i_CluValidObj] = True
                                                break
                                    # 新的有效聚类目标信息
                                    CluValidObjInfo = CluValidObjInfo[CluValidObjInfo_Valid,:]
                                else: # 无人员检测目标
                                    CluValidObjInfo = np.empty([0, 4])
                            
                        # 排除单人举双手的情况
                        PredAreaPtsLocSelect = np.empty([0,4]) # 筛选有效的告警点, [label, x, y, z]
                        if (CluValidObjInfo.shape[0] > 1) and (PredValidObjInfo.shape[0] > 0) and (CluValidObjInfo.shape[0] > PredValidObjInfo.shape[0]):# 单个传感器检测到多个超高                  
                            PredAreaPtsLocUsedFlag = -1 * np.ones([CluValidObjInfo.shape[0]]) # 初始化告警目标点是否已经被使用
                            for i_PredValidObjInfo in range(PredValidObjInfo.shape[0]):
                                CurValidObjAndAlarmMinDis = 1000 # 初始化目标的最近距离和序号
                                CurValidObjAndAlarmMinDisIdx = -1
                                for i_PredAreaPtsLoc in range(CluValidObjInfo.shape[0]):
                                    if PredAreaPtsLocUsedFlag[i_PredAreaPtsLoc] == -1: # 未被使用的告警点
                                        CurValidObjAndAlarmDis = np.sqrt((PredValidObjInfo[i_PredValidObjInfo,0]-CluValidObjInfo[i_PredAreaPtsLoc,1])**2 + (PredValidObjInfo[i_PredValidObjInfo,1]-CluValidObjInfo[i_PredAreaPtsLoc,2])**2) 
                                        if CurValidObjAndAlarmMinDis > CurValidObjAndAlarmDis:
                                            CurValidObjAndAlarmMinDis = CurValidObjAndAlarmDis
                                            CurValidObjAndAlarmMinDisIdx = i_PredAreaPtsLoc
                                if not CurValidObjAndAlarmMinDisIdx == -1: # 如果找到了一个最近告警点
                                    PredAreaPtsLocUsedFlag[CurValidObjAndAlarmMinDisIdx] = 1 # 已被使用的告警点
                                    CurPredAreaPtsLoc = np.zeros([1,4])
                                    CurPredAreaPtsLoc[0,:] = CluValidObjInfo[CurValidObjAndAlarmMinDisIdx, :]
                                    PredAreaPtsLocSelect = np.concatenate([PredAreaPtsLocSelect, CurPredAreaPtsLoc]) # 添加有效的目标位置
                        else:
                            PredAreaPtsLocSelect = CluValidObjInfo
                                    
                        # 显示告警的位置信息
                        CurArea_PredAreaPtsLoc = PredAreaPtsLocSelect
                        PredAboveHeight_SingleFrm = -1 * np.ones([CurArea_PredAreaPtsLoc.shape[0], 7])
                        for i_above_height in range(CurArea_PredAreaPtsLoc.shape[0]):
                            PredAboveHeight_SingleFrm[i_above_height, 0] = i_above_height # obj index
                            PredAboveHeight_SingleFrm[i_above_height, 1] = 0
                            PredAboveHeight_SingleFrm[i_above_height, 2] = 0 # num
                            PredAboveHeight_SingleFrm[i_above_height, 3] = CurArea_PredAreaPtsLoc[i_above_height, 1] # [x,y,z]
                            PredAboveHeight_SingleFrm[i_above_height, 4] = CurArea_PredAreaPtsLoc[i_above_height, 2]
                            PredAboveHeight_SingleFrm[i_above_height, 5] = CurArea_PredAreaPtsLoc[i_above_height, 3]
                            PredAboveHeight_SingleFrm[i_above_height, 6] = 1 # valid flag
                        
                        # 合并多个区域
#                        if PredAboveHeight_SingleFrm.shape[0]>0:
#                            PredAboveHeight_AllArea[i_area, :] = PredAboveHeight_SingleFrm[0,:] # 暂时一个区域只显示一个告警，20200809
                            
                        PredAboveHeight_AllArea = np.concatenate((PredAboveHeight_AllArea, PredAboveHeight_SingleFrm)) # 
                            
                    PredAboveHeight = PredAboveHeight_AllArea
                else: # 使用连续帧告警
                    #    只使用一个目标进行连续帧累加
                    PredAboveHeight_Fisrt_S_All = np.empty([0, 7])
                    for i_area in range(DeteArea.shape[0]):
                        # 计算目标人数
                        PredValidObjInfo = np.empty([0, 3]) # 有效的目标个数
                        for nump in range(PredInfo.shape[0]):
                            # 判断目标位置是否在当前传感器检测范围内
                            CurPredObjValidFlag = -1 # 是否在传感器检测区域内
                            i_AllDeteValidArea = i_area
                            if PredInfo[nump][0] > AllDeteValidArea[0,i_AllDeteValidArea] and PredInfo[nump][0] < AllDeteValidArea[1,i_AllDeteValidArea] and PredInfo[nump][1] > AllDeteValidArea[2,i_AllDeteValidArea] and PredInfo[nump][1] < AllDeteValidArea[3,i_AllDeteValidArea]:
                                CurPredObjValidFlag = 1
                            # 选择最近的超高点
                            if CurPredObjValidFlag == 1:
                                CurPredInfo = np.zeros([1,3])
                                CurPredInfo[0,:] = PredInfo[nump]
                                PredValidObjInfo = np.concatenate([PredValidObjInfo, CurPredInfo])
                        # 计算聚类目标是否在当前范围内
                        CluValidObjInfo = np.empty([0, 4]) # 有效的目标个数
                        for num_clu in range(PredAreaPtsLoc.shape[0]):
                            i_AllDeteValidArea = i_area
                            if PredAreaPtsLoc[num_clu][1] > AllDeteValidArea[0,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][1] < AllDeteValidArea[1,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][2] > AllDeteValidArea[2,i_AllDeteValidArea] and PredAreaPtsLoc[num_clu][2] < AllDeteValidArea[3,i_AllDeteValidArea]:
                                CurCluValidObjInfo = np.zeros([1,4])
                                CurCluValidObjInfo[0,1:4] = PredAreaPtsLoc[num_clu][1:4]
                                CurCluValidObjInfo[0,0] = 1
                                CluValidObjInfo = np.concatenate([CluValidObjInfo, CurCluValidObjInfo])
                        
                        # 根据人员和超高匹配关系，挑选目标, [人员有一定高度 + 人员距离超高目标一定范围内，才认为超高目标有效；]
                        if DeteAlarmLevelInfo_above_height__using_human_dete == True:
                            if CluValidObjInfo.shape[0] > 0: # 有超高目标
                                if PredValidObjInfo.shape[0] > 0: # 有人员检测目标
                                    CluValidObjInfo_Valid = [False] * CluValidObjInfo.shape[0] # 有效目标个数
                                    for i_CluValidObj in range(CluValidObjInfo.shape[0]): # 遍历超高目标
                                        CurCluValidObjInfo = CluValidObjInfo[i_CluValidObj,:] # 当前超高目标
                                        for i_PredValidObj in range(PredValidObjInfo.shape[0]): # 遍历人员检测目标
                                            CurPredValidObjInfo = PredValidObjInfo[i_PredValidObj, :] # 当前人员检测目标
                                            CurValidObjAndAlarmDis = np.sqrt((CurPredValidObjInfo[0]-CurCluValidObjInfo[1])**2 + (CurPredValidObjInfo[1]-CurCluValidObjInfo[2])**2) # 聚类目标与人员的距离
                                            if (CurValidObjAndAlarmDis < AboveHeightCluObjToHumanDistThod) and (PredHumanObjHeight[i_PredValidObj] > AboveHeightHumanObjHeightThod): # 人员距离超高点小于阈值 + 人员高度大于阈值
                                                CluValidObjInfo_Valid[i_CluValidObj] = True
                                                break
                                    # 新的有效聚类目标信息
                                    CluValidObjInfo = CluValidObjInfo[CluValidObjInfo_Valid,:]
                                else: # 无人员检测目标
                                    CluValidObjInfo = np.empty([0, 4])
                                
                        # 排除单人举双手的情况
                        PredAreaPtsLocSelect = np.empty([0,4]) # 筛选有效的告警点, [label, x, y, z]
                        if (CluValidObjInfo.shape[0] > 1) and (PredValidObjInfo.shape[0] > 0) and (CluValidObjInfo.shape[0] > PredValidObjInfo.shape[0]):# 单个传感器检测到多个超高                  
                            PredAreaPtsLocUsedFlag = -1 * np.ones([CluValidObjInfo.shape[0]]) # 初始化告警目标点是否已经被使用
                            for i_PredValidObjInfo in range(PredValidObjInfo.shape[0]):
                                CurValidObjAndAlarmMinDis = 1000 # 初始化目标的最近距离和序号
                                CurValidObjAndAlarmMinDisIdx = -1
                                for i_PredAreaPtsLoc in range(CluValidObjInfo.shape[0]):
                                    if PredAreaPtsLocUsedFlag[i_PredAreaPtsLoc] == -1: # 未被使用的告警点
                                        CurValidObjAndAlarmDis = np.sqrt((PredValidObjInfo[i_PredValidObjInfo,0]-CluValidObjInfo[i_PredAreaPtsLoc,1])**2 + (PredValidObjInfo[i_PredValidObjInfo,1]-CluValidObjInfo[i_PredAreaPtsLoc,2])**2)
                                        if CurValidObjAndAlarmMinDis > CurValidObjAndAlarmDis:
                                            CurValidObjAndAlarmMinDis = CurValidObjAndAlarmDis
                                            CurValidObjAndAlarmMinDisIdx = i_PredAreaPtsLoc
                                if not CurValidObjAndAlarmMinDisIdx == -1: # 如果找到了一个最近告警点
                                    PredAreaPtsLocUsedFlag[CurValidObjAndAlarmMinDisIdx] = 1 # 已被使用的告警点
                                    CurPredAreaPtsLoc = np.zeros([1,4])
                                    CurPredAreaPtsLoc[0,:] = CluValidObjInfo[CurValidObjAndAlarmMinDisIdx, :]
                                    PredAreaPtsLocSelect = np.concatenate([PredAreaPtsLocSelect, CurPredAreaPtsLoc]) # 添加有效的目标位置
                        else:
                            PredAreaPtsLocSelect = CluValidObjInfo
                        # 使用连续帧数
                        CurArea_PredAreaPtsLoc = PredAreaPtsLocSelect
                        # DeteArea_One
                        DeteArea_One = np.zeros([1, 4])
                        DeteArea_One[0,:] = DeteArea[i_area,:]
                        if CurArea_PredAreaPtsLoc.shape[0]>0:
                            PredAreaPtsLoc_Fisrt = np.zeros([1, 4])
                            PredAreaPtsLoc_Fisrt[0,:] = CurArea_PredAreaPtsLoc[0,:]
                            if PredAboveHeight.shape[0] > 0:
                                PredAboveHeight_Fisrt = -1 * np.ones([1, 7])
                                PredAboveHeight_Fisrt[0,:]= PredAboveHeight[i_area,:]
                                PredAboveHeight_Fisrt_S = AlarmDete_Pts(PredAboveHeight_Fisrt, DeteArea_One, PredAreaPtsLoc_Fisrt, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = AboveHeight_MinFrameNum, MaxFrameNum = AboveHeight_MaxFrameNum)
                            else:
                                PredAboveHeight_Fisrt = -1 * np.ones([1, 7])
                                PredAboveHeight_Fisrt_S = AlarmDete_Pts(PredAboveHeight_Fisrt, DeteArea_One, PredAreaPtsLoc_Fisrt, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = AboveHeight_MinFrameNum, MaxFrameNum = AboveHeight_MaxFrameNum)
                        else:
                            PredAreaPtsLoc_Fisrt = np.zeros([1, 4])
                            if PredAboveHeight.shape[0] > 0:
                                PredAboveHeight_Fisrt = -1 * np.ones([1, 7])
                                PredAboveHeight_Fisrt[0,:]= PredAboveHeight[i_area,:]
                                PredAboveHeight_Fisrt_S = AlarmDete_Pts(PredAboveHeight_Fisrt, DeteArea_One, PredAreaPtsLoc_Fisrt, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = AboveHeight_MinFrameNum, MaxFrameNum = AboveHeight_MaxFrameNum)
                            else:
                                PredAboveHeight_Fisrt = -1 * np.ones([1, 7])
                                PredAboveHeight_Fisrt_S = AlarmDete_Pts(PredAboveHeight_Fisrt, DeteArea_One, PredAreaPtsLoc_Fisrt, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = AboveHeight_MinFrameNum, MaxFrameNum = AboveHeight_MaxFrameNum)
                        PredAboveHeight_Fisrt_S_All = np.concatenate([PredAboveHeight_Fisrt_S_All, PredAboveHeight_Fisrt_S])
                    #    返回结果
                    PredAboveHeight = PredAboveHeight_Fisrt_S_All

            else: # 不在检测时间段内
                # 根据多帧结果判断是否告警
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                AlarmValidFlag = False
                if AlarmFrameNumThod == 0:
                    PredAboveHeight = PredAboveHeight
                else:
                    PredAboveHeight = AlarmDete_Pts(PredAboveHeight, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = AboveHeight_MinFrameNum, MaxFrameNum = AboveHeight_MaxFrameNum)
        else:
            PredAboveHeight = [] # 设置告警状态为 -1

            
    #    [32,"非休息时间休息"],
    CurAlarmTypeId = 32 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight']
            # 其他参数
            DeteAreaPtsInnerDis = 0.1# 往内一定区域
            DeteMinHeight = DeteBbox_BedHeight + 0.5 # 龙岗厕所区域设置高度：0.9， 为了排除桶的干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = StayLocFrameThod # 连续 5 帧才启动有效计算
            WRONGTIMELIE_MinFrameNum = -1
            WRONGTIMELIE_MaxFrameNum = AlarmFrameNumThod+5
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                WRONGTIMELIE_MinFrameNum = OneFrmContinuousFrameMin
                WRONGTIMELIE_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            # 判断文件是否被修改了
            if ConfigInfoUpdate == True: # 如果文件有更新，则重置告警结果和告警起始时间
                PredWRONGTIMELIE = []
                print('  PredWRONGTIMELIE re-setting.')
            else:
                WRONGTIMELIENum = 0 # WRONGTIMELIE Num
                if InPeriodTimeFlag == 1:
                    # 先判断目标区域的点云检测结果
                    PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                    for i_area in range(PredAreaPtsLoc.shape[0]):
                        # 计算各传感器检测的区域
                        ValidDeteArea_1 = CalIntersectAreaFromTwoArea(DeteArea[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                        CurMultiAreaEdge = CalMultiAreaEdge(MultiSensorDeteArea) # 计算多传感器检测的边界区域
#                        ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea_1, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis) # 计算超高+边界区域的有效区域
                        # CurDeteArea
                        CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                        if len(ValidDeteArea_1) > 0: # 存在交叉区域
                            ValidDeteArea = CalAboveHeightValidArea(ValidDeteArea_1, CurMultiAreaEdge, DeteAreaInnerDis = DeteAreaPtsInnerDis) # 计算超高+边界区域的有效区域
                            CurDeteArea[0] = ValidDeteArea[0] # xmin
                            CurDeteArea[1] = ValidDeteArea[1] # xmax
                            CurDeteArea[2] = ValidDeteArea[2] # ymin
                            CurDeteArea[3] = ValidDeteArea[3] # ymax
                            CurDeteArea[4] = DeteMinHeight # zmin
                            CurDeteArea[5] = DeteMaxHeight # zmax
    #                    print('CurDeteArea = {}'.format(CurDeteArea))
    
                        # 当前区域是否有未躺下人数
                        CurAlarmValidDeteArea = ValidDeteArea_1
                        NumPerson = PredInfo.shape[0] # 检测到的目标人数
                        for nump in range(NumPerson):
    #                        if PredInfo[nump][0] > DeteArea[i_area][0] and PredInfo[nump][0] < DeteArea[i_area][1] \
    #                            and PredInfo[nump][1] > DeteArea[i_area][2] and PredInfo[nump][1] < DeteArea[i_area][3] \
    #                            and ImgDeteLabel[nump] == CurModelLyingLabel: # 目标在区域内+目标不是lying
    
#                            if PredInfo[nump][0] > CurAlarmValidDeteArea[0] and PredInfo[nump][0] < CurAlarmValidDeteArea[1] \
#                                and PredInfo[nump][1] > CurAlarmValidDeteArea[2] and PredInfo[nump][1] < CurAlarmValidDeteArea[3] \
#                                and ImgDeteLabel[nump] == CurModelLyingLabel: # 目标在区域内+目标不是lying
    
#                            if PredInfo[nump][0] > CurAlarmValidDeteArea[0] and PredInfo[nump][0] < CurAlarmValidDeteArea[1] \
#                                and PredInfo[nump][1] > CurAlarmValidDeteArea[2] and PredInfo[nump][1] < CurAlarmValidDeteArea[3] \
#                                and (ImgDeteLabel[nump] == CurModelLyingLabel or PredInfo[nump][2] < DeteBbox_BedHeight+0.1): # 目标在区域内+（目标不是lying 或者 目标位置很低）
                                
                            if PredInfo[nump][0] > CurAlarmValidDeteArea[0] and PredInfo[nump][0] < CurAlarmValidDeteArea[1] \
                                and PredInfo[nump][1] > CurAlarmValidDeteArea[2] and PredInfo[nump][1] < CurAlarmValidDeteArea[3] \
                                and ImgDeteLabel[nump] == CurModelLyingLabel and PredHumanObjHeight[nump] < DeteBbox_BedHeight+0.6: # 目标在区域内 + 目标不是lying + 目标高度一定
                                      
                                # 当前帧lying 和前一帧lying 判断
                                LyingObjFlag = -1 # 初始化新目标
                                CurObjMatchResult = [] # 初始化前后两帧数距离
                                for i_WRONGTIMELIE in range(PredWRONGTIMELIE_Pre.shape[0]): # 上一帧lying 目标个数
                                    TempDist = np.sqrt((PredInfo[nump][0]-PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][3])**2 + (PredInfo[nump][1]-PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][4])**2) 
                                    if TempDist < 0.5: # 在前一帧目标范围内
                                        CurObjMatchResult.append([i_WRONGTIMELIE, TempDist])
                                # select nearset obj
                                CurObjMatchResult = np.array(CurObjMatchResult)
                                if CurObjMatchResult.shape[0]>0: # 使用最小临近结果，不使用第一个临近结果
                                    LyingObjFlag = 1 # 匹配到前一帧目标
                                    i_WRONGTIMELIE_MaxIdx = np.where(CurObjMatchResult[:,1] == np.min(CurObjMatchResult[:,1])) # 最近obj
                                    i_WRONGTIMELIE_MaxIdx = int(CurObjMatchResult[i_WRONGTIMELIE_MaxIdx,0]) # 最近obj 对应的 idx 值
                                    if PredWRONGTIMELIE_PreValid[0][i_WRONGTIMELIE_MaxIdx] == -1: # 
                                        PredWRONGTIMELIE_PreValid[0][i_WRONGTIMELIE_MaxIdx] = 1 # 匹配到后一帧目标
                                        PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2] = PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2] + 1
                                        PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2] = max(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2], -1)
                                        PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2] = min(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2], StayLocFrameThod+5)
                                        if PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2] > StayLocFrameThod: # 连续监测超过一定次数
                                            PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2], PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2], 1])
                                        else:
                                            PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE_MaxIdx][2], PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2], -1])
                                        WRONGTIMELIENum = WRONGTIMELIENum + 1 # WRONGTIMELIE 目标增 1
                                if LyingObjFlag == -1:
                                    PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, -1, PredInfo[nump][0], PredInfo[nump][1], PredInfo[nump][2], -1]) # 新目标 WRONGTIMELIE
                                    WRONGTIMELIENum = WRONGTIMELIENum + 1 # WRONGTIMELIE 目标增 1
                        # *** WRONGTIMELIE 新目标循环后, 计算前一帧WRONGTIMELIE 状态 *** #
                        for i_WRONGTIMELIE in range(PredWRONGTIMELIE_Pre.shape[0]): # 上一帧lying 目标个数
                            if PredWRONGTIMELIE_PreValid[0][i_WRONGTIMELIE] == -1: # 未匹配到下一帧目标
                                PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] - 1
                                PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = max(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], -1)
                                PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = min(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], StayLocFrameThod+5)
                                if PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] > -1 and PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2]<=StayLocFrameThod: # 连续监测超过一定次数
                                    PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][3], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][4], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][5], -1])
                                if PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2]>StayLocFrameThod: # 连续监测超过一定次数
                                    PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][3], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][4], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][5], 1])          
                                WRONGTIMELIENum = WRONGTIMELIENum + 1 # WRONGTIMELIE 目标增 1        
    #                    print('WRONGTIMELIENum = {}'.format(WRONGTIMELIENum))
                else: # 不在检测时间段内
                    # *** WRONGTIMELIE 新目标循环后, 计算前一帧WRONGTIMELIE 状态 *** #
                    for i_WRONGTIMELIE in range(PredWRONGTIMELIE_Pre.shape[0]): # 上一帧lying 目标个数
                        if PredWRONGTIMELIE_PreValid[0][i_WRONGTIMELIE] == -1: # 未匹配到下一帧目标
                            PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] - 1
                            PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = max(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], -1)
                            PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] = min(PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], StayLocFrameThod+5)
                            if PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2] > -1 and PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2]<=StayLocFrameThod: # 连续监测超过一定次数
                                PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][3], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][4], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][5], -1])
                            if PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2]>StayLocFrameThod: # 连续监测超过一定次数
                                PredWRONGTIMELIE.append([WRONGTIMELIENum, CurModelLyingLabel, PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][2], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][3], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][4], PredWRONGTIMELIE_Pre[i_WRONGTIMELIE][5], 1])          
                            WRONGTIMELIENum = WRONGTIMELIENum + 1 # WRONGTIMELIE 目标增 1 
                            
        else: # 当前传感器不检测此功能
            PredWRONGTIMELIE = [] # 设置告警状态为 -1
    
    
    #    [64,"进入三角区域"],
    
    #    [128,"内务不整"],
    
    #    [512,"单人留仓"],
    if AloneDoorAreaHeightFlag == True: # 是否添加风仓门洞下方人员检测
        CurAlarmTypeId = 16 # 当前告警 id
        if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
            CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
            if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
                # 配置文件设置参数
                if np.array(CurALARM_INFO['DeteArea']).shape[0] > 3: # 超过两个超高目标区域
                    # 风仓门洞下方检测区域
                    AboveHeightAreaThirdTypeIndex = [i_type for i_type in range(len(AboveHeightAreaTypeIdx)) if AboveHeightAreaTypeIdx[i_type]==AboveHeightAreaThirdType]
                    # DeteAreaThird
                    DeteArea_src = np.array(CurALARM_INFO['DeteArea'])
                    DeteAreaThird = DeteArea_src[AboveHeightAreaThirdTypeIndex,:][:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                    # DetePeriodTime
                    DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
                    DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
                    DeteMaxHeight = AloneDoorAreaHeightRange[1]
                    # 其他参数
                    DeteAreaPtsInnerDis = 0.05 # 往内一定区域
                    DeteMinHeight = AloneDoorAreaHeightRange[0] # 为了排除其他干扰
                    NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
                    AlarmFrameNumThod = 0 # 连续 5 帧才启动有效计算
                    
                    # 是否只使用单帧结果
                    if OneFrmDeteInfoFlag == True:
                        DeteContinuousTime = OneFrmContinuousTime
                        AlarmFrameNumThod = OneFrmContinuousFrameNum
                    
                    # 判断是否在告警时间内
                    InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
                    if InPeriodTimeFlag == 1:
                        # 计算特殊区域的人员位置信息
                        PredAreaPtsLocThird = np.empty([0, 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                        AllDeteValidAreaThird = np.zeros([6,DeteAreaThird.shape[0]])
                        # 遍历特殊超高目标区域
                        for i_area in range(DeteAreaThird.shape[0]):
                            # 计算各传感器检测的区域，存在交叉区域的传感器有效
                            ValidDeteAreaSecond = CalIntersectAreaFromTwoArea(DeteAreaThird[i_area], CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域,【暂定单个传感器只有一个检测区域】
                            CurDeteAreaThird = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                            if len(ValidDeteAreaSecond) > 0: # 有效的传感器区域
                                # CurDeteArea
                                CurDeteAreaThird[0] = ValidDeteAreaSecond[0] # xmin
                                CurDeteAreaThird[1] = ValidDeteAreaSecond[1] # xmax
                                CurDeteAreaThird[2] = ValidDeteAreaSecond[2] # ymin
                                CurDeteAreaThird[3] = ValidDeteAreaSecond[3] # ymax
                                CurDeteAreaThird[4] = DeteMinHeight # zmin
                                CurDeteAreaThird[5] = DeteMaxHeight # zmax
                            AllDeteValidAreaThird[:,i_area] = CurDeteAreaThird[:,0]
                            CurAlarmDeteHeightInfoThird = alarm_detet_door_object(CurDeteAreaThird, DeteAreaPtsInnerDis, NewPtsBGValid, FileName) # [M x 3], [x,y,z]
                            if CurAlarmDeteHeightInfoThird.shape[0] > 0: # 是否存在目标
                                PredAreaPtsLocThird = np.concatenate((PredAreaPtsLocThird, CurAlarmDeteHeightInfoThird)) # 多区域多目标累加
                        # 联合目标检测结果
#                        print('1 ', PredHUMANINFO)
                        HUMANINFOCombineObjDist = 0.4 # 暂时设置两类目标的距离大于阈值时，则合并目标
                        PredHUMANINFOCombine = PredHUMANINFO.copy()
                        for i_obj in range(PredAreaPtsLocThird.shape[0]): # 遍历特殊区域目标
                            CurAreaPtsLocThird = PredAreaPtsLocThird[i_obj, 1:]
                            CurAreaPtsLocThirdValid = True # 设置当前特殊区域目标的有效性
                            for j_obj in range(PredHUMANINFO.shape[0]): # 遍历目标
                                CurHUMANINFOLoc = PredHUMANINFO[j_obj,3:6]
                                CurTwoObjDist = np.sqrt((CurAreaPtsLocThird[0]-CurHUMANINFOLoc[0])**2 + (CurAreaPtsLocThird[1]-CurHUMANINFOLoc[1])**2)
                                if CurTwoObjDist < HUMANINFOCombineObjDist:
                                    CurAreaPtsLocThirdValid = False
                            if CurAreaPtsLocThirdValid == True:
                                CurAreaPtsLocThirdCombine = np.array([[PredHUMANINFOCombine.shape[0], 0, -1, CurAreaPtsLocThird[0], CurAreaPtsLocThird[1], CurAreaPtsLocThird[2], 1]])
                                PredHUMANINFOCombine = np.concatenate((PredHUMANINFOCombine, CurAreaPtsLocThirdCombine))
                        # 返回联合目标检测结果
                        PredHUMANINFO = PredHUMANINFOCombine.copy()
#                        print('2 ', PredHUMANINFO)
    else: # 不添加风仓门洞下方人员检测
        PredHUMANINFO = HUMANINFO_Dete # eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
                

    #    [1024,"吊拉窗户"],
    CurAlarmTypeId = 1024 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight'] + 0.25
            # 其他参数
            DeteAreaPtsInnerDis = 0.01 # 往内一定区域
            DeteMinHeight = CurALARM_INFO['DeteMaxHeight'] - 0.1 # 为了排除其他干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = 1 # 连续 5 帧才启动有效计算
            HAULWINDOW_MinFrameNum = -1
            HAULWINDOW_MaxFrameNum = AlarmFrameNumThod+5
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                HAULWINDOW_MinFrameNum = OneFrmContinuousFrameMin
                HAULWINDOW_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            if InPeriodTimeFlag == 1:
                # 先判断目标区域的点云检测结果
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                for i_area in range(PredAreaPtsLoc.shape[0]):
                    CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                    CurDeteArea[0] = DeteArea[i_area][0] # xmin
                    CurDeteArea[1] = DeteArea[i_area][1] # xmax
                    CurDeteArea[2] = DeteArea[i_area][2] # ymin
                    CurDeteArea[3] = DeteArea[i_area][3] # ymax
                    CurDeteArea[4] = DeteMinHeight # zmin
                    CurDeteArea[5] = DeteMaxHeight # zmax
                    CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName)
                    if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                        PredAreaPtsLoc[i_area, :] = CurAlarmDeteHeightInfo
                # 根据多帧结果判断是否告警
                AlarmValidFlag = True
                PredHAULWINDOW = AlarmDete_Pts(PredHAULWINDOW, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = HAULWINDOW_MinFrameNum, MaxFrameNum = HAULWINDOW_MaxFrameNum)
            else: # 不在检测时间段内
                # 根据多帧结果判断是否告警
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                AlarmValidFlag = False
                PredHAULWINDOW = AlarmDete_Pts(PredHAULWINDOW, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = HAULWINDOW_MinFrameNum, MaxFrameNum = HAULWINDOW_MaxFrameNum)
        else:
            PredHAULWINDOW[:,6] = -1
        
    #    [2048,"搭人梯"],
    CurAlarmTypeId = 2048 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight'] + 0.25
            # 其他参数
            DeteAreaPtsInnerDis = 0.01 # 往内一定区域
            DeteMinHeight = CurALARM_INFO['DeteMaxHeight'] - 0.1 # 为了排除其他干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = 1 # 连续 5 帧才启动有效计算
            BUILDLADDER_MinFrameNum = -1
            BUILDLADDER_MaxFrameNum = AlarmFrameNumThod+5
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                BUILDLADDER_MinFrameNum = OneFrmContinuousFrameMin
                BUILDLADDER_MaxFrameNum = OneFrmContinuousFrameMax
            
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            if InPeriodTimeFlag == 1:
                # 先判断目标区域的点云检测结果
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                for i_area in range(PredAreaPtsLoc.shape[0]):
                    CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                    CurDeteArea[0] = DeteArea[i_area][0] # xmin
                    CurDeteArea[1] = DeteArea[i_area][1] # xmax
                    CurDeteArea[2] = DeteArea[i_area][2] # ymin
                    CurDeteArea[3] = DeteArea[i_area][3] # ymax
                    CurDeteArea[4] = DeteMinHeight # zmin
                    CurDeteArea[5] = DeteMaxHeight # zmax
                    CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName)
                    if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                        PredAreaPtsLoc[i_area, :] = CurAlarmDeteHeightInfo
                # 根据多帧结果判断是否告警
                AlarmValidFlag = True
                PredBUILDLADDER = AlarmDete_Pts(PredBUILDLADDER, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = BUILDLADDER_MinFrameNum, MaxFrameNum = BUILDLADDER_MaxFrameNum)
            else: # 不在检测时间段内
                # 根据多帧结果判断是否告警
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                AlarmValidFlag = False
                PredBUILDLADDER = AlarmDete_Pts(PredBUILDLADDER, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = BUILDLADDER_MinFrameNum, MaxFrameNum = BUILDLADDER_MaxFrameNum)
        else:
            PredBUILDLADDER[:,6] = -1
            
    #    [4096,"站在被子上做板报"]
    CurAlarmTypeId = 4096 # 当前告警 id
    if CurAlarmTypeId in ConfigInfo['ALARM_INFO'].keys():
        CurALARM_INFO = ConfigInfo['ALARM_INFO'][CurAlarmTypeId] # 配置文件中当前告警配置信息
        if CurSensorId in CurALARM_INFO['DeteSensor']: # 当前传感器时候检测该功能
            # 配置文件设置参数
            DeteArea = np.array(CurALARM_INFO['DeteArea'])
            DeteArea = DeteArea[:,[0,2,1,3]] # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
            DetePeriodTime = np.array(CurALARM_INFO['DetePeriodTime'])
            DeteContinuousTime = CurALARM_INFO['DeteContinuousTime']
            DeteMaxHeight = CurALARM_INFO['DeteMaxHeight'] + 0.25
            # 其他参数
            DeteAreaPtsInnerDis = 0.01 # 往内一定区域
            DeteMinHeight = CurALARM_INFO['DeteMaxHeight'] - 0.1 # 为了排除其他干扰
            NearMintuePeriodTime = AlarmValidNearMintueTime # 时间段设置前后 10 分钟缓冲时间
            AlarmFrameNumThod = 1 # 连续 5 帧才启动有效计算
            STANDQUILT_MinFrameNum = -1
            STANDQUILT_MaxFrameNum = AlarmFrameNumThod+5
            # 是否只使用单帧结果
            if OneFrmDeteInfoFlag == True:
                DeteContinuousTime = OneFrmContinuousTime
                AlarmFrameNumThod = OneFrmContinuousFrameNum
                STANDQUILT_MinFrameNum = OneFrmContinuousFrameMin
                STANDQUILT_MaxFrameNum = OneFrmContinuousFrameMax
                
            # 判断是否在告警时间内
            InPeriodTimeFlag = CalInValidTime(WorldTime, DetePeriodTime, NearMintuePeriodTime)
            if InPeriodTimeFlag == 1:
                # 先判断目标区域的点云检测结果
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                for i_area in range(PredAreaPtsLoc.shape[0]):
                    CurDeteArea = np.zeros([6,1]) # [xmin, xmax, ymin, ymax, zmin, zmax]
                    CurDeteArea[0] = DeteArea[i_area][0] # xmin
                    CurDeteArea[1] = DeteArea[i_area][1] # xmax
                    CurDeteArea[2] = DeteArea[i_area][2] # ymin
                    CurDeteArea[3] = DeteArea[i_area][3] # ymax
                    CurDeteArea[4] = DeteMinHeight # zmin
                    CurDeteArea[5] = DeteMaxHeight # zmax
                    CurAlarmDeteHeightInfo = select_area_object_dete(CurDeteArea, DeteAreaPtsInnerDis, NewPts, FileName)
                    if CurAlarmDeteHeightInfo[0,0] == 1: # 是否存在目标
                        PredAreaPtsLoc[i_area, :] = CurAlarmDeteHeightInfo
                # 根据多帧结果判断是否告警
                AlarmValidFlag = True
                PredSTANDQUILT = AlarmDete_Pts(PredSTANDQUILT, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = STANDQUILT_MinFrameNum, MaxFrameNum = STANDQUILT_MaxFrameNum)
            else: # 不在检测时间段内
                # 根据多帧结果判断是否告警
                PredAreaPtsLoc = np.zeros([DeteArea.shape[0], 4]) # [目标有效标记, 位置x, 位置y, 位置z]
                AlarmValidFlag = False
                PredSTANDQUILT = AlarmDete_Pts(PredSTANDQUILT, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = STANDQUILT_MinFrameNum, MaxFrameNum = STANDQUILT_MaxFrameNum)
        else:
            PredSTANDQUILT[:,6] = -1
            
    # =============================================
    # 获取各功能检测结果
    # =============================================  
    # 内部监管区域功能： 监管区域人信息 + 厕所区域人信息
    for i_TOILET in range(PredTOILETPos.shape[0]):
        if PredTOILET_STARTTIME[i_TOILET, 1] == 1: # 厕所区域有效
            PredINTERNALSUPERVISORPos.append([len(PredINTERNALSUPERVISORPos), PredTOILETPos[i_TOILET, 1], PredTOILETPos[i_TOILET, 2], PredTOILETPos[i_TOILET, 3], PredTOILETPos[i_TOILET, 4], PredTOILETPos[i_TOILET, 5], 1])
            
    # 告警信息格式转换
    PredINTERNALSUPERVISORPos = np.array(PredINTERNALSUPERVISORPos)
    PredAboveHeight = np.array(PredAboveHeight)
    PredWRONGTIMELIE = np.array(PredWRONGTIMELIE)
    PredINTRIANGLEREGION = np.array(PredINTRIANGLEREGION)
    CurHAULWINDOW = np.array(PredHAULWINDOW)
    CurBUILDLADDER = np.array(PredBUILDLADDER)
    CurSTANDQUILT = np.array(PredSTANDQUILT)
    
    NumericResult = DetectInfo()
    NumericResult.BED = PredBedPos
    NumericResult.INTERNALSUPERVISOR = PredINTERNALSUPERVISORPos
    NumericResult.TOILET = PredTOILETPos
    NumericResult.WINDOW = PredWINDOWPos
    NumericResult.ABOVEHEIGHT = PredAboveHeight
    NumericResult.WRONGTIMELIE = PredWRONGTIMELIE
    NumericResult.INTRIANGLEREGION = PredINTRIANGLEREGION
    NumericResult.TOILET_STARTTIME = PredTOILET_STARTTIME
    NumericResult.WINDOW_STARTTIME = PredWINDOW_STARTTIME
    NumericResult.HOUSEKEEP = PredHOUSEKEEP # HOUSEKEEP
    NumericResult.HAULWINDOW = CurHAULWINDOW # HAULWINDOW
    NumericResult.BUILDLADDER = CurBUILDLADDER # BUILDLADDER
    NumericResult.STANDQUILT = CurSTANDQUILT # STANDQUILT
    NumericResult.HUMANINFO = PredHUMANINFO # HUMANINFO
    
    # 保存当前帧告警数据
#    t1 = time.time()
    if CurFrmAlarmFlag == True and ConfigSaveDataInfo_ABOVEHEIGHT == 1:
        CurSavePts = SrcPts # 存储原始的输入点云数据
        # save depth, 原始深度
#        CurFrmSaveFileName_Depth = os.path.join(SaveDataDepthDir, CurRoomId + '_' + CurSensorId + '_' + SaveDataTimeName + '.depth')
#        SavePtsAsDepth(CurSavePts, CurFrmSaveFileName_Depth) # save depth
        # save ply, 原始点云, PointCloudFuns
#        CurFrmSaveFileName_Ply = os.path.join(SaveDataPlyDir, CurRoomId + '_' + CurSensorId + '_' + SaveDataTimeName + '.ply')
#        SavePt3D2Ply(CurFrmSaveFileName_Ply, CurSavePts.transpose(), 'XYZ')
        # save ply, 原始点云, PointCloudFunsC
        CurFrmSaveFileName_Ply = os.path.join(SaveDataPlyDir, CurRoomId + '_' + CurSensorId + '_' + SaveDataTimeName + '.ply')
        CurPtsC1 = CurSavePts.transpose() # 数据格式转换
        CurPtsC = CurPtsC1.flatten().reshape([CurSavePts.shape[0], CurSavePts.shape[1]]).transpose()
        PointCloudFunsC.SavePointCloudToPly(CurFrmSaveFileName_Ply, CurPtsC)
        
#    t2 = time.time()
#    print('time = ', t2-t1)
    
    return NumericResult
    
    
def GetObjInfoFromBbox(pred_bbox, pred_labels, pred_scores, ImageFilpFlag, NewPts, DepToCptIdx, ZAxisMaxHeight):
    """
    功能：从 bbox 信息中获取目标的位置信息
    输入：
        NewPts：旋转之后的目标位置
    """
    # 无效目标数据点设置一个异常值
    OutLierValue = -1000
    
    # 图像是否需要上下翻转
    if ImageFilpFlag == 1:
        bbox = np.array(pred_bbox)
        for i_bbox in range(bbox.shape[0]):
            bbox[i_bbox,0] = DepToCptIdx.shape[0] - pred_bbox[i_bbox][2]
            bbox[i_bbox,1] = pred_bbox[i_bbox][1]
            bbox[i_bbox,2] = DepToCptIdx.shape[0] - pred_bbox[i_bbox][0]
            bbox[i_bbox,3] = pred_bbox[i_bbox][3]
    else:
        bbox = np.array(pred_bbox) # [xmin ymin xmax ymax]
    
    # labels, scores
    labels = np.array(pred_labels) # object label
#    scores = np.array(pred_scores) # object score
    # NumPerson
    NumPerson = labels.shape[0] # 当前人员个数
    
    # PtsTempAll, 保存所有目标框数据
    PtsTempAll = np.empty([1,3]) # 保存所有目标框数据
    
    # 保存原始检测的目标信息，深度图像中所有的目标信息
    HUMANINFO_Dete = np.zeros([NumPerson, 7]) # eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    
    # 人员信息保存变量
    CalcMaxHeightPtsNum = 10 # 手动设置，使用最高的部分数据点计算最高位置
    PredInfo = np.zeros([NumPerson, 3]) # PredInfo,[mean_x, mean_y, mean_z]
    PredInfoSrc = np.zeros([NumPerson, 6]) # PredInfoSrc,[mean_x, mean_y, mean_z, HeightThodPtsNum, z_std, MaxHeight]
    LyingHeightThodPts = DeteBbox_BedHeight + 0.65 # lying高度阈值 [0.7+0.55]，（WrongTimeLie 设置高度阈值）
    
    # 遍历计算人员目标信息
    for nump in range(NumPerson):
        # each people points
        bboxTemp1 = bbox[nump]
        # 选择目标框中间部分数据结果
        bboxTemp = np.zeros([4])  # 选取bbox中间部分结果数据，获取目标点云位置,20181025
        DisBorderTemp = 5 # 目标框边界范围内有效数据
        bboxTemp[0] = bboxTemp1[0] + (bboxTemp1[2]-bboxTemp1[0])/DisBorderTemp
        bboxTemp[2] = bboxTemp1[2] - (bboxTemp1[2]-bboxTemp1[0])/DisBorderTemp
        bboxTemp[1] = bboxTemp1[1] + (bboxTemp1[3]-bboxTemp1[1])/DisBorderTemp
        bboxTemp[3] = bboxTemp1[3] - (bboxTemp1[3]-bboxTemp1[1])/DisBorderTemp
        bboxTemp[0] = DepToCptIdx.shape[0] - (bboxTemp1[2] - (bboxTemp1[2]-bboxTemp1[0])/DisBorderTemp) # 图像高度方向，上下颠倒？
        bboxTemp[2] = DepToCptIdx.shape[0] - (bboxTemp1[0] + (bboxTemp1[2]-bboxTemp1[0])/DisBorderTemp)
        
        PerMask = np.zeros([DepToCptIdx.shape[0], DepToCptIdx.shape[1]])
        PerMask[int(bboxTemp[0]):int(bboxTemp[2]), int(bboxTemp[1]):int(bboxTemp[3])] = 1
        PtsMask = np.flipud(PerMask)
        PtsTemp = []
        
        PtsMask = np.reshape(PtsMask, (1,PtsMask.shape[0]*PtsMask.shape[1]))
        PtsMaskValidIdx = (PtsMask != 0)
        PtsMaskValidIdx = np.reshape(PtsMaskValidIdx,(1,PtsMaskValidIdx.shape[0]*PtsMaskValidIdx.shape[1]))
        PtsTemp = NewPts[:,PtsMaskValidIdx[0]]

        # Calculate the mean value of three axis
        PtsTemp = np.array(PtsTemp)
        PtsTemp = PtsTemp.transpose() 
        
        # 去除临近传感器无效点（）
        TempSensorHeight = ZAxisMaxHeight-0.5 # 去除原始数据点云中在原点的无效点
        HeightThodIdx = (PtsTemp[:,2]>TempSensorHeight) 
        PtsTemp[HeightThodIdx,2] = 0
        
        # 原始目标框数据结果
        bboxTempSrc = np.zeros([4])  # 原始目标框数据
        bboxTempSrc[0] = bboxTemp1[0]
        bboxTempSrc[1] = bboxTemp1[1]
        bboxTempSrc[2] = bboxTemp1[2]
        bboxTempSrc[3] = bboxTemp1[3]
        bboxTempSrc[0] = DepToCptIdx.shape[0] - bboxTemp1[2] # 图像高度方向，上下颠倒？
        bboxTempSrc[2] = DepToCptIdx.shape[0] - bboxTemp1[0]
        bboxTempSrc[0] = min(max(bboxTempSrc[0], 0), DepToCptIdx.shape[0]) # 限制有效范围
        bboxTempSrc[1] = min(max(bboxTempSrc[1], 0), DepToCptIdx.shape[1])
        bboxTempSrc[2] = min(max(bboxTempSrc[2], 0), DepToCptIdx.shape[0])
        bboxTempSrc[3] = min(max(bboxTempSrc[3], 0), DepToCptIdx.shape[1])
 
        # PerMaskSrc
        PerMaskSrc = np.zeros([DepToCptIdx.shape[0], DepToCptIdx.shape[1]])
        PerMaskSrc[int(bboxTempSrc[0]):int(bboxTempSrc[2]), int(bboxTempSrc[1]):int(bboxTempSrc[3])] = 1
        PtsMaskSrc = np.flipud(PerMaskSrc)
        PtsTempSrc = []
        # PtsMaskSrc
        PtsMaskSrc = np.reshape(PtsMaskSrc, (1,PtsMaskSrc.shape[0]*PtsMaskSrc.shape[1]))
        PtsMaskValidIdxSrc = (PtsMaskSrc != 0)
        PtsMaskValidIdxSrc = np.reshape(PtsMaskValidIdxSrc,(1,PtsMaskValidIdxSrc.shape[0]*PtsMaskValidIdxSrc.shape[1]))
        PtsTempSrc = NewPts[:,PtsMaskValidIdxSrc[0]]
        # Calculate the mean value of three axis   
        PtsTempSrc = np.array(PtsTempSrc)
        PtsTempSrc = PtsTempSrc.transpose() 
        # 去除临近传感器无效点（）
        HeightThodIdxSrc = (PtsTempSrc[:,2]>TempSensorHeight)
        PtsTempSrc[HeightThodIdxSrc,2] = 0
        # 合成所有目标数据
        PtsTempAll = np.concatenate((PtsTempAll,PtsTempSrc),axis=0) # 检测到的目标的数据点
#        PtsTempAll = np.concatenate((PtsTempAll,PtsTemp),axis=0) # 检测到的目标去除一定边界后的数据点

        # PtsTemp, mean value, 去除无效点数据, 去除行
        PtsTempMean = np.delete(PtsTemp,np.where(HeightThodIdx),axis=0) # N*3
        if PtsTempMean.shape[0] > 0: # 是否存在有效数据点
            PredInfo[nump][0] = np.mean(PtsTempMean[:,0]) # mean value
            PredInfo[nump][1] = np.mean(PtsTempMean[:,1])
            PredInfo[nump][2] = np.mean(PtsTempMean[:,2])
        else: # 有效目标数据为空
            PredInfo[nump][0] = OutLierValue # mean value
            PredInfo[nump][1] = OutLierValue
            PredInfo[nump][2] = OutLierValue
        
        # PtsTempSrc, mean value, 去除无效点数据, 去除行
        PtsTempMeanSrc = np.delete(PtsTempSrc,np.where(HeightThodIdxSrc),axis=0) # N*3
        
        if PtsTempMeanSrc.shape[0] > CalcMaxHeightPtsNum: # 目标存在一定点数
            PtsTempMeanSrcMaxPts = np.sort(PtsTempMeanSrc[:,2])[-CalcMaxHeightPtsNum:] # 选择最高部分数据点
            PtsTempMeanSrcMaxPtsStd = np.std(PtsTempMeanSrcMaxPts) # 方差
            PtsTempMeanSrcMaxPtsMean = np.mean(PtsTempMeanSrcMaxPts) # 均值
            PtsTempMeanSrcMaxPtsSelectIndex = (PtsTempMeanSrcMaxPts>PtsTempMeanSrcMaxPtsMean-PtsTempMeanSrcMaxPtsStd) & \
                                        (PtsTempMeanSrcMaxPts<PtsTempMeanSrcMaxPtsMean+PtsTempMeanSrcMaxPtsStd) # 排除部分无效超高点
            PtsTempMeanSrcMaxPtsSelect = PtsTempMeanSrcMaxPts[PtsTempMeanSrcMaxPtsSelectIndex] # 去除无效点后数据点
            CurMaxHeight = np.mean(PtsTempMeanSrcMaxPtsSelect) # 计算目标高度，数据均值
        else:
            if PtsTempMeanSrc.shape[0] > 0:
                CurMaxHeight = np.mean(PtsTempMeanSrc[:,2])
            else:
                CurMaxHeight = OutLierValue
        
        if PtsTempMeanSrc.shape[0] > 0:
            PredInfoSrc[nump][0] = np.mean(PtsTempMeanSrc[:,0]) # mean value
            PredInfoSrc[nump][1] = np.mean(PtsTempMeanSrc[:,1])
            PredInfoSrc[nump][2] = np.mean(PtsTempMeanSrc[:,2])
            PredInfoSrc[nump][3] = len((np.where(PtsTempMeanSrc[:,2]>LyingHeightThodPts))[0]) # 超过阈值点的个数
            PredInfoSrc[nump][4] = np.std(PtsTempMeanSrc[:,2]) # z_std
            PredInfoSrc[nump][5] = CurMaxHeight# 人员目标最高点位置
        else:
            PredInfoSrc[nump][0] = OutLierValue # mean value
            PredInfoSrc[nump][1] = OutLierValue
            PredInfoSrc[nump][2] = OutLierValue
            PredInfoSrc[nump][3] = 0 # 超过阈值点的个数
            PredInfoSrc[nump][4] = 0 # z_std
            PredInfoSrc[nump][5] = CurMaxHeight# 人员目标最高点位置
            
        # detect human info 
        HUMANINFO_Dete[nump, 0] = nump
        HUMANINFO_Dete[nump, 1] = labels[nump] # object label 
        HUMANINFO_Dete[nump, 2] = -1 
        HUMANINFO_Dete[nump, 3:6] = PredInfo[nump, 0:3] # PredInfo object mean value
        HUMANINFO_Dete[nump, 6] = 1

    return PredInfo, PredInfoSrc, HUMANINFO_Dete

def AlarmDete_Loc(PreAlarmDeteInfo, DeteHumanInfo, ImgDeteLabel, ImgDeteScore, DeteArea, DeteMinHeight, DeteMaxHeight, PredAreaPtsLoc=None):
    """
    功能：根据人员目标位置进行判断
    输入：
        PreAlarmDeteInfo：当前告警前一帧数的信息
        DeteHumanInfo：检测到的人员的位置
        DeteArea：告警检测区域
        DetePeriodTime：检测时间段
        DeteContinuousTime：持续帧数
        DeteMaxHeight：超高高度
    """
    # NumPerson
    NumPerson = DeteHumanInfo.shape[0] # 检测到的目标人数
    NumAlarmArea = DeteArea.shape[0] # 当前告警区域个数
    for nump in range(NumPerson):
        for numa in range(NumAlarmArea):
            PreAlarmDeteInfo[numa, 0] = numa
            # 是否结合点云判断
#            print('PredAreaPtsLoc = {}'.format(PredAreaPtsLoc))
            if PredAreaPtsLoc is None:
                if DeteHumanInfo[nump][0] > DeteArea[numa][0] and DeteHumanInfo[nump][0] < DeteArea[numa][1] \
                    and DeteHumanInfo[nump][1] > DeteArea[numa][2] and DeteHumanInfo[nump][1] < DeteArea[numa][3]:
                    PreAlarmDeteInfo[numa,1] = ImgDeteLabel[nump]
                    PreAlarmDeteInfo[numa,3:6] = DeteHumanInfo[nump]
            else:
                if DeteHumanInfo[nump][0] > DeteArea[numa][0] and DeteHumanInfo[nump][0] < DeteArea[numa][1] \
                    and DeteHumanInfo[nump][1] > DeteArea[numa][2] and DeteHumanInfo[nump][1] < DeteArea[numa][3] and PredAreaPtsLoc[numa,0] == 1:
                    PreAlarmDeteInfo[numa,1] = ImgDeteLabel[nump]
                    PreAlarmDeteInfo[numa,3:6] = DeteHumanInfo[nump]
    
    return PreAlarmDeteInfo

def AlarmDete_SequenceFrame(PreAlarmDeteInfo, AlarmFrameNumThod, DeteContinuousTime, WTime, STARTTIME = None, MinFrameNum = -1, MaxFrameNum = 15, ContinuousFrameUpRate = 1, ContinuousFrameDnRate = 1):
    """
    功能：检测连续帧结果
    """
    for i in range(PreAlarmDeteInfo.shape[0]):
        if PreAlarmDeteInfo[i][1] != -1: # have person
            PreAlarmDeteInfo[i][2] = PreAlarmDeteInfo[i][2] + 1*ContinuousFrameUpRate # person detected time add 1
        else:
            PreAlarmDeteInfo[i][2] = PreAlarmDeteInfo[i][2] - 1*ContinuousFrameDnRate # person detected time sub 1
        PreAlarmDeteInfo[i][2] = max(PreAlarmDeteInfo[i][2], MinFrameNum) # value scale of CurDetectInfo.TOILET warning number
        PreAlarmDeteInfo[i][2] = min(PreAlarmDeteInfo[i][2], MaxFrameNum) # init: 10
        # 是否有对应的持续时间设置
        if not STARTTIME is None: # 如果有持续时间段设置
            if PreAlarmDeteInfo[i][2] >= AlarmFrameNumThod: # stay one position
                    if STARTTIME[i][1]==-1:  
                        STARTTIME[i][0] = WTime
                        STARTTIME[i][1] = 1
                    elif (STARTTIME[i][1]==1):
                        if ((WTime - STARTTIME[i][0]) > DeteContinuousTime*60):
                            PreAlarmDeteInfo[i][6] = 1
            else:
                STARTTIME[i][1] = -1
        else: # 如果没有持续时间段设置
            if PreAlarmDeteInfo[i][2] >= AlarmFrameNumThod: # stay one position
                PreAlarmDeteInfo[i][6] = 1
            else:
                PreAlarmDeteInfo[i][6] = -1
    
    return PreAlarmDeteInfo, STARTTIME

    
def AlarmDete_Pts(PreAlarmDeteInfo, DeteArea, PredAreaPtsLoc, AlarmFrameNumThod, DeteContinuousTime, AlarmValidFlag, MinFrameNum = -1, MaxFrameNum = 15):
    """
    功能：通过点云对目标进行检测
    """
#    print('PreAlarmDeteInfo = {}, PredAreaPtsLoc = {}'.format(PreAlarmDeteInfo, PredAreaPtsLoc))

    # DeteArea
    for i_Area in range(DeteArea.shape[0]):
        CurAlarmPredHeightInfo = PredAreaPtsLoc[i_Area]
        PreAlarmDeteInfo[i_Area,:] = sequence_frame_object_dete(DeteArea[i_Area,:-1], PreAlarmDeteInfo[i_Area, :], CurAlarmPredHeightInfo, AlarmFrameNumThod, AlarmValidFlag, SequenceDeteFrameMin = MinFrameNum, SequenceDeteFrameMax = MaxFrameNum)

    return PreAlarmDeteInfo

if __name__ == '__main__':
    print('Start AlarmDete.')
    
    
    
    print('End AlarmDete.')
    