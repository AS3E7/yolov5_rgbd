# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:30:17 2019

@author: HYD
"""

import numpy as np
import time
import sys
import os
from shutil import copyfile
import logging as lgmsg
 
from matplotlib import path
import sklearn.cluster as skc  
try:
    import configparser
except:
    from six.moves import configparser

from detect_alarm.Fighting_Detection import Fighting_Detection
from util.AreaFuns import CalMultiAreaEdge, CalAboveHeightValidArea
from util.PolygonFuns import TransBboxToPoints, TransPointsToBbox
from util.ReadConfig import ReadSensorConfig, ReadIniConfigGlobalInfo, ReadReviseTime

from config import multi_sensor_params as params
from config import online_offline_type, log_info, alarm_level
from util.SaveData import DeleteOutLimitFiles, LogSaveCategoryIndexFun
from detect_alarm.AlarmDeteTime import CalInValidTime, CalInValidTimeTransPeriodTime
from detect_alarm.AlarmIDInfo import AlarmIDInfo
from util.PolygonFuns import PointsInPolygon
from detect_alarm.ObjTrack import TwoFrameObjMatch
from detect_alarm.MultiFrmAlarmDete import SequenceFrmAlarmDete

sys.path.append("..")

#####################################################################
# log 信息
ConfigDebugInfo = log_info['debug_info']
ConfigDebugTypeIndex = LogSaveCategoryIndexFun()
#####################################################################


#####################################################################
# 在线/离线文件地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SensorConfigFileName = './demo/SensorConfig.ini'
    SensorConfigReviseFileName = "./demo/log/FileReviseTime.log"
    SensorConfigCopyFileName = './demo/result/backup/SensorConfig.ini'
else:
    SensorConfigFileName = 'SensorConfig.ini'
    SensorConfigReviseFileName = "log/FileReviseTime.log"
    SensorConfigCopyFileName = 'result/backup/SensorConfig.ini'
##################################################################### 


#####################################################################
# 读取配置文件
global FileReviseBool, SensorInfoAll, SleepTime, INTERNALSUPERVISOR_HumanNum
FileReviseBool = False
# 读取配置文件
SensorInfoAll = ReadIniConfigGlobalInfo(SensorConfigFileName) # 读取配置文件信息
#if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0: # 打印配置文件读取信息
#    lgmsg.info('第一次读取配置文件:SensorConfig.ini')
#    lgmsg.info('    SensorConfig info: {}'.format(SensorInfoAll))

# 备份配置文件 SensorConfig.ini
SensorConfigCopyFileNameSave = SensorConfigCopyFileName.replace('.ini', '_' + str(os.stat(SensorConfigFileName).st_mtime)+'.ini')
if not os.path.exists(SensorConfigCopyFileNameSave):
    copyfile(SensorConfigFileName, SensorConfigCopyFileNameSave) # 复制 配置 文件
    
# 告警松紧度等级信息
WarnLooseDegree = SensorInfoAll['ROOM_INFO']['managelevel'] # 告警宽松度
CurDeteAlarmLevelInfo = alarm_level[str(WarnLooseDegree)]
#if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
#    lgmsg.info('第一次获取告警松紧度等级信息')
#    lgmsg.info('CurDeteAlarmLevelInfo {} = {}'.format(WarnLooseDegree, CurDeteAlarmLevelInfo))
#####################################################################


#####################################################################
# 使用 multi_sensor_params 设置
# 单人留仓人数设置，init:1
AloneStayHumanNum = params['alone_stay_person_num']
# 持续帧数设置
StayLocFrameThod = params['alarm_valid_frame_num'] # init: 10
AlarmFrameLowerLimit = params['alarm_valid_frame_num'] # 告警有效连续帧数下限值
# 内部监管检测设置
#InternalSupervisorContinuousTime = params['internal_supervisor_continuous_time'] # 状态持续时间，init: 1 分钟
InternalSupervisorRoomIndex = [0] # 监管区域所在房间，【0为主仓，1为副仓, 2为其他仓】
# 群体冲突检测设置
ConflictParamHumanNum = params['conflict_person_num'] # init:4
ConflictParamNearRadius = params['conflict_near_radius'] # init:0.3,0.63,0.66,0.76
# 邻近休息时间段前后 10 min 为无效时间段
NearMintueSLEEPTIME = params['alarm_valid_near_time'] # 单位：min
NearMintueDeteTIME = params['alarm_valid_near_time']
# 房间编号
RoomSensorGroup = params['alone_stay_sensor_group'] # 如：[[6,7,8,9,10],[1,2,3,4,5]]
#####################################################################


#####################################################################
# 连续帧累加速率设置
AlarmContinuousFrameUpRate = 1 # 连续帧累计上升速率,1
AlarmContinuousFrameDnRate = 1 # 连续帧累计下降速率,2
#####################################################################


#####################################################################
# 初始化告警ID
global GlobalDeteAlarmID
GlobalDeteAlarmID = 0
GlobalDeteAlarmIDMax = 10**4 #10**4
# 告警目标信息特征个数, [告警类型ID，告警状态，告警ID，传感器编号，位置X，位置Y，位置Z，其他告警信息，人员姿态，累计告警帧数，累计告警起始时间戳，累计告警时长，告警房间编号]
GlobalDeteAlarmFeatureNum = 13
# 前后两帧目标匹配距离，单位：米
TwoFrameMatchDistThod = 0.7 # 1.0
# 多传感器目标融合基础参数
MultiSensorFusionDeteObjEdgeDistBase = 0.4 # 基础检测区域外边界
MultiSensorFusionDeteMultiObjsCluDistBase = 0.5 # 基础检测多传感器聚类距离

# 获取告警ID序号
global AlarmIDExtraInfo
AlarmIDIndex = AlarmIDInfo() # 告警ID 序号
AlarmIDExtraInfo = dict() # 告警ID对应的额外信息，[持续帧数，持续帧下降速率，持续帧上升速率，持续时长]
for id_index in AlarmIDIndex.AlarmIDAll:
#    print('id_index = ', id_index)
    AlarmIDExtraInfo[id_index] = dict()
    # 计算各告警对应的额外信息
    # 未在指定时间休息，ID=1
    if AlarmIDIndex.Alarm_NoLying == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()):
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['wrong_time_nolie_frame_num'] # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['wrong_time_nolie_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
    # 未在指定区域监督，ID=2
    elif AlarmIDIndex.Alarm_InternalSupervisor == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = SensorInfoAll['OTHER']['personthreshold'] # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['internal_supervisor_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.3
    # 厕所区域异常，ID=4
    elif AlarmIDIndex.Alarm_Toilet == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()):
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['toilet_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
    # 高度异常，ID=16
    elif AlarmIDIndex.Alarm_AboveHeight == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['above_height_continuous_frame_num'] # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['above_height_continuous_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
    # 非休息时间休息，ID=32
    elif AlarmIDIndex.Alarm_WrongLying == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['wrong_time_lie_frame_num'] # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['wrong_time_lie_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
    # 单人留仓，ID=512
    elif AlarmIDIndex.Alarm_Alone == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = AloneStayHumanNum # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['alone_continuous_frame_num'] # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['alone_continuous_time']
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime  
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界, 0.2 / 0.6
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离, 0.25 / 0.8

    else: # 其他类型告警
        AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
        AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
        AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
        if id_index in SensorInfoAll['ALARM_INFO'].keys():
            AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
        else:
            AlarmIDConfigDeteContinuousTime = 0
        if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = 0
        else:
            AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
        AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界
        AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离， 0.25
#####################################################################
            

#####################################################################
# 单人留仓区域类型设置，
AloneAreaDefaultType = 1 # 其中：1 表示默认两个单人留仓区域，2 表示特殊单人留仓区域（中间墙门洞）
AloneAreaSecondType = 2
AloneAreaTypeIdx = [1,1,2] # 超高区域类型设置，不同应用场景设置不同
AlonetAreaDefaultTypeIndex = [i_type for i_type in range(len(AloneAreaTypeIdx)) if AloneAreaTypeIdx[i_type]==AloneAreaDefaultType]
AlonetAreaSecondTypeIndex = [i_type for i_type in range(len(AloneAreaTypeIdx)) if AloneAreaTypeIdx[i_type]==AloneAreaSecondType]
#####################################################################                                         


#####################################################################
# 全局目标检测信息
#####################################################################
class GlobalInfo(): # HumanInfoStr
    def __init__(self, HumanInfoStr):
        global FileReviseBool, SensorInfoAll
        global GlobalDeteAlarmID, AlarmIDExtraInfo
        
        self.HumanInfoStr = HumanInfoStr # input human info string
#        self.HumanInfoStr = '{1001,[10003,(1, 0, 1.2530, 1.0160, 1.2590),(1, 0, 1.2630, 1.0160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2060, 1.1430, 1.2590)]}'
        
        self.DetectInfo = [] # detect alarm info
        self.isUpdate = False # 配置文件是否被修改
        #配置文件读取
        FileReviseBool = ReadReviseTime(SensorConfigFileName, SensorConfigReviseFileName)
        if FileReviseBool == False:
            self.isUpdate = True # 配置文件被修改
            SensorInfoAll = ReadIniConfigGlobalInfo(SensorConfigFileName) # 再次读取配置文件
            WarnLooseDegree = SensorInfoAll['ROOM_INFO']['managelevel']
            CurDeteAlarmLevelInfo = alarm_level[str(WarnLooseDegree)] # CurDeteAlarmLevelInfo
            # 备份配置文件 SensorConfig.ini
            SensorConfigCopyFileNameSave = SensorConfigCopyFileName.replace('.ini', '_' + str(os.stat(SensorConfigFileName).st_mtime)+'.ini')
            if not os.path.exists(SensorConfigCopyFileNameSave):
                copyfile(SensorConfigFileName, SensorConfigCopyFileNameSave) # 复制 配置 文件
            # 打印信息
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
                lgmsg.info('reLoad Ini Config File Info.')
                lgmsg.info('isUpdate = {}, CurDeteAlarmLevelInfo {} = {}'.format(self.isUpdate, WarnLooseDegree, CurDeteAlarmLevelInfo))
            # 更新信息
            # 更新 AlarmIDExtraInfo 信息
            for id_index in AlarmIDIndex.AlarmIDAll:
                AlarmIDExtraInfo[id_index] = dict()
                # 计算各告警对应的额外信息
                # 未在指定时间休息，ID=1
                if AlarmIDIndex.Alarm_NoLying == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()):
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['wrong_time_nolie_frame_num'] # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['wrong_time_nolie_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
                # 未在指定区域监督，ID=2
                elif AlarmIDIndex.Alarm_InternalSupervisor == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = SensorInfoAll['OTHER']['personthreshold'] # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['internal_supervisor_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.3
                # 厕所区域异常，ID=4
                elif AlarmIDIndex.Alarm_Toilet == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()):
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['toilet_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
                # 高度异常，ID=16
                elif AlarmIDIndex.Alarm_AboveHeight == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['above_height_continuous_frame_num'] # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['above_height_continuous_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
                # 非休息时间休息，ID=32
                elif AlarmIDIndex.Alarm_WrongLying == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['wrong_time_lie_frame_num'] # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['wrong_time_lie_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界,0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离,0.25
                # 单人留仓，ID=512
                elif AlarmIDIndex.Alarm_Alone == id_index and (id_index in SensorInfoAll['ALARM_INFO'].keys()): 
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = AloneStayHumanNum # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = CurDeteAlarmLevelInfo['alone_continuous_frame_num'] # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = CurDeteAlarmLevelInfo['alone_continuous_time']
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime  
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界, 0.2 / 0.6
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离, 0.25 / 0.8
        
                else: # 其他类型告警
                    AlarmIDExtraInfo[id_index]['DeteHumanNum'] = 1 # 告警人数阈值
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrame'] = AlarmFrameLowerLimit # 检测连续帧数
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameUpRate'] = AlarmContinuousFrameUpRate # 检测连续帧增长速度
                    AlarmIDExtraInfo[id_index]['DeteContinuousFrameDnRate'] = AlarmContinuousFrameDnRate # 检测连续帧减小速度
                    if id_index in SensorInfoAll['ALARM_INFO'].keys():
                        AlarmIDConfigDeteContinuousTime = SensorInfoAll['ALARM_INFO'][id_index]['DeteContinuousTime'] # 原始配置文件中的'DeteContinuousTime'
                    else:
                        AlarmIDConfigDeteContinuousTime = 0
                    if AlarmIDConfigDeteContinuousTime < 0.0001: # 是否存在UI设置告警时长
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = 0
                    else:
                        AlarmIDExtraInfo[id_index]['DeteContinuousTime'] = AlarmIDConfigDeteContinuousTime
                    AlarmIDExtraInfo[id_index]['DeteObjEdgeDist'] = MultiSensorFusionDeteObjEdgeDistBase # 检测区域外边界，0.2
                    AlarmIDExtraInfo[id_index]['DeteMultiObjsCluDist'] = MultiSensorFusionDeteMultiObjsCluDistBase # 检测多传感器聚类距离
            # 打印信息 AlarmIDExtraInfo
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
                lgmsg.info('New AlarmIDExtraInfo = {}'.format(AlarmIDExtraInfo))

        else:
            self.isUpdate =False

        # 读取房间号
        RoomID = '1001'
        RoomIdStrSplit = self.HumanInfoStr.split('{')
        for RoomIdInfo in RoomIdStrSplit: # sensor series
            if len(RoomIdInfo)>0:
                RoomIdInfoSplit = RoomIdInfo.split(',')
                if not len(RoomIdInfoSplit) == 1: # 如果存在  ','
                    RoomID = RoomIdInfoSplit[0] # one room ID
                else: # 如果不存在  ','， 如：'{1002}'
                    RoomIdInfoSplit = RoomIdInfo.split('}')
                    RoomID = RoomIdInfoSplit[0]
        # select one room info from SensorConfig.ini
        SensorInfo = dict()
        if SensorInfoAll['ROOM_INFO']['roomid'] == RoomID: # 是否存在对应房间编号的配准信息
            SensorInfo = SensorInfoAll
        else:
            lgmsg.error('ROOM_INFO roomid error.')
        self.SensorInfo = SensorInfo
        # sensor name group
        SenorNameGroupName = []
        for i_sensor_name in SensorInfo['SENSORS_INFO']['DeteAreaPoints']:
            SenorNameGroupName.append(i_sensor_name)
        self.SenorNameGroupName = SenorNameGroupName
            
        # current room info, DETEAREA
        self.MultiSensorDeteArea = SensorInfo['SENSORS_INFO']['DeteAreaPoints']
        # current room info, ROOMAREA
        CurROOMAREA = SensorInfo['ALARM_INFO'][512]['DeteAreaPoints']
        if len(CurROOMAREA) > 2:
            # 默认单人留仓检测区域
            CurMainAndSubRoom = dict()
            AlonetAreaDefaultTypeIndex = [i_type for i_type in range(len(AloneAreaTypeIdx)) if AloneAreaTypeIdx[i_type]==AloneAreaDefaultType]
            for i_DefaultType in AlonetAreaDefaultTypeIndex:
                CurMainAndSubRoom[i_DefaultType] = CurROOMAREA[i_DefaultType]
            self.MainAndSubRoom = CurMainAndSubRoom
            # 中间墙进出口单人留仓检测区域
            CurMainAndSubRoomEntrance = dict()
            AlonetAreaSecondTypeIndex = [i_type for i_type in range(len(AloneAreaTypeIdx)) if AloneAreaTypeIdx[i_type]==AloneAreaSecondType]
            for i_SecondTypeIndex in AlonetAreaSecondTypeIndex:
                CurMainAndSubRoomEntrance[i_SecondTypeIndex] = CurROOMAREA[i_SecondTypeIndex]
            self.MainAndSubRoomEntrance = CurMainAndSubRoomEntrance
            self.MainAndSubRoomEntranceIndex = AlonetAreaSecondTypeIndex
        else:
            self.MainAndSubRoom = CurROOMAREA
            self.MainAndSubRoomEntrance = []
            self.MainAndSubRoomEntranceIndex = []

        # SENSOREDGEAREA 通过 DETEAREA 计算得到
        CombineMulti_DETEAREA = []
        for i_DETEAREA in SensorInfo['SENSORS_INFO']['DeteAreaPoints']:
            CombineMulti_DETEAREA.append(SensorInfo['SENSORS_INFO']['DeteAreaPoints'][i_DETEAREA])
        self.MultiSensorEdgeArea = np.array(CalMultiAreaEdge(CombineMulti_DETEAREA))

        # CurRoomInfoGroup
        CurRoomInfoGroup = self.read_human_info_str() # input human info numeric
        self.LostDataSensorGroup = CurRoomInfoGroup['SensorLostGroup'] # LostDataSensorGroup
        self.MultiSensorWTimeGroup = CurRoomInfoGroup['SensorTime'] # MultiSensorWTimeGroup

    def detect_alarm(self, CurDeteObjInfo, PreAllAlarmDeteInfo):
        """
        Calculate alarm: one person alone in one room;
        Inputs:
            CurDeteObjInfo: current frame object information
            PreAllAlarmDeteInfo: previous frane detect information
        Outputs:
            CurDeteInfo: current frame detect information
            CurDeteState: current frame alarm detect information
        """
        global GlobalDeteAlarmID, AlarmIDExtraInfo
        # =============================================
        # self 初始化
        # =============================================
        # 配置文件是否被修改,【如果文件被修改，则和连续时间戳相关的告警记录重新开始】
        ConfigInfoUpdate = self.isUpdate
        # 各传感器区域信息
        MultiSensorDeteArea = self.MultiSensorDeteArea # 多边形数据
        MultiSensorEdgeArea = self.MultiSensorEdgeArea
        # 单人留仓房间信息
        MainAndSubRoom = self.MainAndSubRoom
        MainAndSubRoomEntrance = self.MainAndSubRoomEntrance
        MainAndSubRoomEntranceIndex = self.MainAndSubRoomEntranceIndex
        # SensorInfo
        SensorInfo = self.SensorInfo
        ConfigSensorNum = SensorInfo['ROOM_INFO']['sensornum']
        
        # MultiSensorWTimeGroup
        MultiSensorWTimeGroup = self.MultiSensorWTimeGroup
        
        # LostDataSensorGroup
        LostDataSensorGroup = self.LostDataSensorGroup
        if len(LostDataSensorGroup) > 0:
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                lgmsg.debug('LostDataSensorGroup = {}'.format(LostDataSensorGroup))
        
        # GlobalDeteAlarmID
        if GlobalDeteAlarmID > GlobalDeteAlarmIDMax:
            GlobalDeteAlarmID = GlobalDeteAlarmID%GlobalDeteAlarmIDMax
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                lgmsg.debug('reStart GlobalDeteAlarmID = {}'.format(GlobalDeteAlarmID))

        # 当前时间
        WTime = time.time()

        # =============================================
        # 初始化告警信息
        # =============================================        
        # 初始化各类告警信息, [告警类型ID，告警状态，告警ID，传感器编号，位置X，位置Y，位置Z，其他告警信息，人员姿态，累计告警帧数，累计告警起始时间戳，累计告警时长, 仓室编号]
        PreAllAlarmDeteInfo = np.array(PreAllAlarmDeteInfo)
        # 前一帧告警信息转换
        PreMultiFrameMultiSensorAlarmInfo = dict()
        for alarm_id in SensorInfo['ALARM_INFO']:
            PreMultiFrameMultiSensorAlarmInfo[alarm_id] = dict()
            if (PreAllAlarmDeteInfo.shape[0] == 0) or (ConfigInfoUpdate == True): # 无告警记录 或者 配置文件发生改变
                for i_room in range(len(MainAndSubRoom)):
                    PreFrameMultiSensorOneRoom = dict()
                    PreFrameMultiSensorOneRoom['AlarmType'] = alarm_id
                    PreFrameMultiSensorOneRoom['AlarmState'] = -1
                    PreFrameMultiSensorOneRoom['AlarmID'] = -1
                    PreFrameMultiSensorOneRoom['SensorID'] = 10001
                    PreFrameMultiSensorOneRoom['ObjX'] = -1
                    PreFrameMultiSensorOneRoom['ObjY'] = -1
                    PreFrameMultiSensorOneRoom['ObjZ'] = -1
                    PreFrameMultiSensorOneRoom['ReservedInfo'] = -1
                    
                    PreFrameMultiSensorOneRoom['ObjLabel'] = -1
                    PreFrameMultiSensorOneRoom['SumFrame'] = -1
                    PreFrameMultiSensorOneRoom['StartWorldTime'] = -1
                    PreFrameMultiSensorOneRoom['SumTime'] = -1
                    PreFrameMultiSensorOneRoom['RoomID'] = -1
                    PreMultiFrameMultiSensorAlarmInfo[alarm_id][i_room] = PreFrameMultiSensorOneRoom # 初始化每类一个告警
            else:
                # 获取前一帧各告警状态            
                PreAllAlarmDeteInfo_ValidObj = np.empty([0, GlobalDeteAlarmFeatureNum])
                CurAlarmValidObjNum = -1
                for i_alan in range(PreAllAlarmDeteInfo.shape[0]):
                    if PreAllAlarmDeteInfo[i_alan,0] == alarm_id and PreAllAlarmDeteInfo[i_alan,1] != -1: # 告警ID类型和告警状态有效
                        PreAllAlarmDeteInfo_ValidObj = np.concatenate((PreAllAlarmDeteInfo_ValidObj, np.expand_dims(PreAllAlarmDeteInfo[i_alan,:], axis=0)))
                        # PreFrameMultiSensorOneRoom
                        CurAlarmValidObjNum = CurAlarmValidObjNum + 1
                        PreFrameMultiSensorOneRoom = dict()
                        PreFrameMultiSensorOneRoom['AlarmType'] = alarm_id
                        PreFrameMultiSensorOneRoom['AlarmState'] = PreAllAlarmDeteInfo[i_alan,1]
                        PreFrameMultiSensorOneRoom['AlarmID'] = PreAllAlarmDeteInfo[i_alan,2]
                        PreFrameMultiSensorOneRoom['SensorID'] = PreAllAlarmDeteInfo[i_alan,3]
                        PreFrameMultiSensorOneRoom['ObjX'] = PreAllAlarmDeteInfo[i_alan,4]
                        PreFrameMultiSensorOneRoom['ObjY'] = PreAllAlarmDeteInfo[i_alan,5]
                        PreFrameMultiSensorOneRoom['ObjZ'] = PreAllAlarmDeteInfo[i_alan,6]
                        PreFrameMultiSensorOneRoom['ReservedInfo'] = PreAllAlarmDeteInfo[i_alan,7]
                        
                        PreFrameMultiSensorOneRoom['ObjLabel'] = PreAllAlarmDeteInfo[i_alan,8]
                        PreFrameMultiSensorOneRoom['SumFrame'] = PreAllAlarmDeteInfo[i_alan,9]
                        PreFrameMultiSensorOneRoom['StartWorldTime'] = PreAllAlarmDeteInfo[i_alan,10]
                        PreFrameMultiSensorOneRoom['SumTime'] = PreAllAlarmDeteInfo[i_alan,11]
                        PreFrameMultiSensorOneRoom['RoomID'] = PreAllAlarmDeteInfo[i_alan,12]
                        PreMultiFrameMultiSensorAlarmInfo[alarm_id][CurAlarmValidObjNum] = PreFrameMultiSensorOneRoom # 初始化每类一个告警

        # =============================================
        # 融合当前帧各类型告警
        # =============================================
        # 遍历各个告警，当前帧检测的融合结果（按仓室分开）
        CurFrameMultiSensorAlarmInfo = dict()
        for alarm_id in SensorInfo['ALARM_INFO']:
#            print('alarm_id = ', alarm_id)
            CurFrameMultiSensorAlarmInfo[alarm_id] = dict()
            CurAlarmConfigInfo = SensorInfo['ALARM_INFO'][alarm_id]
            # 时间
            CurAlarmDetePeriodTime = CurAlarmConfigInfo['DetePeriodTime']
            CurAlarmDetePeriodTime = CalInValidTimeTransPeriodTime(np.array(CurAlarmDetePeriodTime), NearMintueDeteTIME)
            CurAlarmDetePeriodTimeFlag = CalInValidTime(WTime, np.array(CurAlarmDetePeriodTime), NearMintueDeteTIME) # 是否在检测时间内
            # 区域
            CurAlarmDeteArea = CurAlarmConfigInfo['DeteAreaPoints'] # [N x 8]
            # 持续帧数
            CurAlarmDeteContinuousFrame = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrame']
            CurAlarmDeteContinuousFrameUpRate = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrameUpRate']
            CurAlarmDeteContinuousFrameDnRate = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrameDnRate']
            # 持续时间
            CurAlarmDeteContinuousTime = AlarmIDExtraInfo[alarm_id]['DeteContinuousTime']
            # 人数阈值
            CurAlarmDeteHumanNum = AlarmIDExtraInfo[alarm_id]['DeteHumanNum']
            # 边界融合
            AloneObjEdgeDist = AlarmIDExtraInfo[alarm_id]['DeteObjEdgeDist'] # 区域边界区域, 再增加边界 0.2m
            AloneMultiObjsDisThod = AlarmIDExtraInfo[alarm_id]['DeteMultiObjsCluDist'] # 多目标之间的邻近距离阈值, init:0.25m
            
           
            CurAlarmDeteObjInfo = GlobalInfo.fuse_multi_sensor_dete_info(CurDeteObjInfo[alarm_id], MultiSensorDeteArea, MultiSensorEdgeArea, AddEdgeDist = AloneObjEdgeDist, MultiObjsDisThod = AloneMultiObjsDisThod) # 检测各目标点融合
            CurAlarmDeteObjSensorInfo = GlobalInfo.find_obj_sensor_info(CurAlarmDeteObjInfo, MultiSensorDeteArea, MultiSensorEdgeArea) # 检测各目标点所处传感器编号
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                if len(CurAlarmDeteObjInfo) > 0:
                    lgmsg.debug('alarm_id = {}, fuse_multi_sensor CurAlarmDeteObjInfo = {}'.format(alarm_id, CurAlarmDeteObjInfo))
                    lgmsg.debug('alarm_id = {}, find_obj_sensor CurAlarmDeteObjSensorInfo = {}'.format(alarm_id, CurAlarmDeteObjSensorInfo))
            # 有效目标检测（按告警划定区域）
            CurAlarmDeteAreaHumanInfo = dict()
            for i_area in range(CurAlarmDeteArea.shape[0]): # CurFrameMultiSensorAlarmInfo
                CurAlarmDeteAreaHumanInfo[i_area] = dict()
                CurAlarmDeteAreaHumanInfo[i_area]['ObjInfo'] = []
                CurAlarmDeteAreaHumanInfo[i_area]['ObjNum'] = 0
                CurAlarmDeteOneArea = CurAlarmDeteArea[i_area]
                # 判断一个点是否在指定区域内
                if len(CurAlarmDeteObjInfo) > 0:
                    CurAlarmDeteAreaHumanFlag = PointsInPolygon(CurAlarmDeteObjInfo, CurAlarmDeteOneArea)
                    CurAlarmDeteAreaHumanInfo[i_area]['ObjInfo'] = CurAlarmDeteObjInfo[CurAlarmDeteAreaHumanFlag,:] # 当前区域目标信息
                    CurAlarmDeteAreaHumanInfo[i_area]['ObjNum'] = CurAlarmDeteObjInfo[CurAlarmDeteAreaHumanFlag,:].shape[0] # 当前区域目标个数
                else:
                    CurAlarmDeteAreaHumanInfo[i_area]['ObjInfo'] = []
                    CurAlarmDeteAreaHumanInfo[i_area]['ObjNum'] = 0
            CurFrameMultiSensorAlarmInfo[alarm_id]['DeteAreaHumanInfo'] = CurAlarmDeteAreaHumanInfo # 当前帧当前告警信息
            # 按仓室分区域
            CurAlarmDeteRoomHumanInfo = dict()
            if len(RoomSensorGroup) == 0 or len(MainAndSubRoom) == 0: # 检测告警不区分主/外仓
                CurAlarmDeteAreaHumanInfoSelect = dict() # 初始化当前检测区域目标信息
                for i_area in range(CurAlarmDeteArea.shape[0]):
                    CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.empty([0,3])
                    CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.empty([0,1])
                    CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = 0
                    CurAlarmDeteOneArea = CurAlarmDeteArea[i_area]
                    # 判断一个点是否在指定区域内
                    if len(CurAlarmDeteObjInfo) > 0:
                        CurAlarmDeteAreaHumanFlag = PointsInPolygon(CurAlarmDeteObjInfo, CurAlarmDeteOneArea)
                        CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjInfo'], CurAlarmDeteObjInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                        CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'], CurAlarmDeteObjSensorInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                        CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = CurAlarmDeteAreaHumanInfoSelect['ObjInfo'].shape[0] # 当前区域目标个数
                CurAlarmDeteRoomHumanInfo[0] = CurAlarmDeteAreaHumanInfoSelect # 当前帧当前告警信息
            else: # 检测告警区分主/外仓
                if alarm_id == AlarmIDIndex.Alarm_Alone: # 单人留仓
                    for i_room in range(len(MainAndSubRoom)):
    #                    print('i_room = ', i_room)
                        CurRoomArea = MainAndSubRoom[i_room] # 当前房间区域
                        CurAlarmDeteAreaHumanInfoSelect = dict() # 初始化当前检测区域目标信息
                        # 遍历当前房间各检测区域目标点
                        for i_area in range(CurAlarmDeteArea.shape[0]):
                            CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.empty([0,3])
                            CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.empty([0,1])
                            CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = 0
                            CurAlarmDeteOneArea = CurAlarmDeteArea[i_area]
                            # 判断一个点是否在指定区域内
                            if len(CurAlarmDeteObjInfo) > 0:
                                CurAlarmDeteAreaHumanFlag = PointsInPolygon(CurAlarmDeteObjInfo, CurAlarmDeteOneArea) and PointsInPolygon(CurAlarmDeteObjInfo, CurRoomArea)
                                CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjInfo'], CurAlarmDeteObjInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                                CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'], CurAlarmDeteObjSensorInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                                CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = CurAlarmDeteAreaHumanInfoSelect['ObjInfo'].shape[0] # 当前区域目标个数
                        # 再判断内外仓中间通道目标
                        if len(MainAndSubRoomEntrance) > 0: # 存在两个房间中间门洞通道
                            if CurAlarmDeteAreaHumanInfoSelect['ObjNum'] == 1: # 当前仓室人数为1
                                MainAndSubRoomEntranceHumanNum = 0
                                for i_room_entrance in MainAndSubRoomEntranceIndex: # 内外仓中间通道人数
                                    MainAndSubRoomEntranceHumanNum = MainAndSubRoomEntranceHumanNum + CurFrameMultiSensorAlarmInfo[alarm_id]['DeteAreaHumanInfo'][i_room_entrance]['ObjNum']
                                if MainAndSubRoomEntranceHumanNum > 0: # 如果内外仓中间通道存在人员，则不计算
                                    CurAlarmDeteAreaHumanInfoSelect = dict()
                        # 添加当前告警类型目标信息
                        CurAlarmDeteRoomHumanInfo[i_room] = CurAlarmDeteAreaHumanInfoSelect # 当前帧当前告警信息

                else: # 其他告警
                    for i_room in range(len(MainAndSubRoom)):
    #                    print('i_room = ', i_room)
                        CurRoomArea = MainAndSubRoom[i_room] # 当前房间区域
                        CurAlarmDeteAreaHumanInfoSelect = dict() # 初始化当前检测区域目标信息
                        # 遍历当前房间各检测区域目标点
                        for i_area in range(CurAlarmDeteArea.shape[0]):
                            CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.empty([0,3])
                            CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.empty([0,1])
                            CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = 0
                            CurAlarmDeteOneArea = CurAlarmDeteArea[i_area]
                            # 判断一个点是否在指定区域内
                            if len(CurAlarmDeteObjInfo) > 0:
                                if alarm_id == AlarmIDIndex.Alarm_InternalSupervisor: # 内部监管，不再判断是否只是划定的告警区域内（内部监管区域+厕所区域）
                                    CurAlarmDeteAreaHumanFlag = PointsInPolygon(CurAlarmDeteObjInfo, CurRoomArea) # 判断是否在指定房间内
                                else: # 其他告警，判断是否在划定的告警区域内 + 是否在指定的房间内
                                    CurAlarmDeteAreaHumanFlag = PointsInPolygon(CurAlarmDeteObjInfo, CurAlarmDeteOneArea) and PointsInPolygon(CurAlarmDeteObjInfo, CurRoomArea)
                                CurAlarmDeteAreaHumanInfoSelect['ObjInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjInfo'], CurAlarmDeteObjInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                                CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'] = np.concatenate((CurAlarmDeteAreaHumanInfoSelect['ObjSensorInfo'], CurAlarmDeteObjSensorInfo[CurAlarmDeteAreaHumanFlag,:])) # 当前区域目标信息
                                CurAlarmDeteAreaHumanInfoSelect['ObjNum'] = CurAlarmDeteAreaHumanInfoSelect['ObjInfo'].shape[0] # 当前区域目标个数
                        CurAlarmDeteRoomHumanInfo[i_room] = CurAlarmDeteAreaHumanInfoSelect # 当前帧当前告警信息
                        
                        
            CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'] = CurAlarmDeteRoomHumanInfo # 当前帧当前告警信息
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
                if len(CurAlarmDeteObjInfo) > 0:
                    lgmsg.debug('alarm_id = {}, CurAlarmDeteAreaHumanInfo = {}, '.format(alarm_id, CurAlarmDeteAreaHumanInfo))
                    lgmsg.debug('alarm_id = {}, CurAlarmDeteRoomHumanInfo = {}, '.format(alarm_id, CurAlarmDeteRoomHumanInfo))
                    

        # =============================================
        # 更新各类型告警
        # =============================================
        # 遍历各个告警，当前帧告警更新（按仓室分开）
        CurMultiFrameMultiSensorAlarmInfo = dict()
        for alarm_id in SensorInfo['ALARM_INFO']:
#            print('alarm_id = ', alarm_id)
            CurMultiFrameMultiSensorAlarmInfo[alarm_id] = dict()
            CurAlarmConfigInfo = SensorInfo['ALARM_INFO'][alarm_id]
            # 时间
            CurAlarmDetePeriodTime = CurAlarmConfigInfo['DetePeriodTime']
            CurAlarmDetePeriodTime = CalInValidTimeTransPeriodTime(np.array(CurAlarmDetePeriodTime), NearMintueDeteTIME)
            CurAlarmDetePeriodTimeFlag = CalInValidTime(WTime, CurAlarmDetePeriodTime, NearMintueDeteTIME) # 是否在检测时间内
            # 区域
            CurAlarmDeteArea = CurAlarmConfigInfo['DeteAreaPoints'] # [N x 8]
            # 持续帧数
            CurAlarmDeteContinuousFrame = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrame']
            CurAlarmDeteContinuousFrameUpRate = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrameUpRate']
            CurAlarmDeteContinuousFrameDnRate = AlarmIDExtraInfo[alarm_id]['DeteContinuousFrameDnRate']
            # 持续时间
            CurAlarmDeteContinuousTime = AlarmIDExtraInfo[alarm_id]['DeteContinuousTime']
            # 人数阈值
            CurAlarmDeteHumanNum = AlarmIDExtraInfo[alarm_id]['DeteHumanNum']
            # 前后帧数匹配
            CurAlarmValidObjNum = -1
            if len(CurFrameMultiSensorAlarmInfo[alarm_id]['DeteAreaHumanInfo']) > 0: # 该功能存在划定的检测区域
                PreAlarmDeteAreaHumanInfo = PreMultiFrameMultiSensorAlarmInfo[alarm_id] # 前一帧目标检测结果
                MatchDistThod = TwoFrameMatchDistThod # 前后帧目标匹配距离
                for i_area in range(len(CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'])):
                    # 按照主/外仓进行告警区分
                    #    第一类告警：根据各传感器给出的结果，以人员为告警坐标
                    #    第二类告警：根据各传感器给出的结果，以区域为告警坐标，包括：内部监管
                    CurAlarmDeteHumanNumFlag = False
                    if alarm_id == AlarmIDIndex.Alarm_InternalSupervisor: # 内部监管
#                        if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                            lgmsg.debug('alarm_id = Alarm_InternalSupervisor')
                        # 检测区域的中心位置
                        CurAlarmDeteAreaBbox = TransPointsToBbox(CurAlarmDeteArea[0]) # 默认选择第一个监管区域中心,[xmin, xmax, ymin, ymax]
                        CurAlarmDeteAreaBboxCenter = np.array([[(CurAlarmDeteAreaBbox[1]+CurAlarmDeteAreaBbox[0])/2, (CurAlarmDeteAreaBbox[3]+CurAlarmDeteAreaBbox[2])/2, 1.7]])
                        # 前后两帧对应的目标位置
                        # 前一帧obj 信息
                        PreFrameMultiSensorAlarmInfoAllIndex = []
                        PreFrameMultiSensorAlarmInfoAll = np.empty([0, 3])
                        for i_PreAlarmObj in PreAlarmDeteAreaHumanInfo:
                            if PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['RoomID'] == i_area:
                                PreFrameMultiSensorAlarmInfoAllIndex.append(i_PreAlarmObj)
                                PreFrameMultiSensorAlarmInfoAll = np.concatenate((PreFrameMultiSensorAlarmInfoAll, np.array([[PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjX'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjY'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjZ']]])))
                        # 当前帧obj 信息
                        CurFrameMultiSensorAlarmInfoAll = CurAlarmDeteAreaBboxCenter
                        CurFrameMultiSensorAlarmInfoAllSenorName = np.array([[int(10000+int(ConfigSensorNum))]]) # 默认使用最后一个传感器编号
                        # 告警人数阈值
                        if CurFrameMultiSensorAlarmInfo[AlarmIDIndex.Alarm_Alone]['RoomHumanInfo'][i_area]['ObjNum'] > 0: # 当前仓室有人
                            if CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjNum'] < CurAlarmDeteHumanNum:
                                CurAlarmDeteHumanNumFlag = True
                        else: # 当前仓室无人
                            CurAlarmDeteHumanNumFlag = False
                        # 如果存在传感器掉线，则全局告警无效
                        if len(LostDataSensorGroup) > 0:
                            CurAlarmDeteHumanNumFlag = False
                            
                    elif alarm_id == AlarmIDIndex.Alarm_Alone: # 单人留仓
#                        if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                            lgmsg.debug('alarm_id = Alarm_Alone')
                        # 前后两帧对应的目标位置
                        # 前一帧obj 信息
                        PreFrameMultiSensorAlarmInfoAllIndex = []
                        PreFrameMultiSensorAlarmInfoAll = np.empty([0, 3])
                        for i_PreAlarmObj in PreAlarmDeteAreaHumanInfo:
                            if PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['RoomID'] == i_area:
                                PreFrameMultiSensorAlarmInfoAllIndex.append(i_PreAlarmObj)
                                PreFrameMultiSensorAlarmInfoAll = np.concatenate((PreFrameMultiSensorAlarmInfoAll, np.array([[PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjX'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjY'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjZ']]])))
                        # 当前帧obj 信息
                        CurFrameMultiSensorAlarmInfoAll = CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjInfo']
                        CurFrameMultiSensorAlarmInfoAllSenorName = CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjSensorInfo']
                        # 告警人数阈值
                        if CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjNum'] <= CurAlarmDeteHumanNum:
                            CurAlarmDeteHumanNumFlag = True
                        # 如果存在传感器掉线，则全局告警无效
                        if len(LostDataSensorGroup) > 0:
                            CurAlarmDeteHumanNumFlag = False

                    else: # 其他告警
                        # 前后两帧对应的目标位置
                        # 前一帧obj 信息
                        PreFrameMultiSensorAlarmInfoAllIndex = []
                        PreFrameMultiSensorAlarmInfoAll = np.empty([0, 3])
                        for i_PreAlarmObj in PreAlarmDeteAreaHumanInfo:
                            if PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['RoomID'] == i_area:
                                PreFrameMultiSensorAlarmInfoAllIndex.append(i_PreAlarmObj)
                                PreFrameMultiSensorAlarmInfoAll = np.concatenate((PreFrameMultiSensorAlarmInfoAll, np.array([[PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjX'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjY'], PreAlarmDeteAreaHumanInfo[i_PreAlarmObj]['ObjZ']]])))
                        # 当前帧obj 信息
                        CurFrameMultiSensorAlarmInfoAll = CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjInfo']
                        CurFrameMultiSensorAlarmInfoAllSenorName = CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjSensorInfo']
                        # 告警人数阈值
                        if CurFrameMultiSensorAlarmInfo[alarm_id]['RoomHumanInfo'][i_area]['ObjNum'] >= CurAlarmDeteHumanNum:
                            CurAlarmDeteHumanNumFlag = True
                    
                    # 目标告警有效性，[人数 + 时间节点]
                    CurAlarmDeteValid = False
                    if CurAlarmDeteHumanNumFlag==True and CurAlarmDetePeriodTimeFlag==1:
                        CurAlarmDeteValid = True

                    # 前后帧匹配关系
                    PreFrameAlarmHumanMatch, CurFrameAlarmHumanMatch = TwoFrameObjMatch(PreFrameMultiSensorAlarmInfoAll, CurFrameMultiSensorAlarmInfoAll, MatchDistThod) # 前后两帧目标对应关系
                    
#                    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                        lgmsg.debug('ValidFlag = ', CurAlarmDeteHumanNumFlag, CurAlarmDetePeriodTimeFlag, CurAlarmDeteValid)
#                        lgmsg.debug('TwoFrameObjInfo = ', PreFrameMultiSensorAlarmInfoAll, CurFrameMultiSensorAlarmInfoAll)
#                        lgmsg.debug('ObjMatch = ', PreFrameAlarmHumanMatch, CurFrameAlarmHumanMatch)

                    # 匹配关系
                    #   前一帧未匹配到的目标，前一帧告警减累计值
                    if len(PreFrameAlarmHumanMatch)>0:
                        PreFrameAlarmHumanMatchMiss = [i for i in range(len(PreFrameAlarmHumanMatch)) if PreFrameAlarmHumanMatch[i]==-1]
                        for i_MatchMiss in PreFrameAlarmHumanMatchMiss:
                            # 更新告警信息
                            AlarmDete_CurFrmObjInfo = np.zeros([0,3])
                            AlarmDete_PreAlarmInfo = PreAlarmDeteAreaHumanInfo[PreFrameMultiSensorAlarmInfoAllIndex[i_MatchMiss]]
                            AlarmDete_PreAlarmInfo['RoomID'] = i_area
                            CurSensorCurFrmTime = MultiSensorWTimeGroup[str(int(AlarmDete_PreAlarmInfo['SensorID']))] # 当前传感器当前帧时间戳
                            CurOneAlarmIDInfo = SequenceFrmAlarmDete(AlarmDete_CurFrmObjInfo, AlarmDete_PreAlarmInfo, WTime, SensorFrmTime=CurSensorCurFrmTime, SeqFrmThod=CurAlarmDeteContinuousFrame, SeqTimeThod=CurAlarmDeteContinuousTime, SeqFrmMin=-1, SeqFrmMax=CurAlarmDeteContinuousFrame+5, SeqFrmUpRate=CurAlarmDeteContinuousFrameUpRate, SeqFrmDnRate=CurAlarmDeteContinuousFrameDnRate, CurFrmValidFlag=CurAlarmDeteValid)
                            if CurOneAlarmIDInfo['AlarmState'] >= -1: # 如果当前目标为不告警状态，则删除此条告警记录
                                CurAlarmValidObjNum = CurAlarmValidObjNum + 1 # 当前告警目标个数增1
                                CurMultiFrameMultiSensorAlarmInfo[alarm_id][CurAlarmValidObjNum] = CurOneAlarmIDInfo
                                
#                            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                                lgmsg.debug('CurOneAlarmIDInfo miss = ', PreFrameMultiSensorAlarmInfoAllIndex[i_MatchMiss], CurOneAlarmIDInfo)

                    #   当前帧未匹配到的目标，当前帧新增告警ID
                    if len(CurFrameAlarmHumanMatch)>0:
                        if CurAlarmDeteValid == True:
                            CurFrameAlarmHumanMatchMiss = [i for i in range(len(CurFrameAlarmHumanMatch)) if CurFrameAlarmHumanMatch[i]==-1]
                            for i_MatchMiss in CurFrameAlarmHumanMatchMiss:
                                GlobalDeteAlarmID = GlobalDeteAlarmID + 1 # 全局变量AlarmID个数增1
                                CurAlarmValidObjNum = CurAlarmValidObjNum + 1 # 当前告警目标个数增1
                                # 新增一个告警ID
                                AlarmDete_CurFrmObjInfo = CurFrameMultiSensorAlarmInfoAll[i_MatchMiss] # [x,y,z]
                                AlarmDete_PreAlarmInfo = np.zeros([0,3])
                                # PreFrameMultiSensorOneRoom
                                PreFrameMultiSensorOneRoom = dict()
                                PreFrameMultiSensorOneRoom['AlarmType'] = alarm_id
                                PreFrameMultiSensorOneRoom['AlarmState'] = 0 # 起始状态
                                PreFrameMultiSensorOneRoom['AlarmID'] = GlobalDeteAlarmID
                                PreFrameMultiSensorOneRoom['SensorID'] = CurFrameMultiSensorAlarmInfoAllSenorName[i_MatchMiss][0]
                                PreFrameMultiSensorOneRoom['ObjX'] = AlarmDete_CurFrmObjInfo[0]
                                PreFrameMultiSensorOneRoom['ObjY'] = AlarmDete_CurFrmObjInfo[1]
                                PreFrameMultiSensorOneRoom['ObjZ'] = AlarmDete_CurFrmObjInfo[2]
                                PreFrameMultiSensorOneRoom['ReservedInfo'] = -1
                                PreFrameMultiSensorOneRoom['ObjLabel'] = -1
                                PreFrameMultiSensorOneRoom['SumFrame'] = 0
                                PreFrameMultiSensorOneRoom['StartWorldTime'] = -1
                                PreFrameMultiSensorOneRoom['SumTime'] = -1
                                PreFrameMultiSensorOneRoom['RoomID'] = i_area
                                CurOneAlarmIDInfo = PreFrameMultiSensorOneRoom
                                CurMultiFrameMultiSensorAlarmInfo[alarm_id][CurAlarmValidObjNum] = CurOneAlarmIDInfo
                                
#                                if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                                    lgmsg.debug('CurFrameAlarmHumanMatchMiss = ', CurFrameAlarmHumanMatchMiss)
#                                    lgmsg.debug('CurOneAlarmIDInfo add = ', CurOneAlarmIDInfo)

                    #   当前帧匹配到的目标，更新告警信息
                    if len(CurFrameAlarmHumanMatch)>0:
                        CurFrameAlarmHumanMatchRight = [i for i in range(len(CurFrameAlarmHumanMatch)) if CurFrameAlarmHumanMatch[i]!=-1]
                        for i_MatchRight in CurFrameAlarmHumanMatchRight:
                            # 更新告警信息
                            AlarmDete_CurFrmObjInfo = CurFrameMultiSensorAlarmInfoAll[i_MatchRight]
                            AlarmDete_PreAlarmInfo = PreAlarmDeteAreaHumanInfo[PreFrameMultiSensorAlarmInfoAllIndex[CurFrameAlarmHumanMatch[i_MatchRight]]]
                            AlarmDete_PreAlarmInfo['RoomID'] = i_area
                            AlarmDete_PreAlarmInfo['SensorID'] = CurFrameMultiSensorAlarmInfoAllSenorName[i_MatchRight][0]
                            CurSensorCurFrmTime = MultiSensorWTimeGroup[str(int(AlarmDete_PreAlarmInfo['SensorID']))] # 当前传感器当前帧时间戳
                            CurOneAlarmIDInfo = SequenceFrmAlarmDete(AlarmDete_CurFrmObjInfo, AlarmDete_PreAlarmInfo, WTime, SensorFrmTime=CurSensorCurFrmTime, SeqFrmThod=CurAlarmDeteContinuousFrame, SeqTimeThod=CurAlarmDeteContinuousTime, SeqFrmMin=-1, SeqFrmMax=CurAlarmDeteContinuousFrame+5, SeqFrmUpRate=CurAlarmDeteContinuousFrameUpRate, SeqFrmDnRate=CurAlarmDeteContinuousFrameDnRate, CurFrmValidFlag=CurAlarmDeteValid)
                            if CurOneAlarmIDInfo['AlarmState'] >= -1: # 如果当前目标为不告警状态，则删除此条告警记录
                                CurAlarmValidObjNum = CurAlarmValidObjNum + 1 # 当前告警目标个数增1
                                CurMultiFrameMultiSensorAlarmInfo[alarm_id][CurAlarmValidObjNum] = CurOneAlarmIDInfo
                                
#                            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 单人留仓各仓人数/告警位置
#                                lgmsg.debug('CurOneAlarmIDInfo right = ', i_MatchRight, CurOneAlarmIDInfo)
                
            else:
                if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                    lgmsg.debug('{} alarm have not DeteArea.'.format(alarm_id))
            # 打印信息
            if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                if len(CurMultiFrameMultiSensorAlarmInfo[alarm_id]) > 0:
                    lgmsg.debug('{} alarm obj info is {}'.format(alarm_id, CurMultiFrameMultiSensorAlarmInfo[alarm_id]))

        # 生成告警结果
        CurAllAlarmDeteInfo = []
        for alarm_id in SensorInfo['ALARM_INFO']:
            CurMultiFrameMultiSensorAlarmInfoOne = CurMultiFrameMultiSensorAlarmInfo[alarm_id]
            CurAlarmObjNum = len(CurMultiFrameMultiSensorAlarmInfoOne)
            for i_obj in range(CurAlarmObjNum):
                CurMultiFrameMultiSensorAlarmInfoOneObj = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                CurAlarmObjListInfo = []
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['AlarmType'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['AlarmState'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['AlarmID'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['SensorID'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['ObjX'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['ObjY'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['ObjZ'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['ReservedInfo'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['ObjLabel'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['SumFrame'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['StartWorldTime'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['SumTime'])
                CurAlarmObjListInfo.append(CurMultiFrameMultiSensorAlarmInfoOneObj['RoomID'])
                CurAllAlarmDeteInfo.append(CurAlarmObjListInfo)
        CurAllAlarmDeteInfo = np.array(CurAllAlarmDeteInfo)
        
        return CurAllAlarmDeteInfo, CurFrameMultiSensorAlarmInfo, CurMultiFrameMultiSensorAlarmInfo
        
        
    def read_human_info_str(self):
        """
        Read human information from input string.
        Inputs:
            HumanInfoStr: string format
                          {RoomID, [SensorID: (WarningType, Human1_pose, Human1_x, Human1_y, Human1_z),(WarningType, Human2_pose, Human2_x, Human2_y, Human2_z),...]}
                    注：WarningType = 1 表示 Humaninfo 信息； WarningType == 2 标志 INTERNALSUPERVISOR 信息
        Outputs:
            human_info: numeric format
                        dict [ID:[[pose,x,y,z],[pose,x,y,z]]
        """
        # number of human object features, eg:[WarningType, pose, x, y, z]
        HumanFeatureNum = 5
        # current room info
        CurRoomInfoGroup = dict()
        # room id
        RoomId = 0
        # sensor input havenot data
        LostDataSensorGroup = []
        # multi sensor current frame world time
        MultiSensorFrmWTime = dict()
        for i_sensor_name in self.SenorNameGroupName:
            MultiSensorFrmWTime[i_sensor_name] = -1
        
        # init human info
        human_info = dict() # 初始化各传感器返回的告警结果信息
        SensorInfo = self.SensorInfo
        for alarm_id in SensorInfo['ALARM_INFO']:
            human_info[alarm_id] = dict()
        
        # read input room-id info
        RoomIdStrSplit = self.HumanInfoStr.split('{')
        for RoomIdInfo in RoomIdStrSplit: # sensor series
            if len(RoomIdInfo)>0:
                RoomIdInfoSplit = RoomIdInfo.split(',')
                if not len(RoomIdInfoSplit) == 1: # 如果存在  ','
                    RoomId = RoomIdInfoSplit[0] # one room ID
                else: # 如果不存在  ','， 如：'{1002}'
                    RoomIdInfoSplit = RoomIdInfo.split('}')
                    RoomId = RoomIdInfoSplit[0]
         
        # read input human info string
        HumanInfoStrSplit = self.HumanInfoStr.split('[')
        for SensorHumanInfo in HumanInfoStrSplit: # sensor series
            if len(SensorHumanInfo)>0 and SensorHumanInfo!=HumanInfoStrSplit[0]: # HumanInfoStrSplit index from 1 ~ N
                SensorHumanInfoSplit = SensorHumanInfo.split('(')
                for OneHumanInfo in SensorHumanInfoSplit: # human series
                    OneHumanInfoSplit1 = OneHumanInfo.split(')')
                    OneHumanInfoSplit2 = OneHumanInfo.split(',')
                    
                    if len(OneHumanInfoSplit1)==1: # sensor id
                        # 获取当前传感器 ID 和 时间戳
                        if len(OneHumanInfoSplit2) == 2:
                            OneSensorID = OneHumanInfoSplit2[0] # temp sensor ID
                            OneSensorTime = -1 # temp sensor time
                        else:
                            OneSensorID = OneHumanInfoSplit2[0] # temp sensor ID
                            OneSensorTime = float(OneHumanInfoSplit2[1]) # temp sensor time
                        # 初始化当前传感器各类告警
                        for i_alarm in human_info:
                            human_info[i_alarm][OneSensorID] = []
                        # 获取传感器时间戳
                        MultiSensorFrmWTime[OneSensorID] = OneSensorTime
                    else: # human info
                        for OneHumanInfoSplit1_s in OneHumanInfoSplit1[0].split(','):
                            OneHumanInfoSplit1_s = float(OneHumanInfoSplit1_s)
                            # 获取各传感器各类告警检测结果
                            CurAlarmID = float(OneHumanInfoSplit1[0].split(',')[0])
                            if CurAlarmID in human_info.keys():
                                human_info[CurAlarmID][OneSensorID].append(OneHumanInfoSplit1_s)
                            else:
                                if CurAlarmID == -1: # 传感器掉线
                                    if OneSensorID not in LostDataSensorGroup:
                                        LostDataSensorGroup.append(OneSensorID)
#                                    print('{} input Type = {}.'.format(OneSensorID, CurAlarmID))
                                else:
                                    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0: # 融合后单人留仓人员信息
                                        lgmsg.debug('Input info WarningType = {} error.'.format(CurAlarmID))
        # reshape human info to [N x HumanFeatureNum]
        for i_alarm in human_info:
            for i_sensor in human_info[i_alarm]:
                human_info[i_alarm][i_sensor] = np.reshape(human_info[i_alarm][i_sensor],[int(len(human_info[i_alarm][i_sensor])/HumanFeatureNum),HumanFeatureNum])[:,1:] 
        
        # CurRoomInfoGroup
        CurRoomInfoGroup['RoomId'] = RoomId
        CurRoomInfoGroup['HumanInfo'] = human_info
        CurRoomInfoGroup['SensorLostGroup'] = LostDataSensorGroup
        CurRoomInfoGroup['SensorTime'] = MultiSensorFrmWTime
                
        return CurRoomInfoGroup
        
    def alarm_detect_info(DetectInfoNumeric, DMode):
        """
        Generate detect information from global human information, retrun alram result.
        Inputs:
            DetectInfo: numeric format
                          dirct [ID:[[pose,x,y,z],[pose,x,y,z]]
            DMode: R/W
            
        Outputs:
            detect_info: string format
                        [AlramID: (Human1_x, Human1_y, Human1_z),(Human2_x, Human2_y, Human2_z),...]
                        eg: [10101,(1,1,2,3)],[10102,(1,1,1,1),(1,2,2,2),(1,3,3,3)]
        """
        
        if DMode == 'R': # string to numeric
            print('read DetectInfoNumeric')
        
        elif DMode == 'W': # numeric to string
            # init detect info
            detect_info = ''
            
            # numeric to string
            DetectInfo = DetectInfoNumeric
            i_alarm_num = 0
            for i_alarm in DetectInfo:
                i_alarm_num = i_alarm_num + 1
                if len(DetectInfo[i_alarm])==0:
                    detect_info = detect_info + '[' + i_alarm + ']'
                else:
                    TempOneAlarmInfo = ''
                    try:
#                        TempNumLine = DetectInfo[i_alarm].shape[1] # 二维数据
                        for i_alinfo in range(len(DetectInfo[i_alarm])):
                            TempOneAlarmOneObjInfo = ''
                            for i_feature in range(len(DetectInfo[i_alarm][i_alinfo])):
                                if i_feature<len(DetectInfo[i_alarm][i_alinfo])-1:
                                    TempOneAlarmOneObjInfo = TempOneAlarmOneObjInfo + str(DetectInfo[i_alarm][i_alinfo][i_feature]) + ','
                                else:
                                    TempOneAlarmOneObjInfo = TempOneAlarmOneObjInfo + str(DetectInfo[i_alarm][i_alinfo][i_feature])
                            TempOneAlarmInfo = TempOneAlarmInfo + ',(' + TempOneAlarmOneObjInfo + ')'
                    except: # 一维数据
                            TempOneAlarmOneObjInfo = ''                        
                            for i_feature in range(len(DetectInfo[i_alarm])):
                                if i_feature<len(DetectInfo[i_alarm])-1:
                                    TempOneAlarmOneObjInfo = TempOneAlarmOneObjInfo + str(DetectInfo[i_alarm][i_feature]) + ','
                                else:
                                    TempOneAlarmOneObjInfo = TempOneAlarmOneObjInfo + str(DetectInfo[i_alarm][i_feature])
                            TempOneAlarmInfo = TempOneAlarmInfo + ',(' + TempOneAlarmOneObjInfo + ')'
                # 一个目标信息        
                if i_alarm_num<len(DetectInfo):
                    detect_info = detect_info + '[' + i_alarm + TempOneAlarmInfo + '],'
                else:
                    detect_info = detect_info + '[' + i_alarm + TempOneAlarmInfo + ']'
            result = detect_info
                    
        return result
        
    def trans_alarm_info(CurDeteState, DMode, AlarmFeatureNum = 13):
        """
        Generate alram information and human information, string result
        Inputs:
            CurDeteState: detect info
                          单人留仓告警, [告警类型，房间编号，告警次数，告警位置X，告警位置Y，告警位置Z，传感器编号，告警状态]
        Outputs:
            DMode == 'R' : string -> numeric
            DMode == 'W' : numeric -> string
            
        """
        
        # 读写检测信息
        if DMode == 'R': # string to numeric
            # 初始化Numeric 结果
            NumeTemp =[]
            # 读取文件
            lines = CurDeteState.split('\n')
            for line in lines:
                lineSp = line.split()
                for lineSpT in lineSp:
                    NumeTemp.append(float(lineSpT))
            # 转换格式
            try:
                NumeTemp = np.reshape(NumeTemp,[int(len(NumeTemp)/GlobalDeteAlarmFeatureNum),GlobalDeteAlarmFeatureNum]) # [告警类型，房间编号，告警次数，告警位置X，告警位置Y，告警位置Z，传感器编号，告警状态]
                result = NumeTemp
            except:
                result = [] # 如果输入数据不规范，则返回空数据
        
        elif DMode == 'W': # numeric to string
            # 初始化String 结果
            StrTemp = ''
            # 告警信息
            for n_line in range(CurDeteState.shape[0]): # 告警信息行数
                StrTemp = StrTemp + str(CurDeteState[n_line,0]) + ' ' + str(CurDeteState[n_line,1]) + ' ' + str(CurDeteState[n_line,2]) + ' ' + str(CurDeteState[n_line,3]) + ' ' \
                                                + str(round(CurDeteState[n_line,4],3)) + ' ' + str(round(CurDeteState[n_line,5],3)) + ' ' + str(round(CurDeteState[n_line,6],3)) + ' ' \
                                                + str(CurDeteState[n_line,7]) + ' ' + str(CurDeteState[n_line,8]) + ' ' + str(CurDeteState[n_line,9]) + ' ' + str(CurDeteState[n_line,10]) + ' ' \
                                                + str(CurDeteState[n_line,11]) + ' ' + str(CurDeteState[n_line,12]) \
                                                +'\n'
            result = StrTemp

        return result
        
    def fuse_multi_sensor_dete_info(CurHumanInfo, MultiSensorDeteArea, MultiSensorEdgeArea, AddEdgeDist = 0.2, MultiObjsDisThod = 0.25, AlarmArea = None):
        """
        功能：融合多个传感器检测的信息，先对各传感器区域分割，再去除边界重合的目标点
        输入：各传感器检测到的目标信息：CurHumanInfo
             各传感器检测的区域信息：MultiSensorDeteArea
             各传感器检测的区域之间的边界区域：MultiSensorEdgeArea
             当前告警类型所在区域：AlarmArea
        输出：融合后的目标信息：MultiSensorDeteInfoSelect
        
        方法：对边界范围内目标使用聚类方法合并数据，对非边界目标按各传感器检测范围选择
        
        """
        # 区域边界区域
        TempDist = AddEdgeDist # 再增加边界 0.15m
        # 多目标之间的邻近距离阈值
        ObjDisThod = MultiObjsDisThod # init:0.25

        # select useful multi detect objects from limit area
        MultiSensorDeteInfo = np.empty([0, 3]) # 整体目标信息
        MultiSensorEdgeDeteInfo = [] # 边界区域的目标
        MultiSensorNotEdgeDeteInfo = [] # 非边界区域的目标
        for sensor_dete in CurHumanInfo: # 按传感器个数遍历目标点
            TempSensorDeteInfoSrc = CurHumanInfo[sensor_dete] # 当前传感器对应的目标点

            # select useful area
            for i_obj in range(TempSensorDeteInfoSrc.shape[0]): # 遍历当前目标点
                # 判断是否在当前传感器检测范围内
                TempSensorDeteArea = MultiSensorDeteArea[sensor_dete] # MultiSensorDeteArea =  dict()
                # 求检测区域边界，【暂时使用最大矩形边界】
                CurOneArea_bbox = TransPointsToBbox(TempSensorDeteArea)
                TempSensorDeteArea = CalAboveHeightValidArea(CurOneArea_bbox, MultiSensorEdgeArea, TempDist) # 
                TempSensorDeteArea = TransBboxToPoints(TempSensorDeteArea)
                
                # 多边形区域内
                TempSensorDeteArea = np.array(TempSensorDeteArea)
                CurrPoly = TempSensorDeteArea.reshape([int(len(TempSensorDeteArea)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                pCurrPoly = path.Path(CurrPoly)
                TempData = np.array([[0.0, 0.0]])
                TempData[0,0] = TempSensorDeteInfoSrc[i_obj,1]
                TempData[0,1] = TempSensorDeteInfoSrc[i_obj,2]
                binAreaAll = pCurrPoly.contains_points(TempData) # limit xy, [2 x N]
                
                # 是否在多边形内
                if binAreaAll[0]:
#                    MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,1])
#                    MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,2])
#                    MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,3])  
                    
                    # 判断当前点是否在边界上，在边界区域内为1，否则为-1
                    TempObjValidFlag = -1
                    for i_area in range(MultiSensorEdgeArea.shape[0]): # 按照区域
                        TempAreaSrc = MultiSensorEdgeArea[i_area]
#                        TempArea = [TempAreaSrc[0]-TempDist, TempAreaSrc[1]+TempDist, TempAreaSrc[2]-TempDist, TempAreaSrc[3]+TempDist]
                        TempArea = [TempAreaSrc[0]-TempDist-0.1, TempAreaSrc[1]+TempDist+0.1, TempAreaSrc[2]-TempDist-0.1, TempAreaSrc[3]+TempDist+0.1] # 边界区域与非边界区域目标重合情况

                        if TempData[0,0]>TempArea[0] and TempData[0,0]<TempArea[1] and TempData[0,1]>TempArea[2] and TempData[0,1]<TempArea[3]: # 选择区域内
                            TempObjValidFlag = 1
                    # 判断目标点是否在边界区域
                    if TempObjValidFlag == -1: # 非边界
                        MultiSensorNotEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,1])
                        MultiSensorNotEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,2])
                        MultiSensorNotEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,3])
                    else: # 边界区域
                        MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,1])
                        MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,2])
                        MultiSensorEdgeDeteInfo.append(TempSensorDeteInfoSrc[i_obj,3])   
                        
        # 转换格式
        if len(MultiSensorEdgeDeteInfo) > 0:
            MultiSensorEdgeDeteInfo = np.reshape(MultiSensorEdgeDeteInfo,[int(len(MultiSensorEdgeDeteInfo)/3),3])
        else:
            MultiSensorEdgeDeteInfo = np.empty([0, 3])
        if len(MultiSensorNotEdgeDeteInfo) > 0:
            MultiSensorNotEdgeDeteInfo = np.reshape(MultiSensorNotEdgeDeteInfo,[int(len(MultiSensorNotEdgeDeteInfo)/3),3])
        else:
            MultiSensorNotEdgeDeteInfo = np.empty([0, 3])

        # 结合 边界区域的目标 和 非边界区域的目标
        CombineEdgeObjMethod = 2
        if CombineEdgeObjMethod == 1: # 分开计算
            # 通过聚类计算目标个数
            MultiSensorDeteInfo_XY = MultiSensorEdgeDeteInfo[:,:-1]
            MultiSensorDeteInfoSelect = []
            if MultiSensorDeteInfo_XY.shape[0]>0:
                CurCluInfo = skc.DBSCAN(eps=ObjDisThod, min_samples=1).fit(MultiSensorDeteInfo_XY)
                CurCluInfoLabel = CurCluInfo.labels_
                CurCluInfoNumClu = len(set(CurCluInfoLabel)) - (1 if -1 in CurCluInfoLabel else 0)
                for i_clu in range(CurCluInfoNumClu):
                    OneCluPts = MultiSensorEdgeDeteInfo[CurCluInfoLabel==i_clu]
                    OneCluPtsCenter = np.mean(OneCluPts,0)
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[0]) # 保存中心位置
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[1]) # 
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[2]) # 
                MultiSensorDeteInfoSelect = np.reshape(MultiSensorDeteInfoSelect,[int(len(MultiSensorDeteInfoSelect)/3),3])
            else:
                MultiSensorDeteInfoSelect = np.empty([0, 3]) 
            # 非边界区域的目标 + 融合后的边界区域目标, [N x 3]
            MultiSensorDeteInfo = np.concatenate((MultiSensorNotEdgeDeteInfo, MultiSensorDeteInfoSelect))
        else: # 合并计算
            MultiSensorEdgeDeteInfo = np.concatenate((MultiSensorNotEdgeDeteInfo, MultiSensorEdgeDeteInfo))
            # 通过聚类计算目标个数
            MultiSensorDeteInfo_XY = MultiSensorEdgeDeteInfo[:,:-1]
            MultiSensorDeteInfoSelect = []
            if MultiSensorDeteInfo_XY.shape[0]>0:
                CurCluInfo = skc.DBSCAN(eps=ObjDisThod, min_samples=1).fit(MultiSensorDeteInfo_XY)
                CurCluInfoLabel = CurCluInfo.labels_
                CurCluInfoNumClu = len(set(CurCluInfoLabel)) - (1 if -1 in CurCluInfoLabel else 0)
                for i_clu in range(CurCluInfoNumClu):
                    OneCluPts = MultiSensorEdgeDeteInfo[CurCluInfoLabel==i_clu]
                    OneCluPtsCenter = np.mean(OneCluPts,0)
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[0]) # 保存中心位置
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[1]) # 
                    MultiSensorDeteInfoSelect.append(OneCluPtsCenter[2]) # 
                MultiSensorDeteInfoSelect = np.reshape(MultiSensorDeteInfoSelect,[int(len(MultiSensorDeteInfoSelect)/3),3])
            else:
                MultiSensorDeteInfoSelect = np.empty([0, 3]) 
            # 非边界区域的目标 + 融合后的边界区域目标, [N x 3]
            MultiSensorDeteInfo = MultiSensorDeteInfoSelect

        return MultiSensorDeteInfo
        
    def find_obj_sensor_info(CurAlarmDeteObjInfo, MultiSensorDeteArea, MultiSensorEdgeArea):
        """
        功能：找到各目标点所处传感器编号
        """
        CurAlarmDeteObjSensorInfo = 10001*np.ones([CurAlarmDeteObjInfo.shape[0],1])
        for i_obj in range(CurAlarmDeteObjInfo.shape[0]):
            for i_area in MultiSensorDeteArea:
                TempSensorDeteArea = MultiSensorDeteArea[i_area]
                # 判断目标点是否在传感器检测范围内
                TempSensorDeteArea = np.array(TempSensorDeteArea) # 多边形区域内
                CurrPoly = TempSensorDeteArea.reshape([int(len(TempSensorDeteArea)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                pCurrPoly = path.Path(CurrPoly)
                TempData = np.array([[0.0, 0.0]])
                TempData[0,0] = CurAlarmDeteObjInfo[i_obj,0]
                TempData[0,1] = CurAlarmDeteObjInfo[i_obj,1]
                binAreaAll = pCurrPoly.contains_points(TempData) # limit xy, [2 x N]
                if binAreaAll:
                    CurAlarmDeteObjSensorInfo[i_obj,0] = i_area
        return CurAlarmDeteObjSensorInfo


    