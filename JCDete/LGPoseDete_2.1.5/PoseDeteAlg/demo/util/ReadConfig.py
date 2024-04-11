# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:55:45 2020

@author: HYD
"""

import sys
import numpy as np
import time 
from collections import defaultdict

try:
    import configparser
except:
    from six.moves import configparser
    
from detect_alarm import AlarmInfoStruct
from .PolygonFuns import TransBboxToPoints
    
def ReadIniConfig(ConfigFileName):
    """
    功能：读取配置文件
    输入：
        ConfigFileName：.ini 文件名
            其中：#告警id	"warningTypeModel":[
                    [1,"未在指定时间休息"],
                    [2,"未在制定区域监督"],
                    [4,"厕所区域异常"],
                    [8,"窗户区域异常"],
                    [16,"高度异常"],
                    [32,"非休息时间休息"],
                    [64,"进入三角区域"],
                    [128,"内务不整"],
                    [512,"单人留仓"],
                    [1024,"吊拉窗户"],
                    [2048,"搭人梯"],
                    [4096,"站在被子上做板报"]
                    ]

    输入：
        ConfigInfo
    """
    
    # 读取配置文件
    config = configparser.ConfigParser()
    fp = open(ConfigFileName, encoding= 'utf-8') # utf-8
    config.read_file(fp) 
    sections = config.sections()
    
    # RoomInfo info
    RoomInfo = dict()
    RoomInfo['ALARM_INFO'] = dict()

    for sect in sections:
        TempSect = sect
        
        # ROOM_INFO
        if TempSect == 'ROOM_INFO':
            CurSectItemInfo = dict()
            for item in config.items(TempSect):
                CurSectItemInfo[item[0]] = item[1]
            RoomInfo[TempSect] = CurSectItemInfo
            
            # 如果没有‘managelevel’，则设置默认值：2
            NewVariableName = 'managelevel'
            NewVariableInit = 2
            if NewVariableName not in RoomInfo[TempSect].keys():
                RoomInfo[TempSect][NewVariableName] = NewVariableInit
                
        # SENSORS_INFO
        if TempSect == 'SENSORS_INFO':
            CurSectItemInfo = dict()
            CurSENSORS_INFO = dict()
            for item in config.items(TempSect):
                CurSectItemInfo[item[0]] = item[1]
            # currsensorid, serverPack, Area
            AllSensorId = []
            for key, value in CurSectItemInfo.items():
                # serverPack
                if key.find('serverpack')>-1:
                    AllSensorId.append(key.split('_')[1])
            CurSENSORS_INFO['currsensorid'] = CurSectItemInfo['currsensorid']
            for sensor_id in AllSensorId:
                CurSENSORS_INFO[sensor_id] = dict()
                # H
                CurSENSORS_INFO_H = []
                for H_i in CurSectItemInfo['serverpack_' + sensor_id].split(' '):
                    if len(H_i)>0:
                        CurSENSORS_INFO_H.append(float(H_i))
                CurSENSORS_INFO_H = np.reshape(np.array(CurSENSORS_INFO_H), [4,4])
                CurSENSORS_INFO[sensor_id]['H'] = CurSENSORS_INFO_H 
                # Area
                CurSENSORS_INFO[sensor_id]['Area'] = []
                if int(CurSectItemInfo['sensor_' + sensor_id + '_pointnum']) == 2: # 区域格式：[x1,y1,x2,y2], 【单个传感器是否会出现多个检测区域？】
                    CurSENSORS_INFO_Area = []
                    CurSENSORS_INFO_Area.append(float(CurSectItemInfo['sensor_' + sensor_id + '_area_x1']))
                    CurSENSORS_INFO_Area.append(float(CurSectItemInfo['sensor_' + sensor_id + '_area_y1']))
                    CurSENSORS_INFO_Area.append(float(CurSectItemInfo['sensor_' + sensor_id + '_area_x2']))
                    CurSENSORS_INFO_Area.append(float(CurSectItemInfo['sensor_' + sensor_id + '_area_y2']))
                    CurSENSORS_INFO[sensor_id]['Area'].append(CurSENSORS_INFO_Area)
            RoomInfo[TempSect] = CurSENSORS_INFO
            
        # ALARM_INFO
        if TempSect.find('ALARM_INFO')>-1:
            CurSectItemInfo = dict()
            CurALARM_INFO = dict()
            for item in config.items(TempSect):
                CurSectItemInfo[item[0]] = item[1]
            Cur_alarmtypeid = int(CurSectItemInfo['alarmtypeid'])
            # 此告警类型涉及的传感器
            CurALARM_INFO['DeteSensor'] = []
            for sensor_id in range(int(CurSectItemInfo['sensornum'])):
                 CurALARM_INFO['DeteSensor'].append(CurSectItemInfo['sensorid' + str(sensor_id+1)]) # str:['10002','10003']
            # 区域信息
            CurALARM_INFO['DeteArea'] = []
            for area_idx in range(int(CurSectItemInfo['areanum'])):
                CurIdxArea = [] # 区域格式：[x1,y1,x2,y2]
                CurIdxArea.append(float(CurSectItemInfo['area' + str(area_idx+1) + '_x1']))
                CurIdxArea.append(float(CurSectItemInfo['area' + str(area_idx+1) + '_y1']))
                CurIdxArea.append(float(CurSectItemInfo['area' + str(area_idx+1) + '_x2']))
                CurIdxArea.append(float(CurSectItemInfo['area' + str(area_idx+1) + '_y2']))
                CurALARM_INFO['DeteArea'].append(CurIdxArea)
            # 检测时间段
            CurALARM_INFO['DetePeriodTime'] = []
            for time_idx in range(int(CurSectItemInfo['checktimenum'])):
                CurIdxTime = [] # 区域格式：[h1,m1,h2,m2]
                CurIdxTime.append(int(CurSectItemInfo['time' + str(time_idx+1) + '_start_h']))
                CurIdxTime.append(int(CurSectItemInfo['time' + str(time_idx+1) + '_start_m']))
                CurIdxTime.append(int(CurSectItemInfo['time' + str(time_idx+1) + '_end_h']))
                CurIdxTime.append(int(CurSectItemInfo['time' + str(time_idx+1) + '_end_m']))
                CurALARM_INFO['DetePeriodTime'].append(CurIdxTime)
            # 持续检测时间:[min]
            CurALARM_INFO['DeteContinuousTime'] = float(CurSectItemInfo['triggerduration'])
            # 高度阈值
            CurALARM_INFO['DeteMaxHeight'] = float(CurSectItemInfo['maximumheight'])
            RoomInfo['ALARM_INFO'][Cur_alarmtypeid] = CurALARM_INFO
            
        # OTHER
        if TempSect.find('OTHER')>-1:
            CurSectItemInfo = dict()
            CurOTHER_INFO = dict()
            for item in config.items(TempSect):
                CurSectItemInfo[item[0]] = item[1]
            # personthreshold 人数阈值 (int)
            CurOTHER_INFO['personthreshold'] = int(CurSectItemInfo['personthreshold'])
#            # 图像是否上下翻转
#            CurOTHER_INFO['imageflipudflag'] = int(CurSectItemInfo['imageflipudflag'])
            RoomInfo['OTHER'] = CurOTHER_INFO
            
    fp.close() 
           
    return RoomInfo
    
def TransformConfigInfo(ConfigInfo):
    """
    功能：转换 ConfigInfo 信息
        [转换为以前使用的读取后数据格式]
    """
    # 原始的 ini 配置文件信息
    SrcConfigInfo = ConfigInfo 
    
    # 以前的的配置文件信息格式，SensorInfo
    SensorInfo_Output = AlarmInfoStruct.SensorInfo()
    # isFlip
    SensorInfo_Output.isFlip = np.array([0]) # 图像数据是否翻转，暂时设置为：0
#    SensorInfo_Output.isFlip = np.array([SrcConfigInfo['OTHER']['imageflipudflag']]) # 图像数据是否翻转，暂时设置为：0
    
    # CurSensorId
    CurSensorId = SrcConfigInfo['SENSORS_INFO']['currsensorid']
    # H
    SensorInfo_Output.H = SrcConfigInfo['SENSORS_INFO'][CurSensorId]['H']
    # BED, [1,"未在指定时间休息"]
    CurAlarmID = 1
    BEDMinHeight = 0.0 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            BED = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                BED.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], BEDMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
        else:
            BED = [[0,0,0,0,0,0,0]]
    else:
        BED = [[0,0,0,0,0,0,0]]
    SensorInfo_Output.BED = np.array(BED)
    # INTERNALSUPERVISOR, [2,"未在制定区域监督"]
    CurAlarmID = 2
    INTERNALSUPERVISORMinHeight = 1.25 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            INTERNALSUPERVISOR = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                # 将两点信息转换位4点信息
                TempPts = TransBboxToPoints(np.array([TempBbox])[:,[0,2,1,3]][0]) # [xmin, xmax, ymin, ymax]
                INTERNALSUPERVISOR.append([TempPts[0], TempPts[1], TempPts[2], TempPts[3], TempPts[4], TempPts[5], TempPts[6], TempPts[7], INTERNALSUPERVISORMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [x1, y1, x2, y2, x3, y3, x4, y4, zmin, zmax, ValidArea]  
        else:
            INTERNALSUPERVISOR = [[0,0,0,0,0,0,0,0,0,0,0]]
    else:
        INTERNALSUPERVISOR = [[0,0,0,0,0,0,0,0,0,0,0]]
    SensorInfo_Output.INTERNALSUPERVISOR = np.array(INTERNALSUPERVISOR)
    # TOILET, [4,"厕所区域异常"]
    CurAlarmID = 4
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            TOILET = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                TOILET.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], 1]) # [xmin, xmax, ymin, ymax, ValidArea]
        else:
            TOILET = [[0,0,0,0,0]]
    else:
        TOILET = [[0,0,0,0,0]]
    SensorInfo_Output.TOILET = np.array(TOILET)
    # WINDOW, [8,"窗户区域异常"]
    CurAlarmID = 8
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            WINDOW = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                WINDOW.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], 1]) # [xmin, xmax, ymin, ymax, ValidArea]
        else:
            WINDOW = [[0,0,0,0,0]]
    else:
        WINDOW = [[0,0,0,0,0]]
    SensorInfo_Output.WINDOW = np.array(WINDOW)
    
    # SLEEPTIME, [2,"未在制定区域监督"]/[1,"未在指定时间休息"]
    CurAlarmID = 2 # 可能某些功能不存在 ‘未在指定时间休息’
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        SLEEPTIME = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DetePeriodTime']
    else:
        SLEEPTIME = [[0, 0, 0, 0], [0, 0, 0, 0]]
    SensorInfo_Output.SLEEPTIME = np.array(SLEEPTIME)
    
    # ABOVEHEIGHTTHOD, [16,"高度异常"]
    CurAlarmID = 16
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        ABOVEHEIGHTTHOD = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight']
    else:
        ABOVEHEIGHTTHOD = 3.0
    SensorInfo_Output.ABOVEHEIGHTTHOD = np.array([ABOVEHEIGHTTHOD])
    
    # ABOVEHEIGHTTHOD, [16,"高度异常"]
    CurAlarmID = 16
    ABOVEHEIGHTMinHeight = 1.8 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            ABOVEHEIGHT = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                ABOVEHEIGHT.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], ABOVEHEIGHTMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
        else:
            ABOVEHEIGHT = [[0,0,0,0,0,0,0]]
    else:
        ABOVEHEIGHT = [[0,0,0,0,0,0,0]]
    SensorInfo_Output.ABOVEHEIGHT = np.array(ABOVEHEIGHT)
    
    
    
    
    # INTRIANGLEREGION, [64,"进入三角区域"]
    CurAlarmID = 64
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            INTRIANGLEREGION = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                INTRIANGLEREGION.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], 0,0,0,0, 0,SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'],1]) # [..., ValidArea]
        else:
            INTRIANGLEREGION = [[0,0,0,0,0,0,0,0,0,0,0]]
    else:
        INTRIANGLEREGION = [[0,0,0,0,0,0,0,0,0,0,0]]
    SensorInfo_Output.INTRIANGLEREGION = np.array(INTRIANGLEREGION)
    
    # HOUSEKEEP, [128,"内务不整"]
    CurAlarmID = 128
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            HOUSEKEEP = defaultdict(list)
            HOUSEKEEP_Area = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']: # Area
                HOUSEKEEP_Area.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], 0, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, area_direction]
            HOUSEKEEP[0] = np.array(HOUSEKEEP_Area) 
            DetePeriodTime = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DetePeriodTime'] # Time
            HOUSEKEEP_Time = []
            for TempTime in DetePeriodTime:
                HOUSEKEEP_Time.append([TempTime[0], TempTime[1], TempTime[2], TempTime[3]])
            HOUSEKEEP[1] = np.array(HOUSEKEEP_Time) 
        else:
            HOUSEKEEP = defaultdict(list)
            HOUSEKEEP[0] = np.array([[0,0,0,0,0,0,0]])
            HOUSEKEEP[1] = np.array([[0,0,0,0]])
    else:
        HOUSEKEEP = defaultdict(list)
        HOUSEKEEP[0] = np.array([[0,0,0,0,0,0,0]])
        HOUSEKEEP[1] = np.array([[0,0,0,0]])
    SensorInfo_Output.HOUSEKEEP = HOUSEKEEP
    
    # HAULWINDOW, [1024,"吊拉窗户"]
    CurAlarmID = 1024
    HAULWINDOWMinHeight = 1.8 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            HAULWINDOW = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                HAULWINDOW.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], HAULWINDOWMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
        else:
            HAULWINDOW = [[0,0,0,0,0,0,0]]
    else:
        HAULWINDOW = [[0,0,0,0,0,0,0]]
    SensorInfo_Output.HAULWINDOW = np.array(HAULWINDOW)
    
    # BUILDLADDER, [2048,"搭人梯"]
    CurAlarmID = 2048
    BUILDLADDERMinHeight = 3.0 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            BUILDLADDER = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                BUILDLADDER.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], BUILDLADDERMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
        else:
            BUILDLADDER = [[0,0,0,0,0,0,0]]
    else:
        BUILDLADDER = [[0,0,0,0,0,0,0]]
    SensorInfo_Output.BUILDLADDER = np.array(BUILDLADDER)
    
    # STANDQUILT, [4096,"站在被子上做板报"]
    CurAlarmID = 4096
    STANDQUILTMinHeight = 2.0 # 最低高度
    if CurAlarmID in ConfigInfo['ALARM_INFO'].keys():
        CurDeteSensor = SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteSensor']
        if CurSensorId in CurDeteSensor:
            STANDQUILT = []
            for TempBbox in SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteArea']:
                STANDQUILT.append([TempBbox[0], TempBbox[1], TempBbox[2], TempBbox[3], STANDQUILTMinHeight, SrcConfigInfo['ALARM_INFO'][CurAlarmID]['DeteMaxHeight'], 1]) # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
        else:
            STANDQUILT = [[0,0,0,0,0,0,0]]
    else:
        STANDQUILT = [[0,0,0,0,0,0,0]]
    SensorInfo_Output.STANDQUILT = np.array(STANDQUILT)
    

    return SensorInfo_Output

    
def CalConfigMultiAreaEdge(ConfigInfo):
    """
    功能：根据当前配置文件信息计算多个区域的边界信息
    """
    MultiSensorDeteArea = []
    
    # SENSORS_INFO
    ConfigSensorInfo = ConfigInfo['SENSORS_INFO']

    for sensor_name in ConfigSensorInfo:
        if sensor_name == 'currsensorid':
            continue
        CurSensorArea = ConfigSensorInfo[sensor_name]['Area'][0]
        MultiSensorDeteArea.append(CurSensorArea)
    
    return MultiSensorDeteArea
    
if __name__ == '__main__':
    print('ReadIniConfig')
    
    TestReadIniConfigFlag = 1 # 测试读取 ini 文件
    if TestReadIniConfigFlag == 1:
#        SensorConifgFileName = '../SensorConfig.ini'
#        SensorConifgFileName = '../SensorConfig_test.ini'

        # 读取 ini 配置文件
        t1 = time.time()
        ConfigInfo = ReadIniConfig(SensorConifgFileName)
        t2 = time.time()
    
        # 转换为以前文件格式
        TransConfigInfo = TransformConfigInfo(ConfigInfo)
        
        print('  ConfigInfo = {}'.format(ConfigInfo))
        print('  TransConfigInfo = {}'.format(TransConfigInfo))
        print('Sensor Information: ')
        print('isFlip: ' + str(TransConfigInfo.isFlip))
        print('H: \n' + str(TransConfigInfo.H))
        print('BED: \n' + str(TransConfigInfo.BED))
        print('INTERNALSUPERVISOR: \n' + str(TransConfigInfo.INTERNALSUPERVISOR))
        print('TOILET: \n' + str(TransConfigInfo.TOILET))
        print('WINDOW: \n' + str(TransConfigInfo.WINDOW))
        print('SLEEPTIME: \n' + str(TransConfigInfo.SLEEPTIME))
        
        print('ReadIniConfig time = {}'.format(t2 - t1))
        
