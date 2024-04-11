# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:55:45 2020

@author: HYD
"""

import sys
import numpy as np
import time 
from collections import defaultdict
import copy
import os
try:
    import configparser
except:
    from six.moves import configparser
    
from util.PolygonFuns import TransBboxToPoints
#from PolygonFuns import TransBboxToPoints


#####################################################################
# 读取配置文件：
#       ReadSensorConfig(FileName)
#####################################################################   
def ReadSensorConfig(FileName):
    # input: sensor information file .ini
    # output: senfor information
    #       ROOM, DETEAREA, SENSOREDGEAREA
    
    # read config information
    config = configparser.ConfigParser()
#    config.readfp(open(FileName))

    fp = open(FileName, mode='r')
    config.read_file(fp)
    
    # init room config info
    RoomInfo = dict()
    RoomNum = int(config.get('ROOMID', 'roomcount'))
    RoomIndex = [] # room index, eg: 1,2,3,4
    for roomid in config.items('ROOMID'):
        if roomid[0].find('roomid')!=-1:
            roomid_idx = roomid[0].find('_')
            RoomIndex.append(roomid[0][roomid_idx+1:]) # room index, eg: 1,2,3,4
            RoomInfo[roomid[0][roomid_idx+1:]] = dict() # init RoomInfo-->dict()
            RoomInfo[roomid[0][roomid_idx+1:]]['ROOMID'] = roomid[1] # ROOMID
            RoomInfo[roomid[0][roomid_idx+1:]]['ROOMAREA'] = dict() # init ROOMAREA-->dict()
            RoomInfo[roomid[0][roomid_idx+1:]]['SUPERVISEAREA'] = dict() # init ROOMAREA-->dict()
            RoomInfo[roomid[0][roomid_idx+1:]]['DETEAREA'] = dict() # init DETEAREA-->dict()
    
    RoomIDName = []
    # config all sections
    sections = config.sections()
    for sect in sections:
        TempSect = sect
        # ROOMID
        if TempSect == 'ROOMID':
            # room number
            roomnum = int(config.get(TempSect, 'roomcount'))
            # room id
            for i_room in range(roomnum):
                RoomIDName.append(config.get(TempSect, 'roomid_'+ RoomIndex[i_room]))
                
        # ROOMAREA
        if TempSect == 'ROOMAREA':
            # room area
            for i_room in range(roomnum):
                # roomareacount
                roomareacount = int(config.get(TempSect, 'roomareacount'+ RoomIndex[i_room]))
                TempRoomArea = []
                for roomarea in range(roomareacount):
                    numbers = list(map(float, config.get(TempSect, 'roomarea'+ RoomIndex[i_room] + '_' + str(roomarea+1)).split()))
                    TempRoomArea = np.hstack((TempRoomArea,numbers)) # N_roomarea x 4
#                    print('numbers = {}'.format(numbers))
                    RoomInfo[RoomIndex[i_room]]['ROOMAREA'][roomarea] = numbers
        
        # SUPERVISEAREA
        if TempSect == 'SUPERVISEAREA':
            # room area
            for i_room in range(roomnum):
                # superviseareacount
                superviseareacount = int(config.get(TempSect, 'superviseareacount'+ RoomIndex[i_room]))
                for supervisearea in range(superviseareacount):
                    numbers = list(map(float, config.get(TempSect, 'supervisearea'+ RoomIndex[i_room] + '_' + str(supervisearea+1)).split()))
                    RoomInfo[RoomIndex[i_room]]['SUPERVISEAREA'][supervisearea] = numbers
                    
        # DETEAREA
        if TempSect == 'DETEAREA':
            # dete area
            for i_room in range(roomnum):
                # deteareacount
                deteareacount = int(config.get(TempSect, 'deteareacount'+ RoomIndex[i_room]))
                for detearea in range(deteareacount):
                    numbers = list(map(float, config.get(TempSect, 'detearea'+ RoomIndex[i_room] + '_' + str(detearea+1)).split()))
                    RoomInfo[RoomIndex[i_room]]['DETEAREA'][str(int(numbers[0]))] = numbers[1:] # dict( N_detearea )
                
        # SENSOREDGEAREA
        if TempSect == 'SENSOREDGEAREA':
            # sensor edge area
            for i_room in range(roomnum):
                # ensoredgearea
                sensoredgeareacount = int(config.get(TempSect, 'sensoredgeareacount'+ RoomIndex[i_room]))
                TempSensorEdgeArea = []
                for sensoredgearea in range(sensoredgeareacount):
                    numbers = list(map(float, config.get(TempSect, 'sensoredgearea'+ RoomIndex[i_room] + '_' + str(sensoredgearea+1)).split()))
                    TempSensorEdgeArea = np.hstack((TempSensorEdgeArea,numbers)) # N_sensoredgearea x 4
                RoomInfo[RoomIndex[i_room]]['SENSOREDGEAREA'] = np.reshape(TempSensorEdgeArea,[int(len(TempSensorEdgeArea)/4),4])

        # SLEEPTIME
        if TempSect == 'SLEEPTIME':
            # sleeptime 
            sleeptimecount = int(config.get(TempSect, 'sleeptimecount'))
            TempSleepTime = []
            for sleeptime in range(sleeptimecount):
                numbers = list(map(float, config.get(TempSect, 'sleeptime' + '_' + str(sleeptime+1)).split()))
                TempSleepTime = np.hstack((TempSleepTime,numbers)) # N_sensoredgearea x 4
            SleepTime = np.reshape(TempSleepTime,[int(len(TempSleepTime)/4),4]) # [start_h, start_m, end_h, end_m]

        # INTERNALSUPERVISORHUMANNUM
        if TempSect == 'INTERNALSUPERVISORHUMANNUM':
            # sleeptime 
            INTERNALSUPERVISOR_HumanNum = int(config.get(TempSect, 'INTERNALSUPERVISOR_HumanNum'))
            
        # WARNLOOSEDEGREE
        if TempSect == 'WARNLOOSEDEGREE':
            # sleeptime 
            WarnLooseDegree = int(config.get(TempSect, 'warn_loose_degree'))
            
    # 如果没有‘WarnLooseDegree’，则设置默认值：2
    NewItemsName = 'WARNLOOSEDEGREE'
    NewVariableInit = 2
    if NewItemsName not in config.sections():
        WarnLooseDegree = NewVariableInit

    # fclose
    fp.close()
            
    return RoomInfo, SleepTime, INTERNALSUPERVISOR_HumanNum, WarnLooseDegree
    
#####################################################################
# 读取ini配置文件：
#       ReadIniConfig(FileName)
#####################################################################  
def ReadIniConfig(ConfigFileName):
    """
    功能：读取配置文件
    输入：
        ConfigFileName：.ini 文件名
            其中：#告警id	"warningTypeModel":[
                    [1,"未在指定时间休息"],  vvv
                    [2,"未在指定区域监督"],  vvv
                    [4,"厕所区域异常"],     vvv
                    [8,"窗户区域异常"],
                    [16,"高度异常"],        vvv
                    [32,"非休息时间休息"],   vvv
                    [64,"进入三角区域"],
                    [128,"内务不整"],
                    [512,"单人留仓"],       vvv
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
    
def ReadIniConfigGlobalInfo(ConfigFileName):
    """
    功能：读取ini配置文件，并且转换检测区域格式
    """
    # 读取原始 ini配置文件
    ConfigInfoSrc = ReadIniConfig(ConfigFileName)
    ConfigGlobalInfo = copy.deepcopy(ConfigInfoSrc)
    # DETEAREA
    ConfigGlobalInfo['SENSORS_INFO']['DeteAreaPoints'] = []
    DETEAREA = dict()
    for key in ConfigInfoSrc['SENSORS_INFO']:
        if key.isdigit(): # '10001'/'10010'
            CurRoomInfoArea = np.array(ConfigInfoSrc['SENSORS_INFO'][key]['Area']) # [xmin, ymin, xmax, ymax]
            # 转换检测区域坐标
            if CurRoomInfoArea.shape[0]>0:
                CurRoomInfoAreaSelect = CurRoomInfoArea[0,[0,2,1,3]]
                CurRoomInfoAreaTrans = TransBboxToPoints(CurRoomInfoAreaSelect) # [x1,y1,x2,y2,x3,y3,x4,y4,...]
                DETEAREA[key] = CurRoomInfoAreaTrans
            else:
                DETEAREA[key] = []
    ConfigGlobalInfo['SENSORS_INFO']['DeteAreaPoints'] = DETEAREA

    # ROOMAREA
    for alarm_key in ConfigInfoSrc['ALARM_INFO']:
        ConfigGlobalInfo['ALARM_INFO'][alarm_key]['DeteAreaPoints'] = []
        CurAlarmInfo = ConfigInfoSrc['ALARM_INFO'][alarm_key] # 1/2/16/512/
        CurAlarmInfoDeteArea = np.array(CurAlarmInfo['DeteArea']) # [[xmin, ymin, xmax, ymax],[xmin, ymin, xmax, ymax],...]
        # 转换检测区域坐标
        if CurAlarmInfoDeteArea.shape[0]>0:
            CurRoomInfoAreaSelect = CurAlarmInfoDeteArea[:,[0,2,1,3]] # [Nx4]
            CurRoomInfoAreaTrans = np.zeros([CurRoomInfoAreaSelect.shape[0], 8]) # [Nx8]
            for i in range(CurRoomInfoAreaSelect.shape[0]):
                CurRoomInfoAreaTrans[i] = TransBboxToPoints(CurRoomInfoAreaSelect[i]) # [x1,y1,x2,y2,x3,y3,x4,y4,...]
            ConfigGlobalInfo['ALARM_INFO'][alarm_key]['DeteAreaPoints'] = CurRoomInfoAreaTrans
        else:
            ConfigGlobalInfo['ALARM_INFO'][alarm_key]['DeteAreaPoints'] = CurAlarmInfoDeteArea
    
    return ConfigGlobalInfo
    
    
#####################################################################
# 判断文件时间是否修改
##################################################################### 
def ReadReviseTime(SensorConfigFileName, SensorConfigReviseFileName):
    """
    功能：对比 cinfig 文件时间，判断文件是否被修改
    """
    #global FileReviseBool
    # 读取文件文件上次修改信息
    with open(SensorConfigReviseFileName, "r") as fp:
        log_data = fp.read()
    # 对比时间
    mtime = os.stat(SensorConfigFileName).st_mtime
    file_modify_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
    # 是否相等
    FileReviseBool = (log_data == file_modify_time)
    if FileReviseBool == False: # 如果不相同则保存当前修改时间
        with open(SensorConfigReviseFileName, "w") as fp:
            fp.write(file_modify_time)
    
    return FileReviseBool
    
if __name__ == '__main__':
    print('ReadIniConfig')
    
    TestReadIniConfigFlag = 1 # 测试读取 ini 文件
    if TestReadIniConfigFlag == 1:
        SensorConfigFileName = '../SensorConfig.ini'
        SensorConfigFileName_Trans = '../SensorConfig_trans.ini'

        # 读取 ini 配置文件
        t1 = time.time()
#        ConfigInfo = ReadIniConfig(SensorConfigFileName)
        SensorInfoAll, SleepTime, INTERNALSUPERVISOR_HumanNum, WarnLooseDegree = ReadSensorConfig(SensorConfigFileName_Trans)
        ConfigGloablInfo = ReadIniConfigGlobalInfo(SensorConfigFileName)
        
        t2 = time.time()
        
#        print('ConfigInfo = ', SensorInfoAll)
#        print('ConfigGloablInfo = ', ConfigGloablInfo)
        print('ConfigGloablInfo = ', ConfigGloablInfo['SENSORS_INFO']['DeteAreaPoints'])
        print('ConfigGloablInfo = ', ConfigGloablInfo['ALARM_INFO'][512]['DeteAreaPoints'])

#        print('ReadSensorConfig = ', SensorInfoAll, SleepTime, INTERNALSUPERVISOR_HumanNum, WarnLooseDegree)
    
        # 转换为以前文件格式
#        TransConfigInfo = TransformConfigInfo(ConfigInfo)
        
#        print('  ConfigInfo = {}'.format(ConfigInfo))
#        print('  TransConfigInfo = {}'.format(TransConfigInfo))
#        print('Sensor Information: ')
#        print('isFlip: ' + str(TransConfigInfo.isFlip))
#        print('H: \n' + str(TransConfigInfo.H))
#        print('BED: \n' + str(TransConfigInfo.BED))
#        print('INTERNALSUPERVISOR: \n' + str(TransConfigInfo.INTERNALSUPERVISOR))
#        print('TOILET: \n' + str(TransConfigInfo.TOILET))
#        print('WINDOW: \n' + str(TransConfigInfo.WINDOW))
#        print('SLEEPTIME: \n' + str(TransConfigInfo.SLEEPTIME))
        
#        print('ReadIniConfig time = {}'.format(t2 - t1))
        
