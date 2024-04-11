# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:29:00 2019

    同步多传感器数据

@author: HYD
"""

import numpy as np
import os
import csv

# --------------------------------------------------------------------
# 同步多传感器数据
#       SyncData(SrcDepthData) # 同步多传感器原始数据
#       SyncDeteInfo(SensorNameGroup, Mode) # 同步多传感器检测结果信息
# --------------------------------------------------------------------
def SyncData(Mode):
    """
    Synchronize multi sensor source data
    Inputs:
        Mode: string format
                'OnLine' or 'OffLine'
    Outputs:
        data: multi sensor data
                [N x 3]
    """
    print('Start SyncData.')
    
    return 0
    
def SyncDeteInfo(SensorNameGroup, SensorDeteHumanInfoGroup, Mode):
    """
    Synchronize multi sensor detect information
    Inputs:
        SensorDeteHumanInfoGroup: sensor detect human info, list 
                
        Mode: string format
                'OnLine' or 'OffLine'
    Outputs:
        data: multi sensor data
                [N x 3]
    """
    print('Start SyncDeteInfo.')
    
    # Mode
    if Mode == 'OnLine':
        print("Mode == 'OnLine'.")
    elif Mode == 'OffLine':
        print("Mode == 'OffLine'.")
        # 读取每个传感器信息的方法
        GetSensorDeteInfoFlag = 2 # if GetSensorDeteInfoFlag == 1, 按帧序号保存结果
                                  # if GetSensorDeteInfoFlag == 2, 按时间戳保存结果

        if GetSensorDeteInfoFlag == 1: # 按帧序号保存结果
            # 读取文件
            OneSensorDeteInfo = dict() 
            for i_sensor in range(len(SensorDeteHumanInfoGroup)):
                if os.path.exists(SensorDeteHumanInfoGroup[i_sensor]) == 1: # 文件是否存在
                    print(SensorDeteHumanInfoGroup[i_sensor])
                    # 读取excel 文件
                    OutDeteHumanInfo = open(SensorDeteHumanInfoGroup[i_sensor], "r")
                    OutDeteHumanInfo_csv = csv.reader(OutDeteHumanInfo)
                    for row in OutDeteHumanInfo_csv:
                        OneLineDeteInfoInt = []
                        OneLineDeteInfoInt.append(float(SensorNameGroup[i_sensor])) # sensor name, eg:171
                        OneLineDeteInfoInt.append(float(row[2])) # label
                        OneLineDeteInfoInt.append(float(row[3])) # x
                        OneLineDeteInfoInt.append(float(row[4])) # y
                        OneLineDeteInfoInt.append(float(row[5])) # z
                        # 保存[label, x, y, z]                    
                        try: 
                            TempPreSensorDeteInfo = OneSensorDeteInfo[int(float(row[0]))]
                            TempPreSensorDeteInfo = np.row_stack((TempPreSensorDeteInfo, OneLineDeteInfoInt))
                            OneSensorDeteInfo[int(float(row[0]))] = TempPreSensorDeteInfo # 更新多个传感器同一帧结果
                        except:
                            OneSensorDeteInfo[int(float(row[0]))] = OneLineDeteInfoInt                     
                    OutDeteHumanInfo.close()
            result = OneSensorDeteInfo
                    
        if GetSensorDeteInfoFlag == 2: # 按时间戳保存结果
            # 基准传感器编号
            BaseSensorName = ''
            # 多传感器时间戳信息
            MultiSensorWorldTimeInfo = dict() # 多传感器时间信息
            MultiSensorDeteInfo = dict() # 多传感器检测信息
            # 寻找对应时间戳, eg: '171':[frame_idx, world_time, label, x, y, z]
            for i_sensor in range(len(SensorDeteHumanInfoGroup)):
                if os.path.exists(SensorDeteHumanInfoGroup[i_sensor]) == 1: # 文件是否存在
                    print(SensorDeteHumanInfoGroup[i_sensor])
                    # 获取基准传感器编号
                    if BaseSensorName == '':
                        BaseSensorName = SensorNameGroup[i_sensor]
                    # 读取excel 文件
                    OutDeteHumanInfo = open(SensorDeteHumanInfoGroup[i_sensor], "r")
                    OutDeteHumanInfo_csv = csv.reader(OutDeteHumanInfo)
                    for row in OutDeteHumanInfo_csv:
                        OneLineDeteInfoInt = []
                        OneLineDeteInfoInt.append(float(row[0])) # frame index
                        OneLineDeteInfoInt.append(float(row[6])) # world time 
                        OneLineDeteInfoInt.append(float(row[2])) # label
                        OneLineDeteInfoInt.append(float(row[3])) # x
                        OneLineDeteInfoInt.append(float(row[4])) # y
                        OneLineDeteInfoInt.append(float(row[5])) # z
#                        print('OneLineDeteInfoInt = {}'.format(OneLineDeteInfoInt))
                        try: 
                            TempPreSensorDeteInfo = MultiSensorWorldTimeInfo[SensorNameGroup[i_sensor]]
                            TempPreSensorDeteInfo = np.row_stack((TempPreSensorDeteInfo, OneLineDeteInfoInt))
                            MultiSensorWorldTimeInfo[SensorNameGroup[i_sensor]] = TempPreSensorDeteInfo # 更新多个传感器同一帧结果
                        except:
                            MultiSensorWorldTimeInfo[SensorNameGroup[i_sensor]] = OneLineDeteInfoInt                     
                    OutDeteHumanInfo.close()
            # 初始化检测关系
            ValidObjIndex = dict()
            for i_sensor in range(len(SensorDeteHumanInfoGroup)):
                try:
                    ValidObjIndex[SensorNameGroup[i_sensor]] = -1*np.ones([MultiSensorWorldTimeInfo[SensorNameGroup[i_sensor]].shape[0]]) # 初始化有效目标标签
                except:
                    print(SensorNameGroup[i_sensor] + ' not exist.')
            # 对应关系, eg: '1544013021182':[171,line_idx,frame_idx,world_time]
            if not (BaseSensorName == ''):
                BaseSensorWorldTimeInfo = MultiSensorWorldTimeInfo[BaseSensorName]
                BaseSensorNumLine = BaseSensorWorldTimeInfo.shape[0]
                for i_line in range(BaseSensorNumLine): # 基准传感器时间信息
                    TempBaseSensorLine = BaseSensorWorldTimeInfo[i_line] # [index, worldtime]
                    # 保存当前传感器检测信息
                    OneLineDeteInfoInt = []
#                    OneLineDeteInfoInt.append(float(BaseSensorName)) # sensor name, eg:171
#                    OneLineDeteInfoInt.append(float(i_line)) # line
#                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,0])) # index
#                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,1])) # worldtime

                    OneLineDeteInfoInt.append(float(BaseSensorName)) # sensor name, eg:171
                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,2])) # label
                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,3])) # x
                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,4])) # y
                    OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_line,5])) # z
                    try:
                        if ValidObjIndex[BaseSensorName][i_line] == -1:
                            MultiSensorDeteInfo[int(TempBaseSensorLine[0])] = np.row_stack((MultiSensorDeteInfo[int(TempBaseSensorLine[0])], OneLineDeteInfoInt))
                            ValidObjIndex[BaseSensorName][i_line] = 1
                    except:
                        if ValidObjIndex[BaseSensorName][i_line] == -1:
                            MultiSensorDeteInfo[int(TempBaseSensorLine[0])] = OneLineDeteInfoInt
                            ValidObjIndex[BaseSensorName][i_line] = 1
                            
                    for i_sensor in SensorNameGroup: # 其他传感器信息
                        if (not (i_sensor==BaseSensorName)):
                            try:
                                TempDist = abs(MultiSensorWorldTimeInfo[i_sensor][:,1]-TempBaseSensorLine[1])
                                MinTempDist = min(TempDist)
                                if MinTempDist < 700: # 在 1000ms 内有效
                                    TempIdx = np.where(TempDist == MinTempDist)
#                                    print('TempDist TempIdx = {}'.format(TempIdx))
                                    # 保存其他传感器信息
                                    for i_other_idx in TempIdx[0]:
                                        i_other = i_other_idx
#                                        print('i_other = {}'.format(i_other))
                                        OneLineDeteInfoInt = []
#                                        OneLineDeteInfoInt.append(float(i_sensor)) # sensor name, eg:171
#                                        OneLineDeteInfoInt.append(float(i_other)) # index
#                                        OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_other,0])) # index
#                                        OneLineDeteInfoInt.append(float(BaseSensorWorldTimeInfo[i_other,1])) # worldtime

                                        OneLineDeteInfoInt.append(float(i_sensor)) # sensor name, eg:171
                                        OneLineDeteInfoInt.append(float(MultiSensorWorldTimeInfo[i_sensor][i_other,2])) # label
                                        OneLineDeteInfoInt.append(float(MultiSensorWorldTimeInfo[i_sensor][i_other,3])) # x
                                        OneLineDeteInfoInt.append(float(MultiSensorWorldTimeInfo[i_sensor][i_other,4])) # y
                                        OneLineDeteInfoInt.append(float(MultiSensorWorldTimeInfo[i_sensor][i_other,5])) # z
                                        
                                        if ValidObjIndex[i_sensor][i_other] == -1:
                                            MultiSensorDeteInfo[int(TempBaseSensorLine[0])] = np.row_stack((MultiSensorDeteInfo[int(TempBaseSensorLine[0])], OneLineDeteInfoInt))
                                            ValidObjIndex[i_sensor][i_other] = 1
                            except:
                                continue
                           
#            print('BaseSensorName = {}'.format(BaseSensorName))
#            print('MultiSensorDeteInfo = {}'.format(MultiSensorDeteInfo))
#            print('MultiSensorWorldTimeInfo: \n {}'.format(MultiSensorWorldTimeInfo))

#            # 读取文件
#            MultiSensorDeteInfoAll = dict() # 所有传感器位置信息， eg: 'id':[]
#            for i_sensor in range(len(SensorDeteHumanInfoGroup)):
#                if os.path.exists(SensorDeteHumanInfoGroup[i_sensor]) == 1: # 文件是否存在
#                    print(SensorDeteHumanInfoGroup[i_sensor])
#                    # 读取excel 文件
#                    OutDeteHumanInfo = open(SensorDeteHumanInfoGroup[i_sensor], "r")
#                    OutDeteHumanInfo_csv = csv.reader(OutDeteHumanInfo)
#                    for row in OutDeteHumanInfo_csv:
#                        OneLineDeteInfoInt = []
#                        OneLineDeteInfoInt.append(float(SensorNameGroup[i_sensor])) # sensor name, eg:171
#                        OneLineDeteInfoInt.append(float(row[2])) # label
#                        OneLineDeteInfoInt.append(float(row[3])) # x
#                        OneLineDeteInfoInt.append(float(row[4])) # y
#                        OneLineDeteInfoInt.append(float(row[5])) # z
#                        # 保存[label, x, y, z]                    
#                        try: 
#                            TempPreSensorDeteInfo = OneSensorDeteInfo[int(float(row[0]))]
#                            TempPreSensorDeteInfo = np.row_stack((TempPreSensorDeteInfo, OneLineDeteInfoInt))
#                            OneSensorDeteInfo[int(float(row[0]))] = TempPreSensorDeteInfo # 更新多个传感器同一帧结果
#                        except:
#                            OneSensorDeteInfo[int(float(row[0]))] = OneLineDeteInfoInt                     
#                    OutDeteHumanInfo.close()
#
#            result = MultiSensorWorldTimeInfo
        
            result = MultiSensorDeteInfo
        
    return result


if __name__ == '__main__':
    print('Start.')
    
    TestSyncDataFlag = 0 # Test Sync Data
    TestSyncDeteInfoFlag = 1 # Test Sync Dete Info，读取每个传感器的检测结果
    
    if TestSyncDataFlag == 1:
        # 传感器名称
        SensorNameGroup = ['171', '172', '173', '174', '175', '176', '177']
        # 输入数据
        BaseDepthHSFFileName = 'T:\\NJ{}\\Depth2018-12-07-150000.HSF.HSF'.format(SensorNameGroup[0]) # depth data
        
    if TestSyncDeteInfoFlag == 1:
        # 传感器名称
        SensorNameGroup = ['171', '172', '173', '174', '175', '176', '177']
        # 输入数据
        SensorDeteHumanInfoGroup = []
        for i_sensor in range(len(SensorNameGroup)):
            # 第一次每个传感器的检测结果文件
#            SensorDeteHumanInfoGroup.append('..\\MultiSensorDeteInfo\\DeteHumanInfo_' + str(SensorNameGroup[i_sensor]) + '.csv')
            # 第二次每个传感器的检测结果文件，HSF
            SensorDeteHumanInfoGroup.append('..\\MultiSensorDeteInfo\\DeteCsv\\DeteHumanInfo_' + str(SensorNameGroup[i_sensor]) + '.csv') 
        # SyncDeteInfo
        Mode = 'OffLine'
        result = SyncDeteInfo(SensorNameGroup, SensorDeteHumanInfoGroup, Mode)
        print('result {}'.format(result))
        
    
    print('End.')
