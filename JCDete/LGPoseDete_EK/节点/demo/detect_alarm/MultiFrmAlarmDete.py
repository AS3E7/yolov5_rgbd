# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:25:03 2020

@author: HYD
"""

import numpy as np
import copy

def SequenceFrmAlarmDete(CurFrmObjInfo, PreAlarmInfo, WTime, SensorFrmTime=-1, SeqFrmThod=5, SeqTimeThod=0, SeqFrmMin=-1, SeqFrmMax=10, SeqFrmUpRate=1, SeqFrmDnRate=1, CurFrmValidFlag=False):
    """
    功能：序列帧数信息告警检测
    输入：
        CurFrmObjInfo：当前帧目标检测结果:[[x,y,z],[x,y,z],...]，如果无结果则: CurFrmObjInfo=[]
        PreAlarmInfo：前一帧此类告警信息: [告警类型ID，告警状态，告警ID，传感器编号，位置X，位置Y，位置Z，其他告警信息，人员姿态，累计告警帧数，累计告警起始时间戳，累计告警时长，房间编号]
        SeqFrmThod: 连续帧信息
            SeqFrmMin/SeqFrmMax/SeqFrmUpRate/SeqFrmDnRate
        SeqTimeThod：连续时间信息
    """

    CurAlarmInfo = copy.deepcopy(PreAlarmInfo)
    # 告警消失起始帧对应额传感器时间戳（展示使用‘ReservedInfo’）
    if CurFrmValidFlag == True: # 告警有效，存在告警目标
        if PreAlarmInfo['SumFrame'] == SeqFrmMin:
            CurAlarmInfo['ReservedInfo'] = SensorFrmTime
        elif PreAlarmInfo['SumFrame'] == SeqFrmMin+1:
            CurAlarmInfo['ReservedInfo'] = PreAlarmInfo['ReservedInfo'] # 传感器当前帧时间戳，最开始有告警帧
        elif PreAlarmInfo['SumFrame'] == SeqFrmMax:
            CurAlarmInfo['ReservedInfo'] = SensorFrmTime
#        else:
#            CurAlarmInfo['ReservedInfo'] = -1 # 
    else: # 告警无效，不存在告警目标
        if PreAlarmInfo['SumFrame'] == SeqFrmMax:
            CurAlarmInfo['ReservedInfo'] = PreAlarmInfo['ReservedInfo'] # 传感器当前帧时间戳，最开始无告警帧
        
    # 更新连续检测帧数
    if CurFrmValidFlag == True: # 是否在检测时间内
        if CurFrmObjInfo.shape[0] == 0: # 如果当前帧不存在目标
            CurAlarmInfo['SumFrame'] = PreAlarmInfo['SumFrame'] - SeqFrmDnRate*1
        else: # 如果当前帧存在目标
            CurAlarmInfo['ObjX'] = CurFrmObjInfo[0]
            CurAlarmInfo['ObjY'] = CurFrmObjInfo[1]
            CurAlarmInfo['ObjZ'] = CurFrmObjInfo[2]
            CurAlarmInfo['SumFrame'] = PreAlarmInfo['SumFrame'] + SeqFrmUpRate*1
    else:
        CurAlarmInfo['SumFrame'] = PreAlarmInfo['SumFrame'] - SeqFrmUpRate*1
    CurAlarmInfo['SumFrame'] = max(CurAlarmInfo['SumFrame'], SeqFrmMin)
    CurAlarmInfo['SumFrame'] = min(CurAlarmInfo['SumFrame'], SeqFrmMax)
    # 根据检测帧数判断告警状态
    # 告警持续时间
    if CurAlarmInfo['SumFrame'] >= SeqFrmThod and CurAlarmInfo['StartWorldTime'] < 0:#持续一定帧数
        CurAlarmInfo['StartWorldTime'] = WTime
    if CurAlarmInfo['SumFrame'] < SeqFrmThod:
        CurAlarmInfo['StartWorldTime'] = -1
        
    # 告警持续有效性判断
    if CurAlarmInfo['SumFrame'] >= SeqFrmThod:
        CurAlarmInfo['SumTime'] = WTime - CurAlarmInfo['StartWorldTime'] # 告警时长，达到阈值帧数开始计时
        CurAlarmInfo['AlarmState'] = 0 # 告警
        if (CurAlarmInfo['SumFrame'] >= SeqFrmThod) and (CurAlarmInfo['SumTime'] >= SeqTimeThod*60):#持续一定帧数
            CurAlarmInfo['AlarmState'] = 1 # 告警
    else:
        CurAlarmInfo['SumTime'] = -1 # 告警时长
        if CurAlarmInfo['SumFrame'] > SeqFrmMin:
            CurAlarmInfo['AlarmState'] = 0 # 告警
        else:
            CurAlarmInfo['AlarmState'] = -1 # 不告警
            

    
            
            
#    if CurAlarmInfo['SumFrame'] >= SeqFrmThod:
#        CurAlarmInfo['AlarmState'] = 0 # 告警
#        if (CurAlarmInfo['SumFrame'] >= SeqFrmThod) and ((WTime - CurAlarmInfo['StartWorldTime']) >= SeqTimeThod*60) and (CurFrmObjInfo.shape[0] > 0):#持续一定帧数
#            CurAlarmInfo['AlarmState'] = 1 # 告警
#    else:
#        if CurAlarmInfo['SumFrame'] > SeqFrmMin:
#            CurAlarmInfo['AlarmState'] = 0 # 告警
#        else:
#            CurAlarmInfo['AlarmState'] = -1 # 不告警
    
    

#    if CurAlarmInfo['SumFrame'] >= SeqFrmThod and (WTime - CurAlarmInfo['StartWorldTime']) > SeqTimeThod*60:#持续一定帧数
#        if CurFrmObjInfo.shape[0] > 0:
#            CurAlarmInfo['AlarmState'] = 1 # 告警
#        else:
#            CurAlarmInfo['AlarmState'] = 0 # 预告警
#    else:
#        if CurFrmObjInfo.shape[0] > 0:
#            CurAlarmInfo['AlarmState'] = 0 # 预告警
#        else:
#            CurAlarmInfo['AlarmState'] = -1 # 不告警
    
    return CurAlarmInfo

if __name__ == '__main__':
    print('Start.')
    
    
    print('End.')
