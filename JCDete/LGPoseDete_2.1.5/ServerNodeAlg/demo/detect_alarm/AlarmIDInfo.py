# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:32:26 2020

@author: HYD
"""

import numpy as np

class AlarmIDInfo():
    def __init__(self, CaseName='NewLG'):
        # 设置现有告警ID 序号
        self.CaseName = CaseName
        self.AlarmIDAll = [1,2,4,8,16,32,64,128,512,1024,2048,4096,16384] # 所有告警ID 信息
        self.Alarm_NoLying = 1 # "未在指定时间休息"
        self.Alarm_InternalSupervisor = 2 # "未在指定区域监督"
        self.Alarm_Toilet = 4 # "厕所区域异常"
        self.Alarm_Window = 8 # "窗户区域异常"
        self.Alarm_AboveHeight = 16 # "高度异常"
        self.Alarm_WrongLying = 32 # "非休息时间休息"
        self.Alarm_InTriangleArea = 64 # "进入三角区域"
        self.Alarm_Housekeep = 128 # "内务不整"
        self.Alarm_Alone = 512 # "单人留仓"
        self.Alarm_HaulWindow = 1024 # "吊拉窗户"
        self.Alarm_BuildLadder = 2048 # "搭人梯"
        self.Alarm_StandQuilt = 4096 # "站在被子上做板报"
        self.Alarm_SensorCover = 16384 # "传感器遮挡"
        