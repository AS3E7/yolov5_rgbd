# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:25:07 2020

@author: HYD
"""

from detect_alarm import AlarmString

def CreatDetecter(SensorConfigFileName):
    """
    功能：初始化检测结果，用于 外部接口调用 
    """
    
    DetecterStr = AlarmString.CreatDetecter(SensorConfigFileName)
    
    return DetecterStr