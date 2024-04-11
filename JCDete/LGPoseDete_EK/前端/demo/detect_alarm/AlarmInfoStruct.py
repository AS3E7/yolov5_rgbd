# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:15:35 2020

@author: HYD
"""

import numpy as np

# SensorInfo 
class SensorInfo:
    def __init__(self):
        self.isFlip = 0
        self.H = np.array([]) # 4*4
        self.BED = np.array([]) # 16*6
        self.INTERNALSUPERVISOR = np.array([]) # 2*4
        self.TOILET = np.array([]) # 2*4
        self.WINDOW = np.array([]) # 2*4
        self.SLEEPTIME = np.array([]) # 2*4
        self.ABOVEHEIGHTTHOD = np.array([]) # 1
        self.INTRIANGLEREGION = np.array([]) # 1*10, [x1,y1,x2,y2,x3,y3,x4,y4,z1,z2]
        self.HOUSEKEEP = {} # [area, time]
        
# DetectInfo
class DetectInfo:
    def __init__(self):
        self.BED = np.array([]) # 16*7
        self.INTERNALSUPERVISOR = np.array([]) # 2*7
        self.TOILET = np.array([]) # 2*7
        self.WINDOW = np.array([]) # 2*7
        self.ABOVEHEIGHT = np.array([]) # N*7
        self.WRONGTIMELIE = np.array([]) # N*7
        self.INTRIANGLEREGION = np.array([]) # N*7
        self.TOILET_STARTTIME = np.array([]) # 2*2
        self.WINDOW_STARTTIME = np.array([]) # 2*2
        self.HOUSEKEEP = np.array([]) # N*7
        self.HUMANINFO = np.array([]) # N*7
        self.HAULWINDOW = np.array([])
        self.BUILDLADDER = np.array([])
        self.STANDQUILT = np.array([])
        
        
        
        
        