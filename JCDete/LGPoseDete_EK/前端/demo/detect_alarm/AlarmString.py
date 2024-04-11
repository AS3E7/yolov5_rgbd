# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:49:31 2020

@author: HYD
"""

import numpy as np
from collections import defaultdict

from util.ReadConfig import ReadIniConfig, TransformConfigInfo
from detect_alarm.AlarmInfoStruct import DetectInfo, SensorInfo
      
# --------------------------------------------------------------------
# 初始化检测目标
#       CreatDetecter(SensorConfigFileName)
# --------------------------------------------------------------------
def CreatDetecter(SensorConfigFileName):
    # inputs:
    #       SensorConfigFileName: sensor config file name
    # outputs: 
    #       StartNumericResult: BED, INTERNALSUPERVISOR, TOILET, WINDOW, SLEEPTIME
    # 读取配置文件
    ConfigInfo = ReadIniConfig(SensorConfigFileName)
    # 转换为以前使用的读取后数据格式
    CurSensorInfo = TransformConfigInfo(ConfigInfo) # 配置文件信息转换

    # 获取 SensorInfo 部分信息
    H = CurSensorInfo.H
    Bed = CurSensorInfo.BED
    INTERNALSUPERVISOR = CurSensorInfo.INTERNALSUPERVISOR
    TOILET = CurSensorInfo.TOILET
    WINDOW = CurSensorInfo.WINDOW
    HAULWINDOW = CurSensorInfo.HAULWINDOW # 新增 HAULWINDOW， BUILDLADDER， STANDQUILT
    BUILDLADDER = CurSensorInfo.BUILDLADDER
    STANDQUILT = CurSensorInfo.STANDQUILT
    ABOVEHEIGHT = CurSensorInfo.ABOVEHEIGHT
    
    NumBed = Bed.shape[0] # [xmin, xmax, ymin, ymax, zmin, zmax, ValidArea]
    NumINTERNALSUPERVISOR = INTERNALSUPERVISOR.shape[0] # [xmin, xmax, ymin, ymax, ValidArea]
    NumTOILET = TOILET.shape[0] # [xmin, xmax, ymin, ymax, ValidArea]
    NumWINDOW = WINDOW.shape[0] # [xmin, xmax, ymin, ymax, ValidArea]
    NumHAULWINDOW = HAULWINDOW.shape[0] # [xmin, xmax, ymin, ymax, zmin, zmax, ValidFlag]
    NumBUILDLADDER = BUILDLADDER.shape[0] # [xmin, xmax, ymin, ymax, zmin, zmax, ValidFlag]
    NumSTANDQUILT = STANDQUILT.shape[0] # [xmin, xmax, ymin, ymax, zmin, zmax, ValidFlag]
    NumABOVEHEIGHT = ABOVEHEIGHT.shape[0]

    StartNumericResult = DetectInfo()
    StartNumericResult.BED = -1 * np.ones([NumBed, 7]) # PredBedPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    StartNumericResult.INTERNALSUPERVISOR = -1 * np.ones([NumINTERNALSUPERVISOR, 7]) # PredINTERNALSUPERVISORPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    StartNumericResult.TOILET = -1 * np.ones([NumTOILET, 7]) # PredTOILETPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
    StartNumericResult.WINDOW = -1 * np.ones([NumWINDOW, 7]) # PredWINDOWPos, eg: [ID, POSE, WARNINGNUM, X, Y, Z, WARNINGSTATE]
#    StartNumericResult.ABOVEHEIGHT = np.array([]) # 
    StartNumericResult.ABOVEHEIGHT = -1 * np.ones([NumABOVEHEIGHT, 7]) # 
    
    StartNumericResult.WRONGTIMELIE = np.array([]) # 
    StartNumericResult.INTRIANGLEREGION = np.array([]) # 
    StartNumericResult.TOILET_STARTTIME = -1 * np.ones([NumTOILET, 2]) # PredTOILET_STARTTIME, eg: [STARTTIME, STARTTIMESTATE]
    StartNumericResult.WINDOW_STARTTIME = -1 * np.ones([NumWINDOW, 2]) # PredWINDOW_STARTTIME, eg: [STARTTIME, STARTTIMESTATE]
    StartNumericResult.HOUSEKEEP = -1 * np.ones([CurSensorInfo.HOUSEKEEP[0].shape[0], 7]) # pred HOUSEKEEP
    StartNumericResult.HAULWINDOW = -1 * np.ones([NumHAULWINDOW, 7]) # 新增 HAULWINDOW， BUILDLADDER， STANDQUILT
    StartNumericResult.BUILDLADDER = -1 * np.ones([NumBUILDLADDER, 7])
    StartNumericResult.STANDQUILT = -1 * np.ones([NumSTANDQUILT, 7])
    StartNumericResult.HUMANINFO = np.array([]) # pred HUMANINFO
                                                       
    # return string result
    DetecterStrSensorInfo = RWSensorInfo(CurSensorInfo, 'W')
    DetecterStrDetectInfo = RWDetectInfo(StartNumericResult, 'W')
    DetecterStr = DetecterStrSensorInfo + DetecterStrDetectInfo
    
    # print result
    if False:
        TransConfigInfo = CurSensorInfo
        print('Sensor Information: ')
        print('isFlip: ' + str(TransConfigInfo.isFlip))
        print('H: \n' + str(TransConfigInfo.H))
        print('BED: \n' + str(TransConfigInfo.BED))
        print('INTERNALSUPERVISOR: \n' + str(TransConfigInfo.INTERNALSUPERVISOR))
        print('TOILET: \n' + str(TransConfigInfo.TOILET))
        print('WINDOW: \n' + str(TransConfigInfo.WINDOW))
        print('SLEEPTIME: \n' + str(TransConfigInfo.SLEEPTIME))
        print('HAULWINDOW: \n' + str(TransConfigInfo.HAULWINDOW))
        
    
    return DetecterStr

# --------------------------------------------------------------------
# 读写检测信息
#       配置文件信息：SensorInfo: RWSensorInfo(SensorInfoIn, DMode)
#       检测信息：DetectInfo: RWDetectInfo(DetectInfoIn, DMode)
# --------------------------------------------------------------------   
def RWSensorInfo(SensorInfoIn, DMode):
    # inputs: (1) 
    #       SensorInfoIn: string SensorInfo data
    #       DMode: 'R'
    # outputs:
    #       numeric data
    #      
    # inputs: (2)
    #       SensorInfoIn: numeric SensorInfo data
    #       DMode: 'W'
    # outputs:
    #       string data
    if DMode == 'R': # string to numeric
        result = SensorInfo()

        lines = SensorInfoIn.split('\n')
        
        ValueFlipFlag = []
        ValueH = []
        ValueBED = []
        ValueINTERNALSUPERVISOR = []
        ValueTOILET =[]
        ValueWINDOW = []
        ValueSLEEPTIME = []
        ValueABOVEHEIGHTTHOD = []
        ValueINTRIANGLEREGION = []
        ValueHOUSEKEEP = defaultdict(list)
        ValueHAULWINDOW = [] # 新增 HAULWINDOW， BUILDLADDER， STANDQUILT
        ValueBUILDLADDER = []
        ValueSTANDQUILT = []
        
        isFlipFlag = -1
        HFlag = -1
        BEDFlag = -1
        INTERNALSUPERVISORFlag = -1
        TOILETFlag = -1
        WINDOWFlag = -1
        SLEEPTIMEFlag = -1
        ABOVEHEIGHTTHODFlag = -1
        INTRIANGLEREGIONFlag = -1
        HOUSEKEEPFlag = -1
        HAULWINDOWFlag = -1
        BUILDLADDERFlag = -1
        STANDQUILTFlag = -1
        
        for line in lines:
            lineSp = line.split()
            if len(lineSp)>0 and lineSp[0]=='SensorInfo.isFlip':
                isFlipFlag = 1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.H':
                HFlag = 1
                isFlipFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.BED':
                BEDFlag = 1
                HFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.INTERNALSUPERVISOR':
                INTERNALSUPERVISORFlag = 1
                BEDFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.TOILET':
                TOILETFlag = 1
                INTERNALSUPERVISORFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.WINDOW':
                WINDOWFlag = 1
                TOILETFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.SLEEPTIME':
                SLEEPTIMEFlag = 1
                WINDOWFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.ABOVEHEIGHTTHOD':
                ABOVEHEIGHTTHODFlag = 1
                SLEEPTIMEFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.INTRIANGLEREGION':
                INTRIANGLEREGIONFlag = 1
                ABOVEHEIGHTTHODFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.HOUSEKEEP':
                HOUSEKEEPFlag = 1
                INTRIANGLEREGIONFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.HAULWINDOW':
                HAULWINDOWFlag = 1
                HOUSEKEEPFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.BUILDLADDER':
                BUILDLADDERFlag = 1
                HAULWINDOWFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='SensorInfo.STANDQUILT':
                STANDQUILTFlag = 1
                BUILDLADDERFlag = -1
                continue
            
            if len(lineSp)>0 and isFlipFlag == 1:
                for lineSpT in lineSp:
                    ValueFlipFlag.append(float(lineSpT))
            elif len(lineSp)>0 and HFlag == 1:
                for lineSpT in lineSp:
                    ValueH.append(float(lineSpT))
            elif len(lineSp)>0 and BEDFlag == 1:
                for lineSpT in lineSp:
                    ValueBED.append(float(lineSpT))
            elif len(lineSp)>0 and INTERNALSUPERVISORFlag == 1:
                for lineSpT in lineSp:
                    ValueINTERNALSUPERVISOR.append(float(lineSpT))
            elif len(lineSp)>0 and TOILETFlag == 1:
                for lineSpT in lineSp:
                    ValueTOILET.append(float(lineSpT))
            elif len(lineSp)>0 and WINDOWFlag == 1:
                for lineSpT in lineSp:
                    ValueWINDOW.append(float(lineSpT))
            elif len(lineSp)>0 and SLEEPTIMEFlag == 1:
                for lineSpT in lineSp:
                    ValueSLEEPTIME.append(float(lineSpT))
            elif len(lineSp)>0 and ABOVEHEIGHTTHODFlag == 1:
                for lineSpT in lineSp:
                    ValueABOVEHEIGHTTHOD.append(float(lineSpT))
            elif len(lineSp)>0 and INTRIANGLEREGIONFlag == 1:
                for lineSpT in lineSp:
                    ValueINTRIANGLEREGION.append(float(lineSpT))
            elif len(lineSp)>0 and HOUSEKEEPFlag == 1:
                for lineSpT in lineSp:
                    if len(lineSp) == 4: # HOUSEKEEP Time
                        ValueHOUSEKEEP[1].append(float(lineSpT))   
                    else: # HOUSEKEEP Areas
                        ValueHOUSEKEEP[0].append(float(lineSpT))
            elif len(lineSp)>0 and HAULWINDOWFlag == 1:
                for lineSpT in lineSp:
                    ValueHAULWINDOW.append(float(lineSpT))
            elif len(lineSp)>0 and BUILDLADDERFlag == 1:
                for lineSpT in lineSp:
                    ValueBUILDLADDER.append(float(lineSpT))
            elif len(lineSp)>0 and STANDQUILTFlag == 1:
                for lineSpT in lineSp:
                    ValueSTANDQUILT.append(float(lineSpT))   
            
        result.isFlip = ValueFlipFlag
        result.H = np.reshape(ValueH,[4,4])
        result.BED = np.reshape(ValueBED,[int(len(ValueBED)/7),7]) # 添加检测区域标记
        result.INTERNALSUPERVISOR = np.reshape(ValueINTERNALSUPERVISOR,[int(len(ValueINTERNALSUPERVISOR)/11),11]) # [x1,y1,x2,y2,x3,y3,x4,y4,z1,z2,validflag] 
        result.TOILET = np.reshape(ValueTOILET,[int(len(ValueTOILET)/5),5])
        result.WINDOW = np.reshape(ValueWINDOW,[int(len(ValueWINDOW)/5),5])
        result.SLEEPTIME = np.reshape(ValueSLEEPTIME,[int(len(ValueSLEEPTIME)/4),4])
        result.ABOVEHEIGHTTHOD = np.array(ValueABOVEHEIGHTTHOD)
        result.INTRIANGLEREGION = np.reshape(ValueINTRIANGLEREGION,[int(len(ValueINTRIANGLEREGION)/11),11]) # [x1,y1,x2,y2,x3,y3,x4,y4,z1,z2,validflag] 
        ValueHOUSEKEEP[1] = np.reshape(ValueHOUSEKEEP[1],[int(len(ValueHOUSEKEEP[1])/4),4]) # HOUSEKEEP Time
        ValueHOUSEKEEP[0] = np.reshape(ValueHOUSEKEEP[0],[int(len(ValueHOUSEKEEP[0])/7),7]) # HOUSEKEEP Areas, [xmin, xmax, ymin, ymax, zmin, zmax, area_direction]
        result.HOUSEKEEP = ValueHOUSEKEEP
        result.HAULWINDOW = np.reshape(ValueHAULWINDOW,[int(len(ValueHAULWINDOW)/7),7])
        result.BUILDLADDER = np.reshape(ValueBUILDLADDER,[int(len(ValueBUILDLADDER)/7),7])
        result.STANDQUILT = np.reshape(ValueSTANDQUILT,[int(len(ValueSTANDQUILT)/7),7])

            
    elif DMode == 'W': # numeric to string
        StrTemp = ''
        StrTemp = StrTemp + 'SensorInfo.isFlip\n' # isFlip
        StrTemp = StrTemp + str(SensorInfoIn.isFlip[0]) + '\n'
        StrTemp = StrTemp + 'SensorInfo.H\n' # H
        for i in range(SensorInfoIn.H.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.H[i][0]) + ' ' + str(SensorInfoIn.H[i][1]) + ' ' + str(SensorInfoIn.H[i][2]) + ' ' + str(SensorInfoIn.H[i][3]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.BED\n' # BED
        for i in range(SensorInfoIn.BED.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.BED[i][0]) + ' ' + str(SensorInfoIn.BED[i][1]) + ' ' + str(SensorInfoIn.BED[i][2]) + ' ' + str(SensorInfoIn.BED[i][3]) + ' ' + str(SensorInfoIn.BED[i][4]) + ' ' + str(SensorInfoIn.BED[i][5]) + ' ' + str(SensorInfoIn.BED[i][6])+'\n'
        StrTemp = StrTemp + 'SensorInfo.INTERNALSUPERVISOR\n' # INTERNALSUPERVISOR 
        for i in range(SensorInfoIn.INTERNALSUPERVISOR.shape[0]): 
            StrTemp = StrTemp + str(SensorInfoIn.INTERNALSUPERVISOR[i][0]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][1]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][2]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][3])+ ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][4])\
                                    + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][5]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][6]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][7]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][8]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][9]) + ' ' + str(SensorInfoIn.INTERNALSUPERVISOR[i][10]) + '\n'
        StrTemp = StrTemp + 'SensorInfo.TOILET\n' # TOILET
        for i in range(SensorInfoIn.TOILET.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.TOILET[i][0]) + ' ' + str(SensorInfoIn.TOILET[i][1]) + ' ' + str(SensorInfoIn.TOILET[i][2]) + ' ' + str(SensorInfoIn.TOILET[i][3])+ ' ' + str(SensorInfoIn.TOILET[i][4]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.WINDOW\n' # WINDOW
        for i in range(SensorInfoIn.WINDOW.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.WINDOW[i][0]) + ' ' + str(SensorInfoIn.WINDOW[i][1]) + ' ' + str(SensorInfoIn.WINDOW[i][2]) + ' ' + str(SensorInfoIn.WINDOW[i][3])+ ' ' + str(SensorInfoIn.WINDOW[i][4]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.SLEEPTIME\n' # SLEEPTIME
        for i in range(SensorInfoIn.SLEEPTIME.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.SLEEPTIME[i][0]) + ' ' + str(SensorInfoIn.SLEEPTIME[i][1]) + ' ' + str(SensorInfoIn.SLEEPTIME[i][2]) + ' ' + str(SensorInfoIn.SLEEPTIME[i][3]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.ABOVEHEIGHTTHOD\n' # ABOVEHEIGHTTHOD
        for i in range(SensorInfoIn.ABOVEHEIGHTTHOD.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.ABOVEHEIGHTTHOD[i])+'\n'
        StrTemp = StrTemp + 'SensorInfo.INTRIANGLEREGION\n' # INTRIANGLEREGION
        for i in range(SensorInfoIn.INTRIANGLEREGION.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.INTRIANGLEREGION[i][0]) + ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][1]) + ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][2]) + ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][3])+ ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][4])\
                                    + ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][5])+ ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][6])+ ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][7])+ ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][8])+ ' ' + str(SensorInfoIn.INTRIANGLEREGION[i][9]) +' ' + str(SensorInfoIn.INTRIANGLEREGION[i][10]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.HOUSEKEEP\n' # HOUSEKEEP
        for i in range(len(SensorInfoIn.HOUSEKEEP)):
            for j in range(SensorInfoIn.HOUSEKEEP[i].shape[0]):
                for k in range(SensorInfoIn.HOUSEKEEP[i][j].shape[0]):
                    StrTemp = StrTemp + str(SensorInfoIn.HOUSEKEEP[i][j][k]) + ' '
                StrTemp = StrTemp +'\n'
        StrTemp = StrTemp + 'SensorInfo.HAULWINDOW\n' # HAULWINDOW
        for i in range(SensorInfoIn.HAULWINDOW.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.HAULWINDOW[i][0]) + ' ' + str(SensorInfoIn.HAULWINDOW[i][1]) + ' ' + str(SensorInfoIn.HAULWINDOW[i][2]) + ' ' + str(SensorInfoIn.HAULWINDOW[i][3])+ ' ' + str(SensorInfoIn.HAULWINDOW[i][4]) + ' ' + str(SensorInfoIn.HAULWINDOW[i][5]) + ' ' + str(SensorInfoIn.HAULWINDOW[i][6]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.BUILDLADDER\n' # BUILDLADDER
        for i in range(SensorInfoIn.BUILDLADDER.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.BUILDLADDER[i][0]) + ' ' + str(SensorInfoIn.BUILDLADDER[i][1]) + ' ' + str(SensorInfoIn.BUILDLADDER[i][2]) + ' ' + str(SensorInfoIn.BUILDLADDER[i][3])+ ' ' + str(SensorInfoIn.BUILDLADDER[i][4]) + ' ' + str(SensorInfoIn.BUILDLADDER[i][5]) + ' ' + str(SensorInfoIn.BUILDLADDER[i][6]) +'\n'
        StrTemp = StrTemp + 'SensorInfo.STANDQUILT\n' # STANDQUILT
        for i in range(SensorInfoIn.STANDQUILT.shape[0]):
            StrTemp = StrTemp + str(SensorInfoIn.STANDQUILT[i][0]) + ' ' + str(SensorInfoIn.STANDQUILT[i][1]) + ' ' + str(SensorInfoIn.STANDQUILT[i][2]) + ' ' + str(SensorInfoIn.STANDQUILT[i][3])+ ' ' + str(SensorInfoIn.STANDQUILT[i][4]) + ' ' + str(SensorInfoIn.STANDQUILT[i][5]) + ' ' + str(SensorInfoIn.STANDQUILT[i][6]) +'\n'

                
        result = StrTemp

    return result
    
def RWDetectInfo(DetectInfoIn, DMode):
    # inputs: (1) 
    #       SensorInfoIn: string SensorInfo data
    #       DMode: 'R'
    # outputs:
    #       numeric data
    #      
    # inputs: (2)
    #       SensorInfoIn: numeric SensorInfo data
    #       DMode: 'W'
    # outputs:
    #       string data
    if DMode == 'R': # string to numeric
        result = DetectInfo()

        lines = DetectInfoIn.split('\n')
        
        ValueBED = []
        ValueINTERNALSUPERVISOR = []
        ValueTOILET =[]
        ValueWINDOW = []
        ValueABOVEHEIGHT = []
        ValueWRONGTIMELIE = []
        ValueINTRIANGLEREGION = []
        ValueTOILET_STARTTIME = []
        ValueWINDOW_STARTTIME = []
        ValueHOUSEKEEP = [] # HOUSEKEEP
        ValueHAULWINDOW = [] # 新增 HAULWINDOW， BUILDLADDER， STANDQUILT
        ValueBUILDLADDER = []
        ValueSTANDQUILT= []
        ValueHUMANINFO = [] # HUMANINFO
        
        BEDFlag = -1
        INTERNALSUPERVISORFlag = -1
        TOILETFlag = -1
        WINDOWFlag = -1
        ABOVEHEIGHTFlag = -1
        WRONGTIMELIEFlag = -1
        INTRIANGLEREGIONFlag = -1
        TOILET_STARTTIMEFlag = -1
        WINDOW_STARTTIMEFlag = -1
        HOUSEKEEPFlag = -1
        HAULWINDOWFlag = -1 
        BUILDLADDERFlag = -1
        STANDQUILTFlag = -1
        HUMANINFOFlag = -1
        
        for line in lines:
            lineSp = line.split()
            if len(lineSp)>0 and lineSp[0]=='DetectInfo.BED':
                BEDFlag = 1
                HFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.INTERNALSUPERVISOR':
                INTERNALSUPERVISORFlag = 1
                BEDFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.TOILET':
                TOILETFlag = 1
                INTERNALSUPERVISORFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.WINDOW':
                WINDOWFlag = 1
                TOILETFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.ABOVEHEIGHT':
                ABOVEHEIGHTFlag = 1
                WINDOWFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.WRONGTIMELIE':
                WRONGTIMELIEFlag = 1
                ABOVEHEIGHTFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.INTRIANGLEREGION':
                INTRIANGLEREGIONFlag = 1
                WRONGTIMELIEFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.TOILET_STARTTIME':
                TOILET_STARTTIMEFlag = 1
                INTRIANGLEREGIONFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.WINDOW_STARTTIME':
                WINDOW_STARTTIMEFlag = 1
                TOILET_STARTTIMEFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.HOUSEKEEP':
                HOUSEKEEPFlag = 1
                WINDOW_STARTTIMEFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.HAULWINDOW':
                HAULWINDOWFlag = 1
                HOUSEKEEPFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.BUILDLADDER':
                BUILDLADDERFlag = 1
                HAULWINDOWFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.STANDQUILT':
                STANDQUILTFlag = 1
                BUILDLADDERFlag = -1
                continue
            elif len(lineSp)>0 and lineSp[0]=='DetectInfo.HUMANINFO':
                HUMANINFOFlag = 1
                STANDQUILTFlag = -1
                continue
            
            if len(lineSp)>0 and BEDFlag == 1:
                for lineSpT in lineSp:
                    ValueBED.append(float(lineSpT))
            elif len(lineSp)>0 and INTERNALSUPERVISORFlag == 1:
                for lineSpT in lineSp:
                    ValueINTERNALSUPERVISOR.append(float(lineSpT))
            elif len(lineSp)>0 and TOILETFlag == 1:
                for lineSpT in lineSp:
                    ValueTOILET.append(float(lineSpT))
            elif len(lineSp)>0 and WINDOWFlag == 1:
                for lineSpT in lineSp:
                    ValueWINDOW.append(float(lineSpT))
            elif len(lineSp)>0 and ABOVEHEIGHTFlag == 1:
                for lineSpT in lineSp:
                    ValueABOVEHEIGHT.append(float(lineSpT))
            elif len(lineSp)>0 and WRONGTIMELIEFlag == 1:
                for lineSpT in lineSp:
                    ValueWRONGTIMELIE.append(float(lineSpT))
            elif len(lineSp)>0 and INTRIANGLEREGIONFlag == 1:
                for lineSpT in lineSp:
                    ValueINTRIANGLEREGION.append(float(lineSpT))
            elif len(lineSp)>0 and TOILET_STARTTIMEFlag == 1:
                for lineSpT in lineSp:
                    ValueTOILET_STARTTIME.append(float(lineSpT))
            elif len(lineSp)>0 and WINDOW_STARTTIMEFlag == 1:
                for lineSpT in lineSp:
                    ValueWINDOW_STARTTIME.append(float(lineSpT))
            elif len(lineSp)>0 and HOUSEKEEPFlag == 1:
                for lineSpT in lineSp:
                    ValueHOUSEKEEP.append(float(lineSpT))
            elif len(lineSp)>0 and HAULWINDOWFlag == 1:
                for lineSpT in lineSp:
                    ValueHAULWINDOW.append(float(lineSpT))
            elif len(lineSp)>0 and BUILDLADDERFlag == 1:
                for lineSpT in lineSp:
                    ValueBUILDLADDER.append(float(lineSpT))
            elif len(lineSp)>0 and STANDQUILTFlag == 1:
                for lineSpT in lineSp:
                    ValueSTANDQUILT.append(float(lineSpT))
            elif len(lineSp)>0 and HUMANINFOFlag == 1:
                for lineSpT in lineSp:
                    ValueHUMANINFO.append(float(lineSpT))            
                    
        result.BED = np.reshape(ValueBED,[int(len(ValueBED)/7),7])
        result.INTERNALSUPERVISOR = np.reshape(ValueINTERNALSUPERVISOR,[int(len(ValueINTERNALSUPERVISOR)/7),7])
        result.TOILET = np.reshape(ValueTOILET,[int(len(ValueTOILET)/7),7])
        result.WINDOW = np.reshape(ValueWINDOW,[int(len(ValueWINDOW)/7),7])
        result.ABOVEHEIGHT = np.reshape(ValueABOVEHEIGHT,[int(len(ValueABOVEHEIGHT)/7),7])
        result.WRONGTIMELIE = np.reshape(ValueWRONGTIMELIE,[int(len(ValueWRONGTIMELIE)/7),7])
        result.INTRIANGLEREGION = np.reshape(ValueINTRIANGLEREGION,[int(len(ValueINTRIANGLEREGION)/7),7])
        result.TOILET_STARTTIME = np.reshape(ValueTOILET_STARTTIME,[int(len(ValueTOILET_STARTTIME)/2),2])
        result.WINDOW_STARTTIME = np.reshape(ValueWINDOW_STARTTIME,[int(len(ValueWINDOW_STARTTIME)/2),2])
        result.HOUSEKEEP = np.reshape(ValueHOUSEKEEP,[int(len(ValueHOUSEKEEP)/7),7]) # HOUSEKEEP [N x 7]
        result.HAULWINDOW = np.reshape(ValueHAULWINDOW,[int(len(ValueHAULWINDOW)/7),7]) # HAULWINDOW [N x 7]
        result.BUILDLADDER = np.reshape(ValueBUILDLADDER,[int(len(ValueBUILDLADDER)/7),7]) # BUILDLADDER [N x 7]
        result.STANDQUILT = np.reshape(ValueSTANDQUILT,[int(len(ValueSTANDQUILT)/7),7]) # STANDQUILT [N x 7]
        result.HUMANINFO = np.reshape(ValueHUMANINFO,[int(len(ValueHUMANINFO)/7),7]) # HUMANINFO [N x 7]
            
    elif DMode == 'W': # numeric to string
        StrTemp = ''
        StrTemp = StrTemp + 'DetectInfo.BED\n' # BED
        for i in range(DetectInfoIn.BED.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.BED[i][0]) + ' ' + str(DetectInfoIn.BED[i][1]) + ' ' + str(DetectInfoIn.BED[i][2]) + ' ' + str(round(DetectInfoIn.BED[i][3],3)) + ' ' \
                                    + str(round(DetectInfoIn.BED[i][4],3)) + ' ' + str(round(DetectInfoIn.BED[i][5],3))+ ' ' + str(DetectInfoIn.BED[i][6]) +'\n'
        StrTemp = StrTemp + 'DetectInfo.INTERNALSUPERVISOR\n' # INTERNALSUPERVISOR 
        for i in range(DetectInfoIn.INTERNALSUPERVISOR.shape[0]): 
            StrTemp = StrTemp + str(DetectInfoIn.INTERNALSUPERVISOR[i][0]) + ' ' + str(DetectInfoIn.INTERNALSUPERVISOR[i][1]) + ' ' + str(DetectInfoIn.INTERNALSUPERVISOR[i][2]) + ' ' + str(round(DetectInfoIn.INTERNALSUPERVISOR[i][3],3))+ ' ' \
                                    + str(round(DetectInfoIn.INTERNALSUPERVISOR[i][4],3)) + ' ' + str(round(DetectInfoIn.INTERNALSUPERVISOR[i][5],3))+ ' ' + str(DetectInfoIn.INTERNALSUPERVISOR[i][6])+ '\n'
        StrTemp = StrTemp + 'DetectInfo.TOILET\n' # TOILET
        for i in range(DetectInfoIn.TOILET.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.TOILET[i][0]) + ' ' + str(DetectInfoIn.TOILET[i][1]) + ' ' + str(DetectInfoIn.TOILET[i][2]) + ' ' + str(round(DetectInfoIn.TOILET[i][3],3)) + ' ' + \
                                    str(round(DetectInfoIn.TOILET[i][4],3)) + ' ' + str(round(DetectInfoIn.TOILET[i][5],3)) + ' ' +str(DetectInfoIn.TOILET[i][6]) +'\n'
        StrTemp = StrTemp + 'DetectInfo.WINDOW\n' # WINDOW
        for i in range(DetectInfoIn.WINDOW.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.WINDOW[i][0]) + ' ' + str(DetectInfoIn.WINDOW[i][1]) + ' ' + str(DetectInfoIn.WINDOW[i][2]) + ' ' + str(round(DetectInfoIn.WINDOW[i][3],3))+ ' ' \
                                    + str(round(DetectInfoIn.WINDOW[i][4],3))+ ' ' + str(round(DetectInfoIn.WINDOW[i][5],3))+ ' ' + str(DetectInfoIn.WINDOW[i][6]) +'\n'
        StrTemp = StrTemp + 'DetectInfo.ABOVEHEIGHT\n' # ABOVEHEIGHT
        if DetectInfoIn.ABOVEHEIGHT.shape[0] > 0:
            for i in range(DetectInfoIn.ABOVEHEIGHT.shape[0]):
                StrTemp = StrTemp + str(DetectInfoIn.ABOVEHEIGHT[i][0]) + ' ' + str(DetectInfoIn.ABOVEHEIGHT[i][1]) + ' ' + str(DetectInfoIn.ABOVEHEIGHT[i][2]) + ' ' + str(round(DetectInfoIn.ABOVEHEIGHT[i][3],3)) + ' ' \
                                        + str(round(DetectInfoIn.ABOVEHEIGHT[i][4],3)) + ' ' + str(round(DetectInfoIn.ABOVEHEIGHT[i][5],3)) + ' ' + str(DetectInfoIn.ABOVEHEIGHT[i][6]) +'\n'
        StrTemp = StrTemp + 'DetectInfo.WRONGTIMELIE\n' # WRONGTIMELIE
        if DetectInfoIn.WRONGTIMELIE.shape[0] > 0:
            for i in range(DetectInfoIn.WRONGTIMELIE.shape[0]):
                StrTemp = StrTemp + str(DetectInfoIn.WRONGTIMELIE[i][0]) + ' ' + str(DetectInfoIn.WRONGTIMELIE[i][1]) + ' ' + str(DetectInfoIn.WRONGTIMELIE[i][2]) + ' ' + str(round(DetectInfoIn.WRONGTIMELIE[i][3],3)) + ' ' \
                                        + str(round(DetectInfoIn.WRONGTIMELIE[i][4],3)) + ' ' + str(round(DetectInfoIn.WRONGTIMELIE[i][5],3)) + ' ' + str(DetectInfoIn.WRONGTIMELIE[i][6]) +'\n'
        StrTemp = StrTemp + 'DetectInfo.INTRIANGLEREGION\n' # INTRIANGLEREGION
        if DetectInfoIn.INTRIANGLEREGION.shape[0] > 0:
            for i in range(DetectInfoIn.INTRIANGLEREGION.shape[0]):
                StrTemp = StrTemp + str(DetectInfoIn.INTRIANGLEREGION[i][0]) + ' ' + str(DetectInfoIn.INTRIANGLEREGION[i][1]) + ' ' + str(DetectInfoIn.INTRIANGLEREGION[i][2]) + ' ' + str(round(DetectInfoIn.INTRIANGLEREGION[i][3],3)) + ' ' \
                                        + str(round(DetectInfoIn.INTRIANGLEREGION[i][4],3)) + ' ' + str(round(DetectInfoIn.INTRIANGLEREGION[i][5],3)) + ' ' + str(DetectInfoIn.INTRIANGLEREGION[i][6]) +'\n'

        StrTemp = StrTemp + 'DetectInfo.TOILET_STARTTIME\n' # TOILET_STARTTIME
        for i in range(DetectInfoIn.TOILET_STARTTIME.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.TOILET_STARTTIME[i][0]) + ' ' + str(DetectInfoIn.TOILET_STARTTIME[i][1]) + '\n'
        StrTemp = StrTemp + 'DetectInfo.WINDOW_STARTTIME\n' # WINDOW_STARTTIME
        for i in range(DetectInfoIn.WINDOW_STARTTIME.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.WINDOW_STARTTIME[i][0]) + ' ' + str(DetectInfoIn.WINDOW_STARTTIME[i][1]) + '\n'
        
        StrTemp = StrTemp + 'DetectInfo.HOUSEKEEP\n' # HOUSEKEEP
        if DetectInfoIn.HOUSEKEEP.shape[0] > 0:
            for i in range(DetectInfoIn.HOUSEKEEP.shape[0]):
                StrTemp = StrTemp + str(DetectInfoIn.HOUSEKEEP[i][0]) + ' ' + str(DetectInfoIn.HOUSEKEEP[i][1]) + ' ' + str(DetectInfoIn.HOUSEKEEP[i][2]) + ' ' + str(round(DetectInfoIn.HOUSEKEEP[i][3],3)) + ' ' \
                                        + str(round(DetectInfoIn.HOUSEKEEP[i][4],3)) + ' ' + str(round(DetectInfoIn.HOUSEKEEP[i][5],3)) + ' ' + str(DetectInfoIn.HOUSEKEEP[i][6]) +'\n'
        
        StrTemp = StrTemp + 'DetectInfo.HAULWINDOW\n' # HAULWINDOW
        for i in range(DetectInfoIn.HAULWINDOW.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.HAULWINDOW[i][0]) + ' ' + str(DetectInfoIn.HAULWINDOW[i][1]) + ' ' + str(DetectInfoIn.HAULWINDOW[i][2]) + ' ' + str(round(DetectInfoIn.HAULWINDOW[i][3],3)) + ' ' \
                                    + str(round(DetectInfoIn.HAULWINDOW[i][4],3)) + ' ' + str(round(DetectInfoIn.HAULWINDOW[i][5],3)) + ' ' + str(DetectInfoIn.HAULWINDOW[i][6]) + '\n'

        StrTemp = StrTemp + 'DetectInfo.BUILDLADDER\n' # BUILDLADDER
        for i in range(DetectInfoIn.BUILDLADDER.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.BUILDLADDER[i][0]) + ' ' + str(DetectInfoIn.BUILDLADDER[i][1]) + ' ' + str(DetectInfoIn.BUILDLADDER[i][2]) + ' ' + str(round(DetectInfoIn.BUILDLADDER[i][3],3)) + ' ' \
                                    + str(round(DetectInfoIn.BUILDLADDER[i][4],3)) + ' ' + str(round(DetectInfoIn.BUILDLADDER[i][5],3)) + ' ' + str(DetectInfoIn.BUILDLADDER[i][6]) + '\n'

        StrTemp = StrTemp + 'DetectInfo.STANDQUILT\n' # STANDQUILT
        for i in range(DetectInfoIn.STANDQUILT.shape[0]):
            StrTemp = StrTemp + str(DetectInfoIn.STANDQUILT[i][0]) + ' ' + str(DetectInfoIn.STANDQUILT[i][1]) + ' ' + str(DetectInfoIn.STANDQUILT[i][2]) + ' ' + str(round(DetectInfoIn.STANDQUILT[i][3],3)) + ' ' \
                                    + str(round(DetectInfoIn.STANDQUILT[i][4],3)) + ' ' + str(round(DetectInfoIn.STANDQUILT[i][5],3)) + ' ' + str(DetectInfoIn.STANDQUILT[i][6]) + '\n'
                  
        StrTemp = StrTemp + 'DetectInfo.HUMANINFO\n' # HUMANINFO
        if DetectInfoIn.HUMANINFO.shape[0] > 0:
            for i in range(DetectInfoIn.HUMANINFO.shape[0]):
                StrTemp = StrTemp + str(DetectInfoIn.HUMANINFO[i][0]) + ' ' + str(DetectInfoIn.HUMANINFO[i][1]) + ' ' + str(DetectInfoIn.HUMANINFO[i][2]) + ' ' + str(round(DetectInfoIn.HUMANINFO[i][3],3)) + ' ' \
                                        + str(round(DetectInfoIn.HUMANINFO[i][4],3)) + ' ' + str(round(DetectInfoIn.HUMANINFO[i][5],3)) + ' ' + str(DetectInfoIn.HUMANINFO[i][6]) +'\n'
        
        result = StrTemp   
                           
    return result
    
    
def DeteInfoNumericToStr(SensorInfoNumeric, AlarmDeteInfoNumeric):
    """
    功能：检测信息：数值形式转换为str形式
    """
    DetecterStrSensorInfo = RWSensorInfo(SensorInfoNumeric, 'W')
    DetecterStrDetectInfo = RWDetectInfo(AlarmDeteInfoNumeric, 'W')
    resultString = DetecterStrSensorInfo + DetecterStrDetectInfo
    
    return resultString
    
    
def DeteInfoStrToNumeric(DeteInfoStr):
    """
    功能：检测信息：str形式转换为数值形式
    """
    SensorInfoStr = ''
    NumericResultStr = ''
    SensorInfoStrFlag = -1
    lines = DeteInfoStr.split('\n')
    for line in lines:
        if len(line)>0:
            if (line == 'DetectInfo.BED'):
                SensorInfoStrFlag = 1
            if SensorInfoStrFlag == -1:
                SensorInfoStr = SensorInfoStr + line + '\n'
            else:
                NumericResultStr = NumericResultStr + line + '\n' 
    # read data from string to numeric, SensorInfo/DetectInfo
    CurSensorInfo = RWSensorInfo(SensorInfoStr, 'R')
    AlarmDeteInfo = RWDetectInfo(NumericResultStr, 'R')
    
    return CurSensorInfo, AlarmDeteInfo
    
    
    