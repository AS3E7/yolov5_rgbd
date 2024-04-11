# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 16:40:24 2020

@author: HYD
"""

def CalInValidTime(CurTime, PeriodTime, NearMintuePeriodTime):
    """
    功能：判断当前时间是否在规定的时间段内
    输入：
        CurTime：当前时刻，
        PeriodTime：多个检测时间段，如：[[22,00,23,59],[00,00,07,00],[12,00,14,00]]
        NearMintuePeriodTime: 邻近休息时间段前后 10 min 为无效时间段
    """
    # 当前时刻
    WTimeMintue = (CurTime%(24*60*60)/60 + 8*60)%(24*60) # WTime Mintues, init: WTime%(24*60*60)/60 + 8*60
    # 判断是否在多个时间段内
    InPeriodTimeFlag = -1 # 是否存在多个时间段内, 
                         # if InSLEEPTIMEFlag == -1, 不在选择的时间段
                         # if InSLEEPTIMEFlag == 0, 无效时间段
                         # if InSLEEPTIMEFlag == 1, 在选择的时间段内
    for n_time in range(PeriodTime.shape[0]): # sleep time
        if (PeriodTime[n_time][2]*60 + PeriodTime[n_time][3]) - (PeriodTime[n_time][0]*60 + PeriodTime[n_time][1]) < 0: # 判断时间段是否跨天, eg:[22:30 - 07:30]
            # 跨天休息时间段
            if InSelectTime(WTimeMintue, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] + NearMintuePeriodTime, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] - NearMintuePeriodTime) == 1:
                InPeriodTimeFlag = 1
            elif InSelectTime(WTimeMintue, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] - NearMintuePeriodTime, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] + NearMintuePeriodTime)==1 \
                               or InSelectTime(WTimeMintue, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] - NearMintuePeriodTime, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] + NearMintuePeriodTime)==1:
                InPeriodTimeFlag = 0
 
        else: # 非跨天休息时间段, eg: [12:00 14:00]
            if InSelectTime(WTimeMintue, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] + NearMintuePeriodTime, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] - NearMintuePeriodTime)==1:
                InPeriodTimeFlag = 1
            elif InSelectTime(WTimeMintue, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] - NearMintuePeriodTime, PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] + NearMintuePeriodTime)==1 \
                               or InSelectTime(WTimeMintue, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] - NearMintuePeriodTime, PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] + NearMintuePeriodTime)==1:
                InPeriodTimeFlag = 0
            # 增加排除，输入数据增加缓冲时间后出现问题
            if PeriodTime[n_time][0]*60 + PeriodTime[n_time][1] + NearMintuePeriodTime > PeriodTime[n_time][2]*60 + PeriodTime[n_time][3] - NearMintuePeriodTime:
                InPeriodTimeFlag = 0
    return InPeriodTimeFlag

# --------------------------------------------------------------------
# 判断当前时刻是否在某一时间段内
#       InSelectTime(CTimeIn, PeriodMinIn, PeriodMaxIn)
# --------------------------------------------------------------------
def InSelectTime(CTimeIn, PeriodMinIn, PeriodMaxIn):
    # 输入：
    #   CTime：当前分钟时间，最大24*60
    # 输出：
    #   PeriodMin：起始时间
    #   PeriodMax：终止时间
    # 例子：
    #   23:30 in [23:00 - 00:05]
    
    # CTime
    if CTimeIn > 24*60:
        CTime = CTimeIn%(24*60)
    else:
        CTime = CTimeIn
    
    # init InSelTimeFlag
    InSelTimeFlag = -1
    
    # PeriodMin <-- PeriodMinIn
    if PeriodMinIn >= 24*60:
        PeriodMin = PeriodMinIn%(24*60)
    elif PeriodMinIn<0:
        PeriodMin = 24*60+PeriodMinIn
    else:
        PeriodMin = PeriodMinIn
    # PeriodMax <-- PeriodMaxIn
    if PeriodMaxIn >= 24*60:
        PeriodMax = PeriodMaxIn%(24*60)
    elif PeriodMaxIn<0:
        PeriodMax = 24*60+PeriodMaxIn
    else:
        PeriodMax = PeriodMaxIn
    
    # period time
    if PeriodMin <= PeriodMax:
        if CTime>=PeriodMin and CTime<PeriodMax:
            InSelTimeFlag = 1
        else:
            InSelTimeFlag = -1
    if PeriodMin > PeriodMax:
        if CTime>=PeriodMin or CTime<PeriodMax:
            InSelTimeFlag = 1
        else:
            InSelTimeFlag = -1

    return InSelTimeFlag