# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:30:17 2019

@author: HYD
"""

import numpy as np 
import time
import os

import logging as lgmsg
import re
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

from detect_alarm.GlobalInfoDete import GlobalInfo
from util.ReadConfig import ReadSensorConfig, ReadIniConfigGlobalInfo

from util.SaveData import DeleteOutLimitFiles, LogSaveCategoryIndexFun
from config import one_sensor_params, online_offline_type, log_info, alarm_level

##############################################################################
# log 信息
#    告警信息打印类型
#       处理过程打印信息：配置文件读取，背景文件读取，接口输入信息，接口输出信息
#       目标检测过程打印信息：目标检测框位置/score，
#       聚类目标检测打印信息：聚类目标信息
#       告警检测打印信息：各类告警信息
#   格式：[0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4 = 打印各类告警检测信息]
##############################################################################
ConfigDebugInfo = log_info['debug_info']
ConfigDebugTypeIndex = LogSaveCategoryIndexFun()
ConfigSaveDetectBboxImageFlag = log_info['save_detect_bbox_image_flag']
ConfigSaveInputPtsFlag = log_info['save_input_pts_flag']


#####################################################################
# 在线/离线文件地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SensorConfigFileName = './demo/SensorConfig.ini'
    SensorConfigReviseFileName = "./demo/log/FileReviseTime.log"
else:
    SensorConfigFileName = 'SensorConfig.ini'
    SensorConfigReviseFileName = "log/FileReviseTime.log"
##################################################################### 


# 保存打印结果
if online_offline_type == 'OnLine':
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
        # 打印文件保存地址
        SaveLogFolderName = os.path.join('demo', 'mylog')
        #日志打印格式
        log_fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = lgmsg.Formatter(log_fmt)
        
        # 创建 log 文件
#        # 以时间创建文件
#        NumLogFile = 7 # 7天
#        file_name = 'msg_log'
#        file_full_name = os.path.join(SaveLogFolderName,file_name)
#        log_file_handler = TimedRotatingFileHandler(filename=file_full_name, when="D", interval=1, backupCount=NumLogFile)
        # 以大小创建文件
        EachLogFileSize = 300 * 1024 * 1024 # Byte
        NumLogFile = 10
        file_name = 'msg_log'
        file_full_name = os.path.join(SaveLogFolderName,file_name)
        log_file_handler = RotatingFileHandler(filename=file_full_name, maxBytes=EachLogFileSize, backupCount=NumLogFile)
        
        log_file_handler.suffix = "%Y-%m-%d_%H-%M_%S.log"
        log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_\d{2}.log$")
        log_file_handler.setFormatter(formatter)

    #    logging.basicConfig(level=logging.DEBUG)
        lgmsg.root.setLevel(lgmsg.DEBUG)
        log = lgmsg.getLogger()
        log.addHandler(log_file_handler)
        
        # 开始打印信息
        lgmsg.info('开始导入算法模块')
else:
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
        # 打印文件保存地址
        SaveLogFolderName = os.path.join('mylog')
        #日志打印格式
        log_fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = lgmsg.Formatter(log_fmt)
        
        # 创建 log 文件
#        # 以时间创建文件
#        NumLogFile = 7 # 7天
#        file_name = 'msg_log'
#        file_full_name = os.path.join(SaveLogFolderName,file_name)
#        log_file_handler = TimedRotatingFileHandler(filename=file_full_name, when="D", interval=1, backupCount=NumLogFile)
        # 以大小创建文件
        EachLogFileSize = 300 * 1024 * 1024 # Byte
        NumLogFile = 10
        file_name = 'msg_log'
        file_full_name = os.path.join(SaveLogFolderName,file_name)
        log_file_handler = RotatingFileHandler(filename=file_full_name, maxBytes=EachLogFileSize, backupCount=NumLogFile)
        
        log_file_handler.suffix = "%Y-%m-%d_%H-%M_%S.log"
        log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_\d{2}.log$")
        log_file_handler.setFormatter(formatter)

    #    logging.basicConfig(level=logging.DEBUG)
        lgmsg.root.setLevel(lgmsg.DEBUG)
        log = lgmsg.getLogger()
        log.addHandler(log_file_handler)
        
        # 开始打印信息
        lgmsg.info('开始导入算法模块')

        
##############################################################################
# 读取配置文件
##############################################################################
if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
    # ReadIniConfigGlobalInfo
    SensorInfoAll = ReadIniConfigGlobalInfo(SensorConfigFileName)
    WarnLooseDegree = SensorInfoAll['ROOM_INFO']['managelevel']
    CurDeteAlarmLevelInfo = alarm_level[str(WarnLooseDegree)]
    # 打印配置文件读取信息
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0: # 打印配置文件读取信息
        lgmsg.info('第一次读取配置文件:SensorConfig.ini')
        lgmsg.info('    SensorConfig info: {}'.format(SensorInfoAll))
    # 告警松紧度等级信息
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
        lgmsg.info('第一次获取告警松紧度等级信息')
        lgmsg.info('CurDeteAlarmLevelInfo {} = {}'.format(WarnLooseDegree, CurDeteAlarmLevelInfo))
        

def callPython(CurHumanInfoStr, PreDeteSateStr):
    """
        服务器端接口程序:
        Inputs:
            CurHumanInfoStr: 当前帧人员目标位置, string
            PreDeteSateStr: 前一帧告警检测结果, string
                          
        Outputs:
            CurDeteStateStr: 当前帧告警检测结果, stirng 
                    每行信息: [告警类型，房间编号，告警次数，告警位置X，告警位置Y，告警位置Z，传感器编号，告警状态]
                        告警类型: 1 表示‘单人留仓’
                        房间编号: 0 为主仓，1 为副仓
                        告警次数: 连续告警次数
                        告警状态: 1 表示产生此类告警，否则,不产生此类告警
    """

#    print('====== Start callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        lgmsg.info('======  Start callPython ======')
    
    # 检测过程
    GlobalInfoFuns = GlobalInfo(CurHumanInfoStr)
    # CurHumanInfoStr -> CurHumanInfoNumeric
    CurRoomInfoGroup = GlobalInfoFuns.read_human_info_str()
    CurHumanInfoNumeric = CurRoomInfoGroup['HumanInfo']

    # PreDeteSateStr -> PreDeteSateNumeric
    PreDeteSateNumeric = GlobalInfo.trans_alarm_info(PreDeteSateStr, 'R')
    # 检测结果
#    print('CurHumanInfoNumeric = {}'.format(CurHumanInfoNumeric))
#    TempCurDeteInfo, TempCurDeteState = GlobalInfoFuns.detect_alarm(CurHumanInfoNumeric, PreDeteSateNumeric) # 检测当前目标结果
#    TempCurDeteState = GlobalInfoFuns.detect_alarm(CurHumanInfoNumeric, PreDeteSateNumeric) # 检测当前目标结果
    TempCurDeteState, _, _ = GlobalInfoFuns.detect_alarm(CurHumanInfoNumeric, PreDeteSateNumeric) # 检测当前目标结果

#    print('TempCurDeteState = {}'.format(TempCurDeteState))
    # TempCurDeteState -> CurDeteStateStr
    CurDeteStateStr = GlobalInfo.trans_alarm_info(TempCurDeteState, 'W')
    
    # 打印输入输出信息
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        # 输入信息
        lgmsg.debug('CurHumanInfoStr = {}'.format(CurHumanInfoStr)) # 多个传感器检测目标结果
#        lgmsg.debug('PreDeteSateStr = {}'.format(PreDeteSateStr)) # 上一帧的检测结果
        # 输出信息
        lgmsg.debug('CurDeteStateStr = {}'.format(CurDeteStateStr)) # 当前帧检测结果
    
#    print('====== End callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        lgmsg.info('======  End callPython ======')
    
    return CurDeteStateStr
    
if __name__=='__main__':
    print('Start.')
    
    