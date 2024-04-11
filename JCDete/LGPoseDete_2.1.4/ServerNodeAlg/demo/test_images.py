# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:02:40 2020

@author: HYD
"""

import numpy as np 
import time
import os

import logging as lgmsg
import re
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

from alg import callPython
from util.ReadConfig import ReadSensorConfig

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


"""
# 保存打印结果
if online_offline_type == 'OffLine':
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
         
    ##############################################################################
    # 读取配置文件
    ##############################################################################
    SensorInfoAll, SleepTime, INTERNALSUPERVISOR_HumanNum, WarnLooseDegree = ReadSensorConfig(SensorConfigFileName)
    # 打印配置文件读取信息
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0: # 打印配置文件读取信息
        lgmsg.info('第一次读取配置文件:SensorConfig.ini')
        lgmsg.info('    SensorConfig info: {}'.format(SensorInfoAll))
    # 告警松紧度等级信息
    CurDeteAlarmLevelInfo = alarm_level[str(WarnLooseDegree)]
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
        lgmsg.info('第一次获取告警松紧度等级信息')
        lgmsg.info('CurDeteAlarmLevelInfo {} = {}'.format(WarnLooseDegree, CurDeteAlarmLevelInfo))

"""
    
if __name__ == '__main__':
    print('Start.')
    import matplotlib.pyplot as plt
    import os
    from matplotlib import path
    from detect_alarm.GlobalInfoDete import GlobalInfo
    from util.ReadLogInfo import ReadMultiSensorDeteFusionInfo, PlotMultiSensorDeteFusionInfo
    from util.ReadConfig import ReadSensorConfig, ReadIniConfigGlobalInfo, ReadReviseTime
    
    TestcallPythonFlag = 3 # TestcallPythonFlag = 1，使用 生成的结果 csv 文件
                           # TestcallPythonFlag = 2，使用服务器端 log 文件
                           # TestcallPythonFlag = 3，使用节点算法保存文件 mylog

    
    if TestcallPythonFlag == 1:
        import SyncData
        # SyncDeteInfo 
        SensorNameGroup = ['171', '172', '173', '174', '175', '176', '177']
        SensorDeteHumanInfoGroup = []
        for i_sensor in range(len(SensorNameGroup)):
            # 第一次每个传感器的检测结果文件
#                    SensorDeteHumanInfoGroup.append('..\\..\\MultiSensorDeteInfo\\DeteHumanInfo_' + str(SensorNameGroup[i_sensor]) + '.csv') 
            # 第二次每个传感器的检测结果文件，HSF
            SensorDeteHumanInfoGroup.append('D:\\xiongbiao\\HYD\\Code\\SolitaryCellDetect\\Code\\DetectPyCode\\MultiSensorDeteInfo\\DeteHumanInfo_20190314\\DeteHumanInfo_' + str(SensorNameGroup[i_sensor]) + '.csv') 
            # 第三次每个传感器的检测结果文件，HSF
#            SensorDeteHumanInfoGroup.append('..\\..\\MultiSensorDeteInfo\\DeteCsv\\DeteHumanInfo_' + str(SensorNameGroup[i_sensor]) + '.csv')
        
        Mode = 'OffLine'
        MultiSensorDeteHumanInfo = SyncData.SyncDeteInfo(SensorNameGroup, SensorDeteHumanInfoGroup, Mode)
#        print('MultiSensorDeteHumanInfo: \n{}'.format(MultiSensorDeteHumanInfo))
        # 初始化检测结果
        PreDeteInfoStr = '' #  [告警类型，告警次数，告警位置X，告警位置Y，告警位置Z，告警状态]
        
        # 循环每帧结果
        TestFrames = range(1,10,1) # 第一次每个传感器的检测结果文件，range(2200,2250,1)
                                    # 第二次每个传感器的检测结果文件，range(1,200,1)
                                    # 第三次每个传感器的检测结果文件，range(1,200,1)
                                        
        for i_frame in TestFrames:
            print('i_frame = {}'.format(i_frame))
            
            # 读取多传感器一帧数据
            OneFrameMultiSensorDeteHumanInfo = MultiSensorDeteHumanInfo[i_frame]
            # print('OneFrameMultiSensorDeteHumanInfo: \n {}'.format(OneFrameMultiSensorDeteHumanInfo))

            # Numeric info, dict()
            OneFrameMultiSensorDeteHumanInfoNumeric = dict()
            for i_obj in range(OneFrameMultiSensorDeteHumanInfo.shape[0]):
                try: 
                    OneFrameMultiSensorDeteHumanInfoNumeric[str(int(OneFrameMultiSensorDeteHumanInfo[i_obj,0]))] = \
                            np.row_stack((OneFrameMultiSensorDeteHumanInfoNumeric[str(int(OneFrameMultiSensorDeteHumanInfo[i_obj,0]))], OneFrameMultiSensorDeteHumanInfo[i_obj,1:5]))
                except:
                    OneFrameMultiSensorDeteHumanInfoNumeric[str(int(OneFrameMultiSensorDeteHumanInfo[i_obj,0]))] = OneFrameMultiSensorDeteHumanInfo[i_obj,1:5]
            # print('OneFrameMultiSensorDeteHumanInfoNumeric: \n{}'.format(OneFrameMultiSensorDeteHumanInfoNumeric))
            # String info
            OneFrameDeteHumanInfoString = GlobalInfo.alarm_detect_info(OneFrameMultiSensorDeteHumanInfoNumeric, 'W')
            # 手动添加房间ID
#            OneFrameDeteHumanInfoString = '{1002,' + OneFrameDeteHumanInfoString + '}'
            
            # 手动设置人员检测位置，20190612
            OneFrameDeteHumanInfoString = '{1001,' + '[10004,(1.0, 0.0,1.583, -3.616, 1.136), \
                                                        (1.0, 0.0,3.45, -3.693, 1.12 ), \
                                                        (1.0, 0.0,2.631, -3.678, 1.187), \
                                                        (1.0, 0.0,2.339, -3.302, 1.133), \
                                                        (2.0, 0.0,0.763, -3.535, 1.288), \
                                                        (2.0, 0.0,0.879, -4.317, 1.204)]' + '}'

#            OneFrameDeteHumanInfoString = '{1002,' + '[171,(0.0,7, -3.616, 1.136)],[177,(0.0, 4, -1, 1)]' + '}'

            print('OneFrameDeteHumanInfoString: \n{}'.format(OneFrameDeteHumanInfoString))
            
#            if (i_frame >75):
#                OneFrameDeteHumanInfoString = ''
            
            # detecter
            detecter_time1 = time.time()
            CurDeteStateStr = callPython(OneFrameDeteHumanInfoString, PreDeteInfoStr)
            print('detecter time = {}'.format(time.time() - detecter_time1))
            PreDeteInfoStr = CurDeteStateStr
            print('CurDeteStateStr: \n{}'.format(CurDeteStateStr))
            
    elif TestcallPythonFlag == 2: # 使用服务器端 log 文件
        from util.ReadLogInfo import ReadMultiSensorDeteFusionInfo
        
        LogFileType = 3 # if LogFileType == 1, 20200418 之前的log 文件格式
                        # if LogFileType == 2, 20200418 之后的log 文件格式
                        # if LogFileType == 3, 20200509 之后的msg_log 文件格式           
                        
        if LogFileType == 1:
            # 服务器端多个传感器检测结果，log 文件地址
    #        LogFileName = 'ZT_20200413_test.txt'
    #        LogFileName = 'ZT_20200413.txt'    
            LogFileName = 'ZT_20200417.txt'
            
            # 是否保存结果
            PlotResultFlag = 1
            PlotImageSaveDirName = r'result'
            MultiSensorArea = []
            if LogFileName == 'ZT_20200417.txt':
                # 20200417
                Area1 = [2.5260, 0.0720, 4.0510, 0.0720, 4.0510, 1.7810, 2.5260, 1.7810]
                Area2 = [0.1910, 0.9630, 0.3520, 0.9630, 0.3520, 1.3430, 0.1910, 1.3430]
                Area3 = [0.3880, 1.7740, 2.5340, 1.7740, 2.5340, 3.1180, 0.3880, 3.1180]
                Area4 = [2.5260, 1.7740, 4.0370, 1.7740, 4.0370, 3.1320, 2.5260, 3.1320]
                Area5 = [0.3960, 0.0720, 2.5340, 0.0720, 2.5340, 1.7810, 0.3960, 1.7810]
            MultiSensorArea = [Area1, Area2, Area3, Area4, Area5]
            MultiSensorName = ['10001', '10002', '10003', '10004', '10005']
            
        elif LogFileType == 2:
            # 服务器端多个传感器检测结果，log 文件地址
#            LogFileName = os.path.join('log','202005081400.txt')  
#            LogFileName = os.path.join('log','202005081800.txt')
#            LogFileName = os.path.join('log','202005082000.txt')
#            LogFileName = os.path.join('log','202005091400.txt')
            LogFileName = os.path.join('log','202005091500.txt')
            
            # 是否保存结果
            PlotResultFlag = 1
            PlotImageSaveFolder = os.path.basename(LogFileName).split('.txt')[0]
            PlotImageSaveDirName = os.path.join('result', PlotImageSaveFolder)
            if not os.path.exists(PlotImageSaveDirName):
                os.mkdir(PlotImageSaveDirName)
            # 各传感器检测区域
            MultiSensorArea = []
            if True:
                # 20200508
                Area1 = [0.1030, 0.0560, 2.4430, 0.0560, 2.4430, 3.5500, 0.1030, 3.5500]
                Area2 = [2.4430, 0.0560, 5.2120, 0.0560, 5.2120, 3.5500, 2.4430, 3.5500]
                Area3 = [5.2120, 0.0560, 7.9250, 0.0560, 7.9250, 3.6280, 5.2120, 3.6280]
                Area4 = [7.9250, 0.0340, 9.8920, 0.0340, 9.8920, 3.6830, 7.9250, 3.6830]
            MultiSensorArea = [Area1, Area2, Area3, Area4]
            # 内部监管区域
            SUPERVISEAREA = []
            if True:
                Area1 = [3.0420,   0.8790,   9.7950,   0.8790,   9.7950,   2.2200,   3.0420,   2.2200]
                Area2 = [0.1350,   0.0770,   3.0300,   0.0770,   3.0300,   2.1860,   0.1350,   2.1860]
            SUPERVISEAREA = [Area1, Area2]
            # 传感器名称
            MultiSensorName = ['10001', '10002', '10003', '10004']

        elif LogFileType == 3:
            SelectCaseName = 'ZT' # 选择场景数据， ZT/LG
            
            if SelectCaseName == 'ZT':
                # 服务器端多个传感器检测结果，log 文件地址
#                LogFileName = os.path.join('log','zt_msg_log_20200509.txt')
#                LogFileName = os.path.join('log','zt_msg_log_test.txt')
                LogFileName = os.path.join('log','zt_msg_log_20200728_TEST.txt')
                
                # 是否保存结果
                PlotResultFlag = 1
                PlotImageSaveFolder = os.path.basename(LogFileName).split('.txt')[0]
                PlotImageSaveDirName = os.path.join('result', PlotImageSaveFolder)
                if not os.path.exists(PlotImageSaveDirName):
                    os.mkdir(PlotImageSaveDirName)
                # 各传感器检测区域
                MultiSensorArea = []
                if True:
                    # 20200511
#                    Area1 = [1.9060, 0.0240, 3.8990, 0.0240, 3.8990, 1.5010, 1.9060, 1.5010]
#                    Area2 = [-0.2410, 0.5360, 0.0440, 0.5360, 0.0440, 1.1940, -0.2410, 1.1940]
#                    Area3 = [0.1610, 1.5080, 1.8980, 1.5080, 1.8980, 3.0500, 0.1610, 3.0500]
#                    Area4 = [1.8980, 1.5080, 3.8480, 1.5080, 3.8480, 3.0430, 1.8980, 3.0430]
#                    Area5 = [0.1610, 0.0100, 1.9060, 0.0100, 1.9060, 1.5080, 0.1610, 1.5080]

#                    # 20200509
#                    Area1 = [2.3960, 0.0070, 4.2800, 0.0070, 4.2800, 1.4970, 2.3960, 1.4970]
#                    Area2 = [-0.6040, 0.6040, 0.0700, 0.6040, 0.0700, 1.5960, -0.6040, 1.5960]
#                    Area3 = [0.5690, 1.4970, 2.3960, 1.4970, 2.3960, 2.9590, 0.5690, 2.9590]
#                    Area4 = [2.4030, 1.4970, 4.2800, 1.4970, 4.2800, 2.9730, 2.4030, 2.9730]
#                    Area5 = [0.5550, 0.0070, 2.3960, 0.0070, 2.3960, 1.4970, 0.5550, 1.4970]

                    # 20200728
                    Area1 = [2.6180, -0.0590, 3.8050, -0.0590, 3.8050, 0.9660, 2.6180, 0.9660]
                    Area2 = [-0.5600, 0.8610, -0.5600, 0.8610, -0.5600, 0.8610, -0.5600, 0.8610]
                    Area3 = [-0.0020, 0.9660, 2.6520, 0.9660, 2.6520, 3.1090, -0.0020, 3.1090]
                    Area4 = [2.6520, 0.9420, 3.8280, 0.9420, 3.8280, 3.1330, 2.6520, 3.1330]
                    Area5 = [-0.0130, -0.0590, 2.6180, -0.0590, 2.6180, 0.9420, -0.0130, 0.9420]
                    
                MultiSensorArea = [Area1, Area2, Area3, Area4, Area5]
                # 内部监管区域
                SUPERVISEAREA = []
                if True:
                    Area1 = [0.5700,   1.0590,   3.0720,   1.0590,   3.0720,   1.8650,   0.5700,   1.8650]
                SUPERVISEAREA = [Area1]
                # 传感器名称
                MultiSensorName = ['10001', '10002', '10003', '10004', '10005']
                
            if SelectCaseName == 'LG':
                # 服务器端多个传感器检测结果，log 文件地址
#                LogFileName = os.path.join('log','lg_msg_log_20200508.txt')
#                LogFileName = os.path.join('log','lg_msg_log_202005151230.txt')
                LogFileName = os.path.join('log','lg_msg_log_202005191330.txt')
                
                
                # 是否保存结果
                PlotResultFlag = 1
                PlotImageSaveFolder = os.path.basename(LogFileName).split('.txt')[0]
                PlotImageSaveDirName = os.path.join('result', PlotImageSaveFolder)
                if not os.path.exists(PlotImageSaveDirName):
                    os.mkdir(PlotImageSaveDirName)
                # 各传感器检测区域
                MultiSensorArea = []
                if True:
                    # 20200508
                    Area1 = [0.1030, 0.0560, 2.4430, 0.0560, 2.4430, 3.5500, 0.1030, 3.5500]
                    Area2 = [2.4430, 0.0560, 5.2120, 0.0560, 5.2120, 3.5500, 2.4430, 3.5500]
                    Area3 = [5.2120, 0.0560, 7.9250, 0.0560, 7.9250, 3.6280, 5.2120, 3.6280]
                    Area4 = [7.9250, 0.0340, 9.8920, 0.0340, 9.8920, 3.6830, 7.9250, 3.6830]
                MultiSensorArea = [Area1, Area2, Area3, Area4]
                # 内部监管区域
                SUPERVISEAREA = []
                if True:
                    Area1 = [3.0420,   0.8790,   9.7950,   0.8790,   9.7950,   2.2200,   3.0420,   2.2200]
                    Area2 = [0.1350,   0.0770,   3.0300,   0.0770,   3.0300,   2.1860,   0.1350,   2.1860]
                SUPERVISEAREA = [Area1, Area2]
                # 传感器名称
                MultiSensorName = ['10001', '10002', '10003', '10004']
        
        # 初始化输入信息
        OneFrameDeteHumanInfoStringAll = []
        # 读取 log 文件信息
        MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = LogFileType)
        OneFrameDeteHumanInfoStringAll = MultiSensorDeteFusionInputInfo
#        print('OneFrameDeteHumanInfoStringAll = {}'.format(OneFrameDeteHumanInfoStringAll))
#        print('OneFrameDeteHumanInfoStringAll 1 = {}'.format(OneFrameDeteHumanInfoStringAll[0]))
#        print('MultiSensorDeteFusionOutputInfo 1  = {}'.format(MultiSensorDeteFusionOutputInfo[0]))
        
        print('OneFrameDeteHumanInfoStringAll size = {}'.format(len(OneFrameDeteHumanInfoStringAll)))
        
        # 选择测试的帧序号
#        MultiSensorDeteInfoSelctFrm = [0, len(OneFrameDeteHumanInfoStringAll)] # 默认选择所有帧序号
#        MultiSensorDeteInfoSelctFrm = [2975, 2976]
#        MultiSensorDeteInfoSelctFrm = [250, 280]
#        MultiSensorDeteInfoSelctFrm = [250, 253] # 20201109
        MultiSensorDeteInfoSelctFrm = [250, 251] # 20201109
        
        # 手动设置输入信息
        # 单人留仓，line-8048
#        OneFrameDeteHumanInfoStringAll = [
#                                       '{1001,[10001,(1, 0, 2.5570, 1.5030, 0.9800)],[10002,(1, 0, 2.6090, 1.3670, 1.0520)],[10003,(1, 0, 2.6500, 1.3410, 1.0070)],[10004,(1, 0, 2.6110, 1.3820, 1.0820)],[10005,(1, 0, 2.5810, 1.4030, 0.9860)]}',
#                                       '{1001,[10001,(1, 0, 2.5670, 1.5360, 0.9850)],[10003,(1, 0, 2.6570, 1.3610, 0.9950)],[10004,(1, 0, 2.5880, 1.4910, 1.0640)],[10005,(1, 0, 2.5960, 1.4230, 0.9710)]}',
#                                       '{1001,[10001,(1, 0, 2.5500, 1.6300, 0.9410)],[10002,(1, 0, 2.5840, 1.5020, 1.0270)],[10003,(1, 0, 2.6730, 1.4160, 0.9580)],[10004,(1, 0, 2.4600, 1.7010, 1.1200)],[10005,(1, 0, 2.5800, 1.6150, 0.9870)]}',
#                                       '{1001,[10001,(1, 0, 2.5320, 1.8750, 1.1720)],[10002,(1, 0, 2.5530, 1.6920, 1.0730)],[10003,(1, 0, 2.5410, 1.7280, 1.0810)],[10004,(1, 0, 2.4410, 1.8880, 1.3390),(2, 1, 2.5150, 1.7080, 1.3570)],[10005,(1, 0, 2.4590, 1.8340, 1.3310)]}',
#                                       '{1001,[10001,(1, 0, 2.5350, 2.0030, 1.3240)],[10002,(1, 0, 2.5620, 1.9400, 1.2930)],[10003,(1, 0, 2.5770, 1.8930, 1.1850)],[10004,(1, 0, 2.5580, 2.3590, 1.3850)],[10005,(1, 0, 2.5100, 1.9650, 1.3420)]}',                                              
#                                              
#                                         ]

        # 单人留仓，20201109
#        MultiSensorDeteInfoSelctFrm = [0, 1]
#        OneFrameDeteHumanInfoStringAll = [
#                                       '{1001,[10001,(1, 1, 20.5270, 2.2710, 0.7160),(1, 1, 20.8050, 1.4640, 1.2570),(1, 0, 21.9890, 1.9610, 0.5170)]}',
#
#                                         ]

#        # 单人留仓，20201223
#        MultiSensorDeteInfoSelctFrm = [0, 6]
#        OneFrameDeteHumanInfoStringAll = [
#                                        '{1001,[10003,(1, 0, 1.2530, 1.0160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2060, 1.1430, 1.2590)],[10004,(-1, -1, -1, -1, -1)]}',
#                                        '{1001,[10003,(1, 0, 1.1530, 1.0160, 1.2590),(1, 0, 1.1730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2160, 1.1430, 1.2590)]}',
#                                        '{1001,[10003,(1, 0, 1.2530, 1.0160, 1.2590),(1, 0, 1.2630, 1.0160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2260, 1.1430, 1.2590)]}',
#                                        '{1001,[10003,(1, 0, 1.2530, 1.0160, 1.2590),(1, 0, 1.2630, 1.1160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10006,(1, 0, 1.2060, 1.1430, 1.2590)]}',
#                                        '{1001,[10003,(1, 0, 1.2530, 1.0160, 1.2590),(1, 0, 1.2630, 1.2160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2060, 1.1430, 1.2590)]}',
#                                        '{1001,[10003,(1, 0, 1.2630, 1.3160, 1.2590),(1, 0, 1.2730, 1.0160, 1.2590),(1, 0, 11.2530, 11.0160, 11.2590)],[10005,(1, 0, 1.2060, 1.1430, 1.2590)]}',
#                                         ]

        # 单人留仓，20201223
#        MultiSensorDeteInfoSelctFrm = [0, 6]
#        OneFrameDeteHumanInfoStringAll = [
#                                        '{1001,[10001,(512, 0, 0.8320, -3.0940, 1.3690)],[10002,(-1, -1, -1, -1, -1)],[10003,(512, 0, 0.8300, -3.0900, 1.3450)],[10004,(512, 0, 0.8340, -3.2210, 1.2590)]}',
#                                        '{1001,[10003,(512, 0, 0.8250, -3.0850, 1.3440)],[10004,(512, 0, 0.8500, -3.1980, 1.2410)]}',
#                                        '{1001,[10003,(512, 0, 0.8290, -3.0830, 1.3380)],[10004,(512, 0, 0.8430, -3.2140, 1.2710)]}',
#                                        '{1001,[10003,(512, 0, 0.8290, -3.0910, 1.3310)],[10004,(512, 0, 0.8420, -3.2170, 1.2750)]}',
#                                        '{1001,[10001,(512, 0, 0.8320, -3.0870, 1.3680)],[10003,(512, 0, 0.8270, -3.0930, 1.3390)],[10004,(512, 0, 0.8350, -3.2320, 1.2840)]}',
#                                        
#                                        '{1001,[10003,(512, 0, 0.8280, -3.0900, 1.3410)],[10004,(512, 0, 0.8360, -3.2270, 1.2740)]}',
#                                        '{1001,[10001,(512, 0, 0.8370, -3.1030, 1.3910)],[10003,(512, 0, 0.8320, -3.0880, 1.3380)],[10004,(512, 0, 0.8390, -3.2340, 1.2930)]}',
#                                        '{1001}',
#                                        '{1001,[10001,(512, 0, 0.8320, -3.0860, 1.3690)],[10003,(512, 0, 0.8210, -3.0910, 1.3260)]}',
#                                        '{1001,[10003,(512, 0, 0.8310, -3.0900, 1.3400)],[10004,(512, 0, 0.8560, -3.1990, 1.2570)]}',
#                                        
#                                        '{1001,[10001,(512, 0, 0.8410, -3.1040, 1.3940)],[10003,(512, 0, 0.8280, -3.0900, 1.3420)],[10004,(512, 0, 0.8390, -3.2220, 1.2850)]}',   
#                                         ]


#        MultiSensorDeteInfoSelctFrm = [0, 6]
#        OneFrameDeteHumanInfoStringAll = [
#                                          '{1001,[10002,(512, 0, -0.6450, -1.7920, 1.4670)],[10005,(512, 0, 2.4390, -2.3720, 1.3650),(512, 0, -0.5630, -1.9330, 1.3670)]}',
#                                          '{1001,[10002,(512, 0, 0.9600, -1.9600, 1.3850)],[10004,(512, 0, 1.0150, -2.0760, 1.3610)],[10005,(512, 0, 1.1520, -1.9670, 1.3710),(512, 0, 2.4380, -2.3780, 1.3710),(2, 1, 1.1590, -1.9810, 1.3660)]}',
#                                        '{1001,[10002,(512, 0, 0.8470, -2.0900, 1.3630)],[10005,(512, 0, 1.0410, -2.0250, 1.3720),(512, 0, 2.4390, -2.3800, 1.3770),(2, 1, 1.0530, -2.0390, 1.3800)]}',
#                                        '{1001,[10002,(512, 0, 0.8230, -2.2050, 1.3770)],[10004,(512, 0, 0.9440, -2.1730, 1.3410)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',
#                                        '{1001,[10001,(512, 0, 0.8080, -2.2970, 1.4430)],[10004,(512, 0, 0.9240, -2.2320, 1.3360)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',
#
#                                        '{1001,[10001,(512, 0, 0.8080, -2.2970, 1.4430)],[10004,(512, 0, 0.9240, -2.2320, 1.3360)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',
#                                        
#                                        '{1001,[10001,(512, 0, 0.5180, -2.4990, 1.3860)],[10002,(512, 0, 0.6910, -2.4300, 1.3660)],[10003,(512, 0, 0.7430, -2.3400, 1.3960)],[10004,(512, 0, 0.7320, -2.5580, 1.3640)]}',
#                                        '{1001,[10001,(512, 0, 0.4920, -2.5270, 1.3970)],[10002,(512, 0, 0.6170, -2.4850, 1.3520)],[10003,(512, 0, 0.5440, -2.4650, 1.3970)],[10004,(512, 0, 0.4610, -2.9150, 1.3820)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',
#                                        '{1001,[10001,(512, 0, 0.4710, -2.7950, 1.3960)],[10002,(512, 0, 0.4160, -2.6180, 1.3810)],[10003,(512, 0, 0.4330, -2.6120, 1.4000)],[10004,(512, 0, 0.4900, -3.0000, 1.3660)]}',
#                                        '{1001,[10001,(512, 0, 0.4480, -2.9700, 1.4090)],[10002,(512, 0, 0.3690, -2.8190, 1.3920)],[10004,(512, 0, 0.4400, -3.2200, 1.3740)],[10005,(512, 0, 0.5010, -2.7420, 1.3580),(512, 0, 2.4390, -2.3750, 1.3720)]}',
#                                        
#                                        
#                                        
#                                        '{1001,[10003,(512, 0, 0.8290, -3.0910, 1.3310)],[10004,(512, 0, 0.8420, -3.2170, 1.2750)]}',
#                                        '{1001,[10001,(512, 0, 0.8320, -3.0870, 1.3680)],[10003,(512, 0, 0.8270, -3.0930, 1.3390)],[10004,(512, 0, 0.8350, -3.2320, 1.2840)]}',
#                                        
#                                        '{1001,[10003,(512, 0, 0.8280, -3.0900, 1.3410)],[10004,(512, 0, 0.8360, -3.2270, 1.2740)]}',
#                                        '{1001,[10001,(512, 0, 0.8370, -3.1030, 1.3910)],[10003,(512, 0, 0.8320, -3.0880, 1.3380)],[10004,(512, 0, 0.8390, -3.2340, 1.2930)]}',
#                                        '{1001}',
#                                        '{1001,[10001,(512, 0, 0.8320, -3.0860, 1.3690)],[10003,(512, 0, 0.8210, -3.0910, 1.3260)]}',
#                                        '{1001,[10003,(512, 0, 0.8310, -3.0900, 1.3400)],[10004,(512, 0, 0.8560, -3.1990, 1.2570)]}',
#                                        
#                                        '{1001,[10001,(512, 0, 0.8410, -3.1040, 1.3940)],[10003,(512, 0, 0.8280, -3.0900, 1.3420)],[10004,(512, 0, 0.8390, -3.2220, 1.2850)]}',   
#                                         ]
                                         
                                         

        MultiSensorDeteInfoSelctFrm = [0, 5]
        OneFrameDeteHumanInfoStringAll = [
                                        '{1001,[10002,1609689600111,(512, 0, -0.6450, -1.7920, 1.4670)],[10005,1609689600112,(512, 0, 2.4390, -2.3720, 1.3650),(512, 0, -0.5630, -1.9330, 1.3670)]}',
                                        '{1001,[10002,(512, 0, 0.9600, -1.9600, 1.3850)],[10004,(512, 0, 1.0150, -2.0760, 1.3610)],[10005,(512, 0, 1.1520, -1.9670, 1.3710),(512, 0, 2.4380, -2.3780, 1.3710),(2, 1, 1.1590, -1.9810, 1.3660)]}',
                                        '{1001,[10002,(512, 0, 0.8470, -2.0900, 1.3630)],[10005,(512, 0, 1.0410, -2.0250, 1.3720),(512, 0, 2.4390, -2.3800, 1.3770),(2, 1, 1.0530, -2.0390, 1.3800)]}',
                                        '{1001,[10002,(512, 0, 0.8230, -2.2050, 1.3770)],[10004,(512, 0, 0.9440, -2.1730, 1.3410)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',
                                        '{1001,[10001,(512, 0, 0.8080, -2.2970, 1.4430)],[10004,(512, 0, 0.9240, -2.2320, 1.3360)],[10005,(512, 0, 2.4370, -2.3770, 1.3720),(512, 0, 0.9090, -2.2590, 1.3690),(2, 1, 0.9430, -2.2390, 1.3370)]}',

                                         ]


        # 循环每帧结果
        TestFrames = range(MultiSensorDeteInfoSelctFrm[0], MultiSensorDeteInfoSelctFrm[1],1) # 第一次每个传感器的检测结果文件，range(2200,2250,1)
                                    # 第二次每个传感器的检测结果文件，range(1,200,1)
                                    # 第三次每个传感器的检测结果文件，range(1,200,1)
        # 初始化检测结果
        PreDeteInfoStr = '' #  [告警类型，告警次数，告警位置X，告警位置Y，告警位置Z，告警状态]
        for i_frame in TestFrames:
            print('i_frame = {}'.format(i_frame))
            OneFrameDeteHumanInfoString = OneFrameDeteHumanInfoStringAll[i_frame]
            
            # callPython
            detecter_time1 = time.time()
            CurDeteStateStr = callPython(OneFrameDeteHumanInfoString, PreDeteInfoStr)
            print('detecter time = {}'.format(time.time() - detecter_time1))
            PreDeteInfoStr = CurDeteStateStr
            print('CurDeteStateStr: \n{}'.format(CurDeteStateStr))
    
#            time.sleep(2)
            PlotResultFlag = 0
            
            # 显示结果
            if PlotResultFlag == 1:
                ColorGroup = ['b','g','m','y','c','k']
                # 输入输出信息
                CurFusionInputStrInfo = OneFrameDeteHumanInfoString
                CurFusionOutputStrInfo = CurDeteStateStr
                # 转换原始输入信息
                GlobalInfoFuns = GlobalInfo(CurFusionInputStrInfo)
                CurFusionInputInfo, CurRoomID = GlobalInfoFuns.read_human_info_str()
#                CurRoomInfoGroup = GlobalInfoFuns.read_human_info_str()
#                CurFusionInputInfo = CurRoomInfoGroup['HumanInfo']
#                CurRoomID = CurRoomInfoGroup['RoomId']

                CurFusionOutputInfo = GlobalInfo.trans_alarm_info(CurFusionOutputStrInfo, 'R')
                # 显示多传感器输入信息
#                CurHumanInfo = CurFusionInputInfo.HumanInfo # 目标检测人员信息
#                CurInternalSupervisorHumanInfo = CurFusionInputInfo.InternalSupervisorInfo # 内部监管人员信息
                CurHumanInfo = CurFusionInputInfo[512] # 目标检测人员信息
                CurInternalSupervisorHumanInfo = CurFusionInputInfo[2] # 内部监管人员信息
                
                # 中间融合结果
                MultiSensorDeteInfoSelect = []
                MultiSensorDeteInternalSupervisorInfoSelect = []
                
                AloneObjEdgeDist = 0.2 # 区域边界区域, 再增加边界 0.15m
                AloneMultiObjsDisThod = 0.25 # 多目标之间的邻近距离阈值, init:0.25
                MultiSensorDeteInfoSelect = GlobalInfo.fuse_multi_sensor_dete_info(CurHumanInfo, GlobalInfoFuns.MultiSensorDeteArea, GlobalInfoFuns.MultiSensorEdgeArea, AddEdgeDist = AloneObjEdgeDist, MultiObjsDisThod = AloneMultiObjsDisThod)
                
                InternalSupervisorObjEdgeDist = 0.2 # 区域边界区域, 再增加边界 0.15m
                InternalSupervisorMultiObjsDisThod = 0.3 # 多目标之间的邻近距离阈值, init:0.3
                MultiSensorDeteInternalSupervisorInfoSelect = GlobalInfo.fuse_multi_sensor_dete_info(CurInternalSupervisorHumanInfo, GlobalInfoFuns.MultiSensorDeteArea, GlobalInfoFuns.MultiSensorEdgeArea, AddEdgeDist = InternalSupervisorObjEdgeDist, MultiObjsDisThod = InternalSupervisorMultiObjsDisThod)

                CurFrameAloneHumanNum = MultiSensorDeteInfoSelect.shape[0]
                # 再次判断是否在主仓区域内，【暂时使用主仓信息，主仓区域包含监管区域+厕所区域】
                MultiSensorDeteInternalSupervisorInfoSelect_New = []
                InternalSupervisorRoomIndex = [0]
                MainAndSubRoom = GlobalInfoFuns.MainAndSubRoom
                CurFrameInternalSupervisorHumanNum = 0
                for room in InternalSupervisorRoomIndex:
                    TempMainAndSubRoom = MainAndSubRoom[room] # 当前房间信息
                    for i_obj in range(MultiSensorDeteInternalSupervisorInfoSelect.shape[0]): # 每个目标位置判断
                        TempMainAndSubRoom = np.array(TempMainAndSubRoom)
                        CurrPoly = TempMainAndSubRoom.reshape([int(len(TempMainAndSubRoom)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                        pCurrPoly = path.Path(CurrPoly)
                        TempData = np.array([[0.0, 0.0]])
                        TempData[0,0] = MultiSensorDeteInternalSupervisorInfoSelect[i_obj,0]
                        TempData[0,1] = MultiSensorDeteInternalSupervisorInfoSelect[i_obj,1]
                        binAreaAll = pCurrPoly.contains_points(TempData) # limit xy, [2 x N]
                        if binAreaAll[0]: # 是否在多边形内
                            MultiSensorDeteInternalSupervisorInfoSelect_New.append(MultiSensorDeteInternalSupervisorInfoSelect[i_obj])
                            CurFrameInternalSupervisorHumanNum = CurFrameInternalSupervisorHumanNum + 1 # 个数增 1
                MultiSensorDeteInternalSupervisorInfoSelect_New = np.array(MultiSensorDeteInternalSupervisorInfoSelect_New)
                
                # 显示结果
                fig = plt.figure(1)
                plt.clf()
                plt.subplot(111)
                ax = plt.gca()
                # 数据显示范围
                Plot_XLim = [0,0]
                Plot_YLim = [0,0]
                # 画原始各传感器检测区域
                for i in range(len(MultiSensorArea)):
                    CurOneArea = np.array(MultiSensorArea[i])
                    CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                    plt.plot(CurOneArea[:,0], CurOneArea[:,1], color=ColorGroup[i%len(ColorGroup)])
                    plt.plot([CurOneArea[-1,0], CurOneArea[0,0]], [CurOneArea[-1,1], CurOneArea[0,1]], color=ColorGroup[i%len(ColorGroup)])
                    # 整个数据边界范围
                    if Plot_XLim[0] > min(CurOneArea[:,0]):
                        Plot_XLim[0] = min(CurOneArea[:,0])
                    if Plot_XLim[1] < max(CurOneArea[:,0]):
                        Plot_XLim[1] = max(CurOneArea[:,0])
                    if Plot_YLim[0] > min(CurOneArea[:,1]):
                        Plot_YLim[0] = min(CurOneArea[:,1])
                    if Plot_YLim[1] < max(CurOneArea[:,1]):
                        Plot_YLim[1] = max(CurOneArea[:,1])
                # 画原始内部监管区域
                for i in range(len(SUPERVISEAREA)):
                    CurOneSUPERVISEAREAArea = np.array(SUPERVISEAREA[i])
                    CurOneSUPERVISEAREAArea = CurOneSUPERVISEAREAArea.reshape([int((CurOneSUPERVISEAREAArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                    plt.plot(CurOneSUPERVISEAREAArea[:,0], CurOneSUPERVISEAREAArea[:,1], 'k--', linewidth=1)
                    plt.plot([CurOneSUPERVISEAREAArea[-1,0], CurOneSUPERVISEAREAArea[0,0]], [CurOneSUPERVISEAREAArea[-1,1], CurOneSUPERVISEAREAArea[0,1]], 'k--', linewidth=1)

                # 画目标点
                for sensor_dete in CurHumanInfo:
                    TempSensorDeteInfoSrc = CurHumanInfo[sensor_dete]
                    for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                        if not sensor_dete in MultiSensorName:
                            print('  sensor_dete error.')
                            continue
                        PlotColorIdx = MultiSensorName.index(sensor_dete)
                        plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = 'o')
                for sensor_dete in CurInternalSupervisorHumanInfo:
                    TempSensorDeteInfoSrc = CurInternalSupervisorHumanInfo[sensor_dete]
                    for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                        if not sensor_dete in MultiSensorName:
                            print('  sensor_dete error.')
                            continue
                        PlotColorIdx = MultiSensorName.index(sensor_dete)
                        plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = '*')

                
                # 显示多传感器输出信息
                AloneValidFlag = -1
                SupervisorValidFlag = -1
                for i_alan in range(CurFusionOutputInfo.shape[0]):
                    if CurFusionOutputInfo[i_alan, 0] == 1: # 单人留仓 当前帧状态
                        if CurFusionOutputInfo[i_alan, -1] == 1:
                            plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = 'o', markersize = 13)
                            AloneValidFlag = 1
                        else:
                            if not (CurFusionOutputInfo[i_alan, 3]==-1 and CurFusionOutputInfo[i_alan, 4]==-1 and CurFusionOutputInfo[i_alan, 5]==-1):
                                plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = 'o', markersize = 8)
                        
                    elif CurFusionOutputInfo[i_alan, 0] == 2: # 内部监管 当前帧状态
                        if CurFusionOutputInfo[i_alan, -1] == 1:
                            plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '*', markersize = 13)
                            SupervisorValidFlag = 1
#                        else:
#                            if not (CurFusionOutputInfo[i_alan, 3]==-1 and CurFusionOutputInfo[i_alan, 4]==-1 and CurFusionOutputInfo[i_alan, 5]==-1):
#                                plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '*', markersize = 8)
                            
                    elif CurFusionOutputInfo[i_alan, 0] == 3: # 群体冲突 当前帧状态
                        if CurFusionOutputInfo[i_alan, -1] == 1: 
                            plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '^', markersize = 13)
                        else:
                            if not (CurFusionOutputInfo[i_alan, 3]==-1 and CurFusionOutputInfo[i_alan, 4]==-1 and CurFusionOutputInfo[i_alan, 5]==-1):
                                plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '^', markersize = 8)
                                
                # 中间融合结果
#                print(MultiSensorDeteInfoSelect)
                for i in range(MultiSensorDeteInfoSelect.shape[0]):
                    plt.scatter(MultiSensorDeteInfoSelect[i,0], MultiSensorDeteInfoSelect[i,1], color = '', marker = 'o', edgecolors='r', s=200)
#                for i in range(MultiSensorDeteInternalSupervisorInfoSelect.shape[0]):
#                    plt.scatter(MultiSensorDeteInternalSupervisorInfoSelect[i,0], MultiSensorDeteInternalSupervisorInfoSelect[i,1], color = '', marker = '*', edgecolors='r', s=200)
                for i in range(MultiSensorDeteInternalSupervisorInfoSelect_New.shape[0]):
                    plt.scatter(MultiSensorDeteInternalSupervisorInfoSelect_New[i,0], MultiSensorDeteInternalSupervisorInfoSelect_New[i,1], color = '', marker = '*', edgecolors='r', s=200)

        
                # title
#                CurFrmTitle = 'Frm = {}, [Alone: HumanNum = {}, State = {}], [Supervisor: HumanNum = {}, State= {}]'.format(str(i_frame).zfill(5), CurFrameAloneHumanNum, AloneValidFlag, CurFrameInternalSupervisorHumanNum, SupervisorValidFlag)
                CurFrmTitle = 'Frm = {}, [Alone: {}, {}], [Supervisor: {}, {}]'.format(str(i_frame).zfill(5), CurFrameAloneHumanNum, AloneValidFlag, CurFrameInternalSupervisorHumanNum, SupervisorValidFlag)

                plt.title(CurFrmTitle)
                plt.axis('equal')
#                plt.xlim([-1, 5])
#                plt.ylim([-1, 4])
                plt.xlim([Plot_XLim[0] - 0.5, Plot_XLim[1] + 0.5])
                plt.ylim([Plot_YLim[0] - 0.5, Plot_YLim[1] + 0.5])
                plt.show()

                # save figure
                if os.path.exists(PlotImageSaveDirName):
                    CurFrameSaveName = os.path.join(PlotImageSaveDirName, str(i_frame).zfill(5) + '.png')
                    plt.savefig(CurFrameSaveName, dpi=300)

    if TestcallPythonFlag == 3: # 使用节点算法保存文件 mylog
        print('TestcallPythonFlag = ', TestcallPythonFlag)
        
        LogFileType = 1 # LogFileType = 1, 202012 修改的节点算法log信息
        
        # log 文件地址
#        TestLogFolderName = r'mylog'
#        TestLogFileName = 'msg_log' # 'msg_log'/'msg_log.1'

        TestLogFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete_Event\detect_global\result'
#        TestLogFileName = 'msg_log_1226_15' # 785
#        SelectFrm = [400, 402] # [300, 780]
        
#        TestLogFileName = 'msg_log_1227_11' # 567
#        SelectFrm = None # None
        
#        TestLogFileName = 'msg_log_1227_15' # 
#        SelectFrm = [748,749] # None  

#        TestLogFileName = 'msg_log_1227_16' # 
#        SelectFrm = None # None  

#        TestLogFileName = 'msg_log_1227_17' # 
#        SelectFrm = None # None  
        
#        TestLogFileName = 'msg_log_1227_1750' # 1257
#        SelectFrm = None # None  
        
#        TestLogFileName = 'msg_log_1228_16' #
#        SelectFrm = None # None  
        
#        TestLogFileName = 'msg_log_1229_09' # 6695
#        SelectFrm = [6060,6062] # None

#        TestLogFileName = 'msg_log_1229_11' # 569
#        SelectFrm = [300, 569] # None
        
#        TestLogFileName = 'msg_log_1229_14' # 
#        SelectFrm = None # None

#        TestLogFileName = 'msg_log_1229_16_02' # 
#        SelectFrm = None # None
        
#        TestLogFileName = 'msg_log_1229_19' # 348
#        SelectFrm = [0,1] # None 

#        TestLogFileName = 'msg_log_1230_14' # 302
#        SelectFrm = None # None 

#        TestLogFileName = 'msg_log_1230_15' # 
#        SelectFrm = [960,1000] # None 
        
#        TestLogFileName = 'msg_log_1230_17' # 3844
#        SelectFrm = [0,1] # None 

#        TestLogFileName = 'msg_log_123109' # 3844
#        SelectFrm = [100,102] # None 
        
#        TestLogFileName = 'msg_log_123110' # 3844
#        SelectFrm = [0,1] # None 

#        TestLogFileName = 'msg_log_0113' # 3844
#        SelectFrm = [500,501] # None
        
        TestLogFileName = 'msg_log_0113_1906' # 3844
        SelectFrm = [100,102] # None
        
        
        if LogFileType == 1:
            # log 文件地址
            LogFileName = os.path.join(TestLogFolderName, TestLogFileName)
            # conifg 文件地址
            ConifgFileName = os.path.join('SensorConfig.ini')
            # log 结果保存
            PlotResultFlag = 1
            PlotImageSaveDirName = os.path.join(r'result', 'log_result')
            PlotImageSaveFolderName = os.path.join(PlotImageSaveDirName, TestLogFileName)
            if not os.path.exists(PlotImageSaveFolderName):
                os.mkdir(PlotImageSaveFolderName)
            # 读取 log 信息
            MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = 4)
            print('MultiSensorDeteFusionInputInfo len = ', len(MultiSensorDeteFusionInputInfo['MultiSensorDeteInfo']))
            
            # 读取 conifg 信息
            SensorInfoAll = ReadIniConfigGlobalInfo(ConifgFileName) # 读取配置文件信息
            SensorNum = int(SensorInfoAll['ROOM_INFO']['sensornum'])
            print('SensorInfoAll room num = ', SensorNum)
#            print(SensorInfoAll)
            
            # 显示 log结果
            MultiSensorArea = []
            MultiSensorName = []
            for i_sensor in range(SensorNum):
                # sensor name
                CurSensorName = str(int(10000 + i_sensor + 1))
                MultiSensorName.append(CurSensorName)
                # sensor area
                CurSensorAreaPoints = SensorInfoAll['SENSORS_INFO']['DeteAreaPoints'][CurSensorName]
                MultiSensorArea.append(CurSensorAreaPoints)
#            print('MultiSensorName = ', MultiSensorName)
#            print('MultiSensorArea = ', MultiSensorArea)
                
                
#            SelectFrm = [300, 780]
#            FigAxisXYRange = [-2,3.5,-5.0,-1.0] # [xmin,xmax,ymin,ymax]
            FigAxisXYRange = None
            PlotImageSaveDirName = PlotImageSaveFolderName
            PlotMultiSensorDeteFusionInfo(MultiSensorArea, MultiSensorName, MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo, PlotImageSaveDirName, SelectFrm = SelectFrm, FigAxisXYRange = FigAxisXYRange)

            # 保存 log结果
            
    
    print('End.')
    
    