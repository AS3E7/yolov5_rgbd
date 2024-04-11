# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:21:53 2020

@author: HYD
"""
#import _init_path

import numpy as np
import os
import cv2
import copy
import time
from shutil import copyfile

import logging as lgmsg
import re
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

import detect_person
from detect_alarm.AlarmString import DeteInfoStrToNumeric, DeteInfoNumericToStr, DetectInfo, RWDetectInfo
from detect_bbox.SSD_Detect.Eval import PlotDeteResult
from util.CloudPointFuns import ReadKinect2FromPly, Rot3D, SavePt3D2Ply
from util.ReadConfig import ReadIniConfig, TransformConfigInfo
from util.BGProcess import CalBGInfo, CalImageSubBG
from util.SaveData import DeleteOutLimitFiles, LogSaveCategoryIndexFun
from util.DeteModeProcess import SelectDeteMode
from config import one_sensor_params, online_offline_type, log_info, alarm_level

##############################################################################
# logging
##############################################################################
global DebugFlag
DebugFlag = 0 # 保存打印结果信息； 
#              #     DebugFlag == 1，保存print 信息
#              #     DebugFlag == 2，保存print 信息 + 保存图片检测结果信息
#if DebugFlag == 1 or DebugFlag == 2:
#    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                        level=logging.DEBUG,
#                        filename='log/test.log',
#                        filemode='a')
#    logging.debug('debug级别，一般用来打印一些调试信息，级别最低')
    
#    logger = logging.getLogger('test')
#    logger.debug('debug级别，一般用来打印一些调试信息，级别最低')
#    logger.info('info级别，一般用来打印一些正常的操作信息')
#    logger.warning('waring级别，一般用来打印警告信息')
#    logger.error('error级别，一般用来打印一些错误信息')
#    logger.critical('critical级别，一般用来打印一些致命的错误信息，等级最高')


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
# 选择使用的数据模式，DataTypeFlag传入所有的数据，需要自己选择需要使用的数据模式
##############################################################################
SelectUsingDataType = 1 # 选择哪些数据模式
                        # SelectUsingDataType == 1， 点云 + 深度
                        # SelectUsingDataType == 2， 点云 + 深度 + RGB
DetectBboxImageType =  one_sensor_params['detect_bbox_image_type']
if DetectBboxImageType == 'colormap':
    SelectUsingDataType = 1
elif DetectBboxImageType == 'rgb':
    SelectUsingDataType = 2
elif DetectBboxImageType == 'colormap_rgb':
    SelectUsingDataType = 3
else:
    SelectUsingDataType = 1
    print('detect_bbox_image_type error !')
                        

##############################################################################
# 读取配置文件和背景文件
##############################################################################
global FileReviseBool, ConfigInfo, BGFileReviseBool, BGInfo
# 读取配置文件
# 在线/离线文件地址
OnOffLineType = online_offline_type # 'OnLine','OffLine'
if OnOffLineType == 'OnLine':
    SensorConfigReviseFileName = "demo/log/FileReviseTime.log"
    SensorConfigFileName = 'demo/SensorConfig.ini'
    BGReviseFileName = "demo/log/BGFileReviseTime.log"
    BGFileName = 'demo/log/bg.ply'
    SaveResultDir = 'demo/result'
    SaveDataDepthDir = 'demo/result/depth'
    SaveDataPlyDir = 'demo/result/ply'
    SensorConfigCopyFileName = 'demo/result/SensorConfig.ini' # 备份 配置文件 至告警保存文件夹中
    BGCopyFileName = 'demo/result/bg.ply' # 备份 背景文件 至告警保存文件夹中
else:
    SensorConfigReviseFileName = "log/FileReviseTime.log"
    SensorConfigFileName = 'SensorConfig.ini'
    BGReviseFileName = "log/BGFileReviseTime.log"
    BGFileName = 'log/bg.ply'
    SaveResultDir = 'result'
    SaveDataDepthDir = 'result/depth'
    SaveDataPlyDir = 'result/ply'
    SensorConfigCopyFileName = 'result/SensorConfig.ini' # 备份 配置文件 至告警保存文件夹中
    BGCopyFileName = 'result/bg.ply' # 备份 背景文件 至告警保存文件夹中
# 保存文件大小设置，配置文件设置单位为：MB
CheckTimeRange = 360 # 单位：秒。默认：3600s，1个小时检测一次，
SaveDataDepthDirSize = log_info['save_data_size_depth'] * 1024 * 1024 # Byte
SaveDataPlyDirSize = log_info['save_data_size_ply'] * 1024 * 1024 # Byte
# 读取配置文件
FileReviseBool = False
ConfigInfo = ReadIniConfig(SensorConfigFileName) # 读取配置文件
# 备份 配置文件
SensorConfigCopyFileNameSave = SensorConfigCopyFileName.replace('.ini', '_' + str(os.stat(SensorConfigFileName).st_mtime)+'.ini')
if not os.path.exists(SensorConfigCopyFileNameSave):
    copyfile(SensorConfigFileName, SensorConfigCopyFileNameSave) # 复制 配置 文件
# 打印配置文件读取信息
if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0: # 打印配置文件读取信息
    lgmsg.info('第一次读取配置文件:SensorConfig.ini')
    lgmsg.info('    SensorConfig info: {}'.format(ConfigInfo))

##############################################################################
# 告警松紧度等级信息
##############################################################################
CurConfigInfoDeteAlarmLevelSetting = ConfigInfo['ROOM_INFO']['managelevel'] # 1/2/3
CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
    lgmsg.info('第一次获取告警松紧度等级信息')
    lgmsg.info('    CurDeteAlarmLevelInfo {} = {}'.format(CurConfigInfoDeteAlarmLevelSetting, CurDeteAlarmLevelInfo))


##############################################################################
# 背景信息
##############################################################################
# 是否读取 bg 信息
DepthWidth = one_sensor_params['depth_width']
DepthHeight = one_sensor_params['depth_height']
ReadDeteBboxBGInfoFlag = one_sensor_params['detect_bbox_use_bg_info'] # 目标检测是否读取使用信息
ReadBGInfoFlag = one_sensor_params['use_bg_info'] # 是否使用BG 信息
BGInfoHeightLimit = one_sensor_params['use_bg_info_height_limit'] # 高度信息更新背景
#BGInfoOutlierDistLimit = one_sensor_params['use_bg_info_outlier_dist'] # 获取边界离群点距离阈值
BGInfoOutlierDistLimit = CurDeteAlarmLevelInfo['use_bg_info_outlier_dist'] # 获取边界离群点距离阈值

# 读取背景文件
BGFileReviseBool = False # 初始化 背景是否更新
BGInfo = np.empty([4,0])
if ReadBGInfoFlag == True:
    # 复制 bg 文件
    BGCopyFileNameSave = BGCopyFileName.replace('.ply', '_' + str(os.stat(BGFileName).st_mtime)+'.ply')
    if not os.path.exists(BGCopyFileNameSave):
        copyfile(BGFileName, BGCopyFileNameSave) # 复制 bg 文件
    # 在线/离线文件地址
    if online_offline_type=='OffLine' and os.path.exists(os.path.join('log', 'bg.ply')):
        BGSrcPts = ReadKinect2FromPly(os.path.join('log', 'bg.ply')) # [3 x N]
        BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit) # [4 x N]
        # 根据配置信息 H 选择一定高度数据更新
        if not BGInfoHeightLimit == False:
            CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
            CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
            NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
            NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
            BGInfo[-1, NewPts_Z_Invalid] = 0
    elif online_offline_type=='OnLine' and os.path.exists(os.path.join('demo', 'log', 'bg.ply')):
        BGSrcPts = ReadKinect2FromPly(os.path.join('demo', 'log', 'bg.ply')) # [3 x N]
        BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit)
        # 根据配置信息 H 选择一定高度数据更新
        if not BGInfoHeightLimit == False:
            CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
            CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
            NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
            NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
            BGInfo[-1, NewPts_Z_Invalid] = 0

# 打印信息
if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
    lgmsg.info('one_sensor_params = {}'.format(one_sensor_params))
    lgmsg.info('    DetectBboxImageType = {}, SelectUsingDataType = {}'.format(DetectBboxImageType, SelectUsingDataType))
    lgmsg.info('第一次获取背景信息')
            
def callPython(RGBData, RGBWidth, RGBHeight, DepthData, Pts, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr):
    """
    功能：监仓检测接口函数
    输入：
        RGBData：RGB数据，一维数据
		RGBWidth：RGB图像宽度，如：Kinect2.0的RGB图像宽度为1920；
		RGBHeight：RGB图像高度，如：Kinect2.0的RGB图像高度为1080；
		DepthData：Depth数据，一维数据
		DepthWidth：Depth图像宽度，如：Kinect2.0的Depth图像宽度为512；
		DepthHeight：Depth图像高度，如：Kinect2.0的Depth图像高度为424；
		Pts：点云数据，二维数据，如：Kinect2.0的点云数据尺寸为[3 x 217088]；
		DataTypeFlag：传入数据类型
            数据类型控制位，[1*10]，控制类型：[点云，深度，rgb,...]		
        WorldTime：世界时间，毫秒单位；
        DeteStr：上一帧数检测的返回结果，string变量；

    输出：
        DeteStr：当前帧的检测结果，string变量；
    """
    print('====== Start Online callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        lgmsg.info('======  Start Online callPython ======')
    
    #配置文件读取
    global FileReviseBool, ConfigInfo, BGInfo
    FileReviseBool = ReadReviseTime(SensorConfigFileName, SensorConfigReviseFileName)
    if FileReviseBool == False:
        # 备份 配置文件
        SensorConfigCopyFileNameSave = SensorConfigCopyFileName.replace('.ini', '_' + str(os.stat(SensorConfigFileName).st_mtime)+'.ini')
        if not os.path.exists(SensorConfigCopyFileNameSave):
            copyfile(SensorConfigFileName, SensorConfigCopyFileNameSave) # 复制 配置 文件
        # 再次读取配置文件
        ConfigInfo = ReadIniConfig(SensorConfigFileName)
        ConfigInfo['isUpdate'] = True
#        if DebugFlag > 0:
#            lgmsg.debug('reload config file = {} '.format(SensorConfigFileName))
#            lgmsg.debug('    config info = {}'.format(ConfigInfo))
        if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
            lgmsg.info('reLoad Ini Config File Info.')
            lgmsg.info('reload config file = {} '.format(SensorConfigFileName))
            lgmsg.info('    config info = {}'.format(ConfigInfo))
            CurConfigInfoDeteAlarmLevelSetting = ConfigInfo['ROOM_INFO']['managelevel'] # 1/2/3
            CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
            lgmsg.info('CurDeteAlarmLevelInfo {} = {}'.format(CurConfigInfoDeteAlarmLevelSetting, CurDeteAlarmLevelInfo))
        # 根据配置信息 H 选择一定高度数据更新
        if not BGInfoHeightLimit == False:
            CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
            CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
            BGSrcPts = ReadKinect2FromPly(os.path.join('demo', 'log', 'bg.ply')) # [3 x N]
            NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
            NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
            BGInfo[-1, NewPts_Z_Invalid] = 0
    else:
        ConfigInfo['isUpdate'] = False
    # 背景文件读取
    BGFileReviseBool = ReadReviseTime(BGFileName, BGReviseFileName)
    if BGFileReviseBool == False: # 重新读取背景文件信息
        # 复制 bg 文件
        BGCopyFileNameSave = BGCopyFileName.replace('.ply', '_' + str(os.stat(BGFileName).st_mtime)+'.ply')
        if not os.path.exists(BGCopyFileNameSave):
            copyfile(BGFileName, BGCopyFileNameSave) # 复制 bg 文件
        print('reLoad BG File Info.')
        BGInfo = np.empty([4,0])
        if ReadBGInfoFlag == True:
            # 重新读取 BGInfoOutlierDistLimit
            ConfigInfo_2 = ReadIniConfig(SensorConfigFileName)
            CurConfigInfoDeteAlarmLevelSetting = ConfigInfo_2['ROOM_INFO']['managelevel'] # 1/2/3
            CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
            BGInfoOutlierDistLimit = CurDeteAlarmLevelInfo['use_bg_info_outlier_dist'] # 获取边界离群点距离阈值
            # 更新 BGInfo
            if online_offline_type=='OffLine' and os.path.exists(os.path.join('log', 'bg.ply')):
                BGSrcPts = ReadKinect2FromPly(os.path.join('log', 'bg.ply')) # [3 x N]
                BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit) # [4 x N]
                # 根据配置信息 H 选择一定高度数据更新
                if not BGInfoHeightLimit == False:
                    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
                    CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
                    NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
                    NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
                    BGInfo[-1, NewPts_Z_Invalid] = 0
            elif online_offline_type=='OnLine' and os.path.exists(os.path.join('demo', 'log', 'bg.ply')):
                BGSrcPts = ReadKinect2FromPly(os.path.join('demo', 'log', 'bg.ply')) # [3 x N]
                BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit)
                # 根据配置信息 H 选择一定高度数据更新
                if not BGInfoHeightLimit == False:
                    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
                    CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
                    NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
                    NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
                    BGInfo[-1, NewPts_Z_Invalid] = 0
    
    # 输入数据模式
    InputDataType = 0
    if DataTypeFlag[0] == 1 and DataTypeFlag[1] == 1: 
        # 输入数据模式1：点云 + 深度
        InputDataType = 1
    elif DataTypeFlag[0] == 1 and DataTypeFlag[1] == 1 and DataTypeFlag[2] == 1: 
        # 输入数据模式1：点云 + 深度 + RGB
        InputDataType = 2
    if DebugFlag > 0: 
        lgmsg.debug('  InputDataType = {}'.format(InputDataType))
        lgmsg.debug('  SelectUsingDataType = {}'.format(SelectUsingDataType))
        
    # 是否保存输入数据
    if ConfigSaveInputPtsFlag == 1:
        InputPtsFileName = os.path.join(SaveResultDir, 'input_pts_' + str(time.time()) + '.ply')
        SavePt3D2Ply(InputPtsFileName, Pts.transpose(), 'XYZ')
    
        
    # 深度数据/点云数据是否添加背景
    PtsDeteBbox = Pts
    if BGInfo.shape[1] > 0: # 有背景信息
        PtsSubBG = CalImageSubBG(Pts, BGInfo, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthWidth)
        # Pts
        if ReadDeteBboxBGInfoFlag == True: # 目标检测是否读取使用信息
            PtsDeteBbox = PtsSubBG
            Pts = PtsSubBG
        else: 
            PtsDeteBbox = Pts
            Pts = PtsSubBG
        # Depth
        DepthData = DepthData # depth 数据对应变换
    else: # 无背景信息
        PtsSubBG = Pts
        # Pts
        PtsDeteBbox = Pts
        # Depth
        DepthData = DepthData # depth 数据对应变换 
        
    # 前一帧结果：转换 string 格式转为 数值 格式
    SensorInfo, PreDete = DeteInfoStrToNumeric(DeteStr)
    
    # 检测目标
    if SelectUsingDataType == 1: # 输入数据模式1：点云 + 深度
        ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag, PtsSubBG=PtsSubBG)
    elif SelectUsingDataType == 2: # 输入数据模式1：点云 + 深度 + RGB
        # trans RGB data format
        im_rgb = np.reshape(RGBData, (RGBHeight, RGBWidth,3))
        im_rgb_plt = copy.deepcopy(im_rgb)
        RGBImageData = cv2.cvtColor(im_rgb_plt, cv2.COLOR_BGR2RGB)
        # DetectRGBDepth
        ImgDeteResult, AlarmResult = detect_person.DetectRGBDepth(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag, PtsSubBG=PtsSubBG)
    elif SelectUsingDataType == 3: # 输入数据模式1：点云 + 深度 + RGB
        # 判断使用数据类型
        CurFrmSelectImageType = SelectDeteMode(RGBData, RGBWidth, RGBHeight, DepthData, DepthWidth, DepthHeight, WorldTime)
        if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
            lgmsg.info('CurFrmSelectImageType = {}'.format(CurFrmSelectImageType))
        # 检测目标
        if CurFrmSelectImageType == 'colormap':
            ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)
        elif CurFrmSelectImageType == 'rgb':
            im_rgb = np.reshape(RGBData, (RGBHeight, RGBWidth,3))
            im_rgb_plt = copy.deepcopy(im_rgb)
            RGBImageData = cv2.cvtColor(im_rgb_plt, cv2.COLOR_BGR2RGB)
            ImgDeteResult, AlarmResult = detect_person.DetectRGBDepth(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)
        else:
            ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)

        
    # 是否保存结果
    if ConfigSaveDetectBboxImageFlag == 1:
        ResultSaveFileName = os.path.join(SaveResultDir, str(time.time()) + '.png')
        PlotDeteResult(ImgDeteResult['Data'], ImgDeteResult['Bbox'], ImgDeteResult['Label'], ImgDeteResult['Score'], ResultSaveFileName)

        
    # 当前帧结果：数值 格式 转换 string 格式
    AlarmStringResult = DeteInfoNumericToStr(SensorInfo, AlarmResult)
    
    # 判断返回结果是否为空
    if (AlarmStringResult == ''):
        lgmsg.error('error AlarmStringResult')
        AlarmResultInit = DetectInfo()
        AlarmStringResult = DeteInfoNumericToStr(SensorInfo, AlarmResultInit)
    
    # print result
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        # 显示当前帧检测人员信息
        lgmsg.debug('AlarmResult HUMANINFO = {}'.format(AlarmResult.HUMANINFO))
        # 显示所有返回结果
#        lgmsg.debug('CurFrmDeteResult = {}'.format(AlarmStringResult))
        # 只显示功能检测结果
        CurFrmDetecterStrDetectInfo = RWDetectInfo(AlarmResult, 'W')
        lgmsg.debug('CurFrmDeteResult = {}'.format(CurFrmDetecterStrDetectInfo))
        
    # 判断文件大小，一段时间删除过多文件
    CurTime = int(time.time())
    if CurTime%CheckTimeRange == 0:
        DeleteOutLimitFiles(SaveDataDepthDir, SaveDataDepthDirSize, PostfixName = 'depth')
        DeleteOutLimitFiles(SaveDataPlyDir, SaveDataPlyDirSize, PostfixName = 'ply')
        
    print('======  End Online callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        lgmsg.info('======  End Online callPython ======')
          
    return AlarmStringResult
    
    
def callPythonOffLine(RGBData, RGBWidth, RGBHeight, DepthData, Pts, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr, SaveFileName):
    """
    功能：监仓检测接口函数
    输入：
        RGBData：RGB数据，一维数据
		RGBWidth：RGB图像宽度，如：Kinect2.0的RGB图像宽度为1920；
		RGBHeight：RGB图像高度，如：Kinect2.0的RGB图像高度为1080；
		DepthData：Depth数据，一维数据
		DepthWidth：Depth图像宽度，如：Kinect2.0的Depth图像宽度为512；
		DepthHeight：Depth图像高度，如：Kinect2.0的Depth图像高度为424；
		Pts：点云数据，二维数据，如：Kinect2.0的点云数据尺寸为[3 x 217088]；
		DataTypeFlag：传入数据类型
            数据类型控制位，[1*10]，控制类型：[点云，深度，rgb,...]		
        WorldTime：世界时间，毫秒单位；
        DeteStr：上一帧数检测的返回结果，string变量；

    输出：
        DeteStr：当前帧的检测结果，string变量；
    """
    print('======  Start Offline callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        lgmsg.info('======  Start Offline callPython ======')
    
    global FileReviseBool, ConfigInfo, BGFileReviseBool, BGInfo
    # 配置文件读取
    FileReviseBool = ReadReviseTime(SensorConfigFileName, SensorConfigReviseFileName)
    if FileReviseBool == False:
        # 备份 配置文件
        SensorConfigCopyFileNameSave = SensorConfigCopyFileName.replace('.ini', '_' + str(os.stat(SensorConfigFileName).st_mtime)+'.ini')
        if not os.path.exists(SensorConfigCopyFileNameSave):
            copyfile(SensorConfigFileName, SensorConfigCopyFileNameSave) # 复制 配置 文件
        # 再次读取配置文件
        ConfigInfo = ReadIniConfig(SensorConfigFileName)
        ConfigInfo['isUpdate'] = True
#        if DebugFlag > 0:
#            logging.debug('reload config file = {} '.format(SensorConfigFileName))
#            logging.debug('    config info = {}'.format(ConfigInfo))
        if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
            lgmsg.info('reLoad Ini Config File Info.')
            lgmsg.info('reload config file = {} '.format(SensorConfigFileName))
            lgmsg.info('    config info = {}'.format(ConfigInfo))
            CurConfigInfoDeteAlarmLevelSetting = ConfigInfo['ROOM_INFO']['managelevel'] # 1/2/3
            CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
            lgmsg.info('CurDeteAlarmLevelInfo {} = {}'.format(CurConfigInfoDeteAlarmLevelSetting, CurDeteAlarmLevelInfo))
        # 根据配置信息 H 选择一定高度数据更新
        if not BGInfoHeightLimit == False:
            CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
            CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
            BGSrcPts = ReadKinect2FromPly(os.path.join('log', 'bg.ply')) # [3 x N]
            NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
            NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
            BGInfo[-1, NewPts_Z_Invalid] = 0
    else:
        ConfigInfo['isUpdate'] = False
    # 背景文件读取
    BGFileReviseBool = ReadReviseTime(BGFileName, BGReviseFileName)
    if BGFileReviseBool == False: # 重新读取背景文件信息
        # 复制 bg 文件
        BGCopyFileNameSave = BGCopyFileName.replace('.ply', '_' + str(os.stat(BGFileName).st_mtime)+'.ply')
        if not os.path.exists(BGCopyFileNameSave):
            copyfile(BGFileName, BGCopyFileNameSave) # 复制 bg 文件
        print('reLoad BG File Info.')
        BGInfo = np.empty([4,0])
        if ReadBGInfoFlag == True:
            # 重新读取 BGInfoOutlierDistLimit
            ConfigInfo_2 = ReadIniConfig(SensorConfigFileName)
            CurConfigInfoDeteAlarmLevelSetting = ConfigInfo_2['ROOM_INFO']['managelevel'] # 1/2/3
            CurDeteAlarmLevelInfo = alarm_level[str(CurConfigInfoDeteAlarmLevelSetting)]
            BGInfoOutlierDistLimit = CurDeteAlarmLevelInfo['use_bg_info_outlier_dist'] # 获取边界离群点距离阈值
            # 更新 BGInfo
            if online_offline_type=='OffLine' and os.path.exists(os.path.join('log', 'bg.ply')):
                BGSrcPts = ReadKinect2FromPly(os.path.join('log', 'bg.ply')) # [3 x N]
                BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit) # [4 x N]
                # 根据配置信息 H 选择一定高度数据更新
                if not BGInfoHeightLimit == False:
                    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
                    CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
                    NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
                    NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
                    BGInfo[-1, NewPts_Z_Invalid] = 0
            elif online_offline_type=='OnLine' and os.path.exists(os.path.join('demo', 'log', 'bg.ply')):
                BGSrcPts = ReadKinect2FromPly(os.path.join('demo', 'log', 'bg.ply')) # [3 x N]
                BGInfo = CalBGInfo(BGSrcPts, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthHeight, StdSet = BGInfoOutlierDistLimit)
                # 根据配置信息 H 选择一定高度数据更新
                if not BGInfoHeightLimit == False:
                    CurSensorId = ConfigInfo['SENSORS_INFO']['currsensorid']
                    CurSensorH = ConfigInfo['SENSORS_INFO'][CurSensorId]['H']
                    NewPts = Rot3D(CurSensorH, BGSrcPts)# [3 x N]
                    NewPts_Z_Invalid = (NewPts[-1,:] < BGInfoHeightLimit)
                    BGInfo[-1, NewPts_Z_Invalid] = 0
    # 输入数据模式
    InputDataType = 0
    if DataTypeFlag[0] == 1 and DataTypeFlag[1] == 1: 
        # 输入数据模式1：点云 + 深度
        InputDataType = 1
    elif DataTypeFlag[0] == 1 and DataTypeFlag[1] == 1 and DataTypeFlag[2] == 1:
        # 输入数据模式1：点云 + 深度 + RGB
        InputDataType = 2
#    if DebugFlag > 0: 
#        logging.debug('  InputDataType = {}'.format(InputDataType))
#        logging.debug('  SelectUsingDataType = {}'.format(SelectUsingDataType))
        
        
    # 是否保存输入数据
    if ConfigSaveInputPtsFlag == 1:
        InputPtsFileName = os.path.join(SaveResultDir, SaveFileName + '.ply')
        SavePt3D2Ply(InputPtsFileName, Pts.transpose(), 'XYZ')
        
    # 深度数据/点云数据是否添加背景
    t1 = time.time()
    PtsDeteBbox = Pts
    if BGInfo.shape[1] > 0: # 有背景信息
        PtsSubBG = CalImageSubBG(Pts, BGInfo, CalMethod = 1, DepthWidth = DepthWidth, DepthHeight = DepthWidth)
        # Pts
        if ReadDeteBboxBGInfoFlag == True: # 目标检测是否读取使用信息
            PtsDeteBbox = PtsSubBG
            Pts = PtsSubBG
        else: 
            PtsDeteBbox = Pts
            Pts = PtsSubBG
        # Depth
        DepthData = DepthData # depth 数据对应变换
    else: # 无背景信息
        PtsSubBG = Pts
        # Pts
        PtsDeteBbox = Pts
        # Depth
        DepthData = DepthData # depth 数据对应变换 

#        SavePt3D2Ply(str(time.time()) + 'pts_2.ply', Pts.transpose(), 'XYZ')
#    print('CalImageSubBG time {}'.format(time.time()-t1)) # 13 ms
        
    # 前一帧结果：转换 string 格式转为 数值 格式
    SensorInfo, PreDete = DeteInfoStrToNumeric(DeteStr)
    
    # 检测目标
    if SelectUsingDataType == 1: # 输入数据模式1：点云 + 深度
        ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag, PtsSubBG=PtsSubBG)
    elif SelectUsingDataType == 2: # 输入数据模式1：点云 + 深度 + RGB
        # trans RGB data format
        im_rgb = np.reshape(RGBData, (RGBHeight, RGBWidth,3))
        im_rgb_plt = copy.deepcopy(im_rgb)
        RGBImageData = cv2.cvtColor(im_rgb_plt, cv2.COLOR_BGR2RGB)
        # DetectRGBDepth
        ImgDeteResult, AlarmResult = detect_person.DetectRGBDepth(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag, PtsSubBG=PtsSubBG)
    elif SelectUsingDataType == 3: # 输入数据模式1：点云 + 深度 + RGB
        # 判断使用数据类型
        CurFrmSelectImageType = SelectDeteMode(RGBData, RGBWidth, RGBHeight, DepthData, DepthWidth, DepthHeight, WorldTime)
        if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
            lgmsg.info('CurFrmSelectImageType = {}'.format(CurFrmSelectImageType))
        # 检测目标
        if CurFrmSelectImageType == 'colormap':
            ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)
        elif CurFrmSelectImageType == 'rgb':
            im_rgb = np.reshape(RGBData, (RGBHeight, RGBWidth,3))
            im_rgb_plt = copy.deepcopy(im_rgb)
            RGBImageData = cv2.cvtColor(im_rgb_plt, cv2.COLOR_BGR2RGB)
            ImgDeteResult, AlarmResult = detect_person.DetectRGBDepth(RGBImageData, RGBWidth, RGBHeight, DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)
        else:
            ImgDeteResult, AlarmResult = detect_person.DetectDepth(DepthData, Pts, PtsDeteBbox, DepthWidth, DepthHeight, WorldTime, PreDete, ConfigInfo, DebugFlag = DebugFlag)

    # 是否保存结果
    if len(SaveFileName)>1 and ConfigSaveDetectBboxImageFlag==1:
        ResultSaveFileName = os.path.join(SaveResultDir, SaveFileName + '.png')
        PlotDeteResult(ImgDeteResult['Data'], ImgDeteResult['Bbox'], ImgDeteResult['Label'], ImgDeteResult['Score'], ResultSaveFileName)
        
    # 当前帧结果：数值 格式 转换 string 格式
    AlarmStringResult = DeteInfoNumericToStr(SensorInfo, AlarmResult)
      
    # 判断返回结果是否为空
#    AlarmStringResult = ''
    if (AlarmStringResult == ''):
        lgmsg.error('error AlarmStringResult')
        AlarmResultInit = DetectInfo()
        AlarmStringResult = DeteInfoNumericToStr(SensorInfo, AlarmResultInit)
        
    # print result
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.FuncInputOutputInfo]) > 0:
        # 显示当前帧检测人员信息
        lgmsg.debug('AlarmResult HUMANINFO = {}'.format(AlarmResult.HUMANINFO))
        # 显示所有返回结果
#        lgmsg.debug('CurFrmDeteResult = {}'.format(AlarmStringResult))
        # 只显示功能检测结果
        CurFrmDetecterStrDetectInfo = RWDetectInfo(AlarmResult, 'W')
        lgmsg.debug('CurFrmDeteResult = {}'.format(CurFrmDetecterStrDetectInfo))
        
    # 判断文件大小，一段时间删除过多文件
    CurTime = int(time.time())
    if CurTime%CheckTimeRange == 0:
        DeleteOutLimitFiles(SaveDataDepthDir, SaveDataDepthDirSize, PostfixName = 'depth')
        DeleteOutLimitFiles(SaveDataPlyDir, SaveDataPlyDirSize, PostfixName = 'ply')
        
    print('======  End Offline callPython ======')
    if int(ConfigDebugInfo[ConfigDebugTypeIndex.BasicProceInfo]) > 0:
        lgmsg.info('======  End Offline callPython ======')
        
    return AlarmStringResult
    
    
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
    print('Start.')
    TestcallPythonFlag = 1 # 测试 callPython
    
    if TestcallPythonFlag == 1:
        # RGB
        RGBData = np.zeros([1920,1080,3])
        RGBWidth = 1920
        RGBHeight = 1080
        # Depth
        DepthData = np.zeros([512, 424])
        Pts = np.zeros([217088, 3])
        DepthWidth = 512
        DepthHeight = 424
        # DataTypeFlag
        DataTypeFlag = [1,1,1,0,0,0,0,0,0,0]
        # other
        WorldTime = 15000000000000
        DeteStr = '171,172'
        # detect
        callPython(RGBData, RGBWidth, RGBHeight, DepthData, Pts, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr)

    print('End.')