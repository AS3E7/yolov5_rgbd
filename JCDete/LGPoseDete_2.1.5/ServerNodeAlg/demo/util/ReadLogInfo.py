# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:18:44 2020

@author: HYD
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from detect_alarm.GlobalInfoDete import GlobalInfo
from detect_alarm.AlarmIDInfo import AlarmIDInfo
from util.AreaFuns import CalMultiAreaEdge, CalAboveHeightValidArea
from util.PolygonFuns import TransBboxToPoints, TransPointsToBbox

AlarmIDIndex = AlarmIDInfo() # 告警ID 序号

def ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = 1):
    """
    功能：读取多传感器融合结果信息
    输入：服务器端多传感器端log 文件
    输出：
        算法输入信息：MultiSensorDeteFusionInputInfo
        算法输出信息：MultiSensorDeteFusionOutputInfo
        LogFileType: 配置文件格式：
                        # if LogFileType == 1, 20200418 之前的log 文件格式
                        # if LogFileType == 2, 20200418 之后的log 文件格式
                        # if LogFileType == 3, 20200509 之后的msg_log 文件格式
                        # if LogFileType == 4, 20201225  mylog/msg_log 文件格式
    """
    # 初始化信息
    MultiSensorDeteFusionInputInfo = []
    MultiSensorDeteFusionOutputInfo = []
    
    if LogFileType == 1:
        MultiSensorInputPrefix = 'mPushdata = '
        MultiSensorOutputPrefix = '----m_string = '
        
        # read txt file
        fp = open(LogFileName, 'r')
        # 按行读取
        line = fp.readline()
        while line:
            if line.find(MultiSensorInputPrefix) > -1:
                # 输入信息
                MultiSensorDeteFusionInputInfo.append(line.strip().split(MultiSensorInputPrefix)[-1])
                                
                # 输出信息
    #            line = fp.readline()
    #            if line.find(MultiSensorOutputPrefix) > -1:
    #                CurFrameOutputInfo = []
    #                CurFrameOutputInfo.append(line.strip().split(MultiSensorOutputPrefix)[-1])
    #                while True:
    #                    line = fp.readline()
    #                    if len(line)>1: # '----m_string = ' 最后结果出现空行
    #                        CurFrameOutputInfo.append(line.strip())
    #                    else:
    #                        MultiSensorDeteFusionOutputInfo.append(CurFrameOutputInfo)
    #                        break
    
                line = fp.readline()
                while line:
                    if line.find(MultiSensorOutputPrefix) > -1:
                        CurFrameOutputInfo = ''
                        CurFrameOutputInfo = CurFrameOutputInfo + line.split(MultiSensorOutputPrefix)[-1]
    
                        while True:
                            line = fp.readline()
                            if len(line)>1: # '----m_string = ' 最后结果出现空行
                                CurFrameOutputInfo = CurFrameOutputInfo + line
                            else:
                                MultiSensorDeteFusionOutputInfo.append(CurFrameOutputInfo)
                                break
                        break
                    if line.find(MultiSensorInputPrefix) > -1:
                        MultiSensorDeteFusionInputInfo = MultiSensorDeteFusionInputInfo[:-1]
                        break
                    line = fp.readline()
    
            # readline
            else:
                line = fp.readline()
     
        fp.close()
        # 判断数据有效性
        if not len(MultiSensorDeteFusionInputInfo) == len(MultiSensorDeteFusionOutputInfo):
            print('Input size {} != Output size {}'.format(len(MultiSensorDeteFusionInputInfo), len(MultiSensorDeteFusionOutputInfo)))
            MultiSensorDeteFusionInputInfo = []
            MultiSensorDeteFusionOutputInfo = []
    elif LogFileType == 2:
        print('LogFileType = {}'.format(LogFileType))
        MultiSensorInputPrefix = 'mPushdata == -----------------------'
        MultiSensorInputPrefix_2 = ']['
        MultiSensorOutputPrefix = '----m_string = '
        
        # read txt file
        fp = open(LogFileName, 'r')
        # 按行读取
        line = fp.readline()
        while line:
            if line.find(MultiSensorInputPrefix) > -1:
                # 输入信息
                MultiSensorDeteFusionInputInfo.append(line.strip().split(MultiSensorInputPrefix_2)[-1].split('}')[0] + '}')

                line = fp.readline()
                while line:
                    if line.find(MultiSensorOutputPrefix) > -1:
                        CurFrameOutputInfo = ''
                        CurFrameOutputInfo = CurFrameOutputInfo + line.split(MultiSensorOutputPrefix)[-1]
    
                        while True:
                            line = fp.readline()
                            if len(line)>1: # '----m_string = ' 最后结果出现空行
                                CurFrameOutputInfo = CurFrameOutputInfo + line
                            else:
                                MultiSensorDeteFusionOutputInfo.append(CurFrameOutputInfo)
                                break
                        break
                    if line.find(MultiSensorInputPrefix) > -1:
                        MultiSensorDeteFusionInputInfo = MultiSensorDeteFusionInputInfo[:-1]
                        break
                    line = fp.readline()
    
            # readline
            else:
                line = fp.readline()

        fp.close()
        # 判断数据有效性
        if not len(MultiSensorDeteFusionInputInfo) == len(MultiSensorDeteFusionOutputInfo):
            print('Input size {} != Output size {}'.format(len(MultiSensorDeteFusionInputInfo), len(MultiSensorDeteFusionOutputInfo)))
            MultiSensorDeteFusionInputInfo = []
            MultiSensorDeteFusionOutputInfo = []
    elif LogFileType == 3:
        print('LogFileType = {}'.format(LogFileType))
        MultiSensorInputPrefix = '单人留仓:当前总人数'
        MultiSensorInputPrefix_2 = '位置：'
        MultiSensorInputPrefix_3 = '房间id = '
        MultiSensorOutputPrefix = ''
        
        # read txt file
#        fp = open(LogFileName, 'r', encoding='UTF-8')
        fp = open(LogFileName, 'r', encoding='gbk')

        # 按行读取
        line = fp.readline()
        while line:
            if line.find(MultiSensorInputPrefix) > -1:
                # 输入信息
                CurRoomName = line.strip().split(MultiSensorInputPrefix_3)[-1].split(',')[0] # 当前房间号
                MultiSensorDeteFusionInputInfo.append('{' + CurRoomName + line.strip().split(MultiSensorInputPrefix_2)[-1].split('}')[0] + '}')

                line = fp.readline()
                
            # readline
            else:
                line = fp.readline()
        fp.close()
        
    elif LogFileType == 4: # 20201225  mylog/msg_log 文件格式
        MultiSensorDeteFusionInputInfo = dict()
        MultiSensorDeteFusionOutputInfo = dict()
        MultiSensorDeteFusionInputInfo['MultiSensorDeteInfo'] = []
        MultiSensorDeteFusionInputInfo['FrameWorldTime'] = []
        MultiSensorDeteFusionOutputInfo['MultiSensorDeteInfo'] = []
        
        print('Read log file, LogFileType = {}'.format(LogFileType))
        MultiSensorInputPrefix = 'CurHumanInfoStr = '
        MultiSensorOutputPrefix = 'CurDeteStateStr = '
        
        LogTimeLen = 23 # 时间戳长度
        
        # read txt file
#        fp = open(LogFileName, 'r', encoding='UTF-8')
        fp = open(LogFileName, 'r', encoding='gbk')

        # 按行读取
        line = fp.readline()
        while line:
            if line.find(MultiSensorInputPrefix) > -1:
                # 输入信息
                CurMultiSensorDeteInfoStr = line.strip().split(MultiSensorInputPrefix)[-1] # 
                CurTimeStr = line.strip()[:LogTimeLen] # 时间戳
                MultiSensorDeteFusionInputInfo['MultiSensorDeteInfo'].append(CurMultiSensorDeteInfoStr)
                MultiSensorDeteFusionInputInfo['FrameWorldTime'].append(CurTimeStr)
                
                line = fp.readline()
                
            # readline
            else:
                line = fp.readline()
        fp.close()
        
    return MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo
    
    
def PlotMultiSensorDeteFusionInfo_Src(MultiSensorArea, MultiSensorName, MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo, PlotImageSaveDirName):
    """
    功能：显示多传感器检测结果
    输入：
        MultiSensorArea: 各传感器区域
        MultiSensorDeteFusionInputInfo：算法输入信息
        MultiSensorDeteFusionOutputInfo：算法输出信息
    """
    ColorGroup = ['b','g','m','y','c','k']
    
    # 遍历每帧结果
    FrameNum = len(MultiSensorDeteFusionInputInfo)
    for i_frame in range(FrameNum):
        print('i_frame = {}'.format(i_frame))
        
        CurFusionInputStrInfo = MultiSensorDeteFusionInputInfo[i_frame]
        CurFusionOutputStrInfo = MultiSensorDeteFusionOutputInfo[i_frame]
        # 转换原始输入信息
        GlobalInfoFuns = GlobalInfo(CurFusionInputStrInfo)
        CurFusionInputInfo, CurRoomID = GlobalInfoFuns.read_human_info_str()
        CurFusionOutputInfo = GlobalInfo.trans_alarm_info(CurFusionOutputStrInfo, 'R')
#        print('CurFusionInputInfo', CurFusionInputInfo)
#        print('CurFusionOutputStrInfo', CurFusionOutputInfo)
        
        # 显示结果
        fig = plt.figure(1)
        plt.clf()
        plt.subplot(111)
        ax = plt.gca()
        # 画原始框
        for i in range(len(MultiSensorArea)):
            CurOneArea = np.array(MultiSensorArea[i])
            CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
            plt.plot(CurOneArea[:,0], CurOneArea[:,1], color=ColorGroup[i%len(ColorGroup)])
            plt.plot([CurOneArea[-1,0], CurOneArea[0,0]], [CurOneArea[-1,1], CurOneArea[0,1]], color=ColorGroup[i%len(ColorGroup)])

        # 显示多传感器输入信息
        CurHumanInfo = CurFusionInputInfo.HumanInfo # 目标检测人员信息
        CurInternalSupervisorHumanInfo = CurFusionInputInfo.InternalSupervisorInfo # 内部监管人员信息
 
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
                else:
                    if not (CurFusionOutputInfo[i_alan, 3]==-1 and CurFusionOutputInfo[i_alan, 4]==-1 and CurFusionOutputInfo[i_alan, 5]==-1):
                        plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '*', markersize = 8)
                    
            elif CurFusionOutputInfo[i_alan, 0] == 3: # 群体冲突 当前帧状态
                if CurFusionOutputInfo[i_alan, -1] == 1: 
                    plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '^', markersize = 13)
                else:
                    if not (CurFusionOutputInfo[i_alan, 3]==-1 and CurFusionOutputInfo[i_alan, 4]==-1 and CurFusionOutputInfo[i_alan, 5]==-1):
                        plt.plot(CurFusionOutputInfo[i_alan,3], CurFusionOutputInfo[i_alan,4], color = 'r', marker = '*', markersize = 8)

        # title
        CurFrmTitle = 'Frm = {}, Alone = {}, Supervisor = {}'.format(str(i_frame).zfill(5), AloneValidFlag, SupervisorValidFlag)
        plt.title(CurFrmTitle)
        plt.axis('equal')
        plt.xlim([-1, 5])
        plt.ylim([-1, 4])
        plt.show()
        # save figure
        if os.path.exists(PlotImageSaveDirName):
            CurFrameSaveName = os.path.join(PlotImageSaveDirName, str(i_frame).zfill(5) + '.png')
            plt.savefig(CurFrameSaveName, dpi=300)
            
            
#        if i_frame > 100:
#            break
        break
    
    return 0
    
def PlotMultiSensorDeteFusionInfo(MultiSensorArea, MultiSensorName, MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo, PlotImageSaveDirName, SelectFrm = None, FigAxisXYRange = None):
    """
    功能：显示多传感器检测结果
    输入：
        MultiSensorArea: 各传感器区域
        MultiSensorDeteFusionInputInfo：算法输入信息
        MultiSensorDeteFusionOutputInfo：算法输出信息
        SelectFrm：选择显示的帧序号
        FigAxisXYRange：图像显示 X/Y 范围, [xmin,xmax,ymin,ymax]
    """
    
    ColorGroup = ['b','g','m','y','c','k']
    
    # 遍历每帧结果
    if SelectFrm == None:
        FrameNumMin = 0
        FrameNumMax = len(MultiSensorDeteFusionInputInfo['MultiSensorDeteInfo'])
    else:
        FrameNumMin = SelectFrm[0]
        FrameNumMax = SelectFrm[1]
        
    CurDeteSateStr = ''
    for i_frame in range(FrameNumMin, FrameNumMax):
        print('i_frame = {}'.format(i_frame))
        
        CurFrameWorldTime = MultiSensorDeteFusionInputInfo['FrameWorldTime'][i_frame]
        CurFusionInputStrInfo = MultiSensorDeteFusionInputInfo['MultiSensorDeteInfo'][i_frame]
#        CurFusionOutputStrInfo = MultiSensorDeteFusionOutputInfo['MultiSensorDeteInfo'][i_frame]
        # 转换原始输入信息
        GlobalInfoFuns = GlobalInfo(CurFusionInputStrInfo)
        MultiSensorEdgeArea = GlobalInfoFuns.MultiSensorEdgeArea
        
#        CurFusionInputInfo, _, _ = GlobalInfoFuns.read_human_info_str()
        CurRoomInfoGroup = GlobalInfoFuns.read_human_info_str()
        CurFusionInputInfo = CurRoomInfoGroup['HumanInfo']
        
        PreDeteSateStr = CurDeteSateStr
        PreDeteSateNumeric = GlobalInfo.trans_alarm_info(PreDeteSateStr, 'R')
        
        CurFusionOutputInfo, CurFrameMultiSensorAlarmInfo, CurMultiFrameMultiSensorAlarmInfo = GlobalInfoFuns.detect_alarm(CurFusionInputInfo, PreDeteSateNumeric)  # 重新计算检测结果
        CurDeteSateStr = GlobalInfo.trans_alarm_info(CurFusionOutputInfo, 'W')

#        print('CurFusionInputStrInfo = ', CurFusionInputStrInfo)
#        print('CurDeteSateStr = ', CurDeteSateStr)
        
#        print('MultiSensorEdgeArea = ', MultiSensorEdgeArea)
#        print('CurFusionInputInfo = ', CurFusionInputInfo)
#        print('CurFusionOutputStrInfo', CurFusionOutputInfo)
#        print('CurFrameMultiSensorAlarmInfo = ' , CurFrameMultiSensorAlarmInfo)
#        print('CurMultiFrameMultiSensorAlarmInfo = ' , CurMultiFrameMultiSensorAlarmInfo)
        

        MultiSensorEdgeAloneDist = 0.6 # 单人留仓边界区域

        # 显示结果
        fig = plt.figure(1)
        plt.clf()
        plt.subplot(111)
        ax = plt.gca()
        # 画各传感器检测框
        for i in range(len(MultiSensorArea)):
            CurOneArea = np.array(MultiSensorArea[i])
            CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
            plt.plot(CurOneArea[:,0], CurOneArea[:,1], color=ColorGroup[i%len(ColorGroup)])
            plt.plot([CurOneArea[-1,0], CurOneArea[0,0]], [CurOneArea[-1,1], CurOneArea[0,1]], color=ColorGroup[i%len(ColorGroup)])

            # 判断是否在当前传感器检测范围内
            TempSensorDeteArea = MultiSensorArea[i]
#            print(TempSensorDeteArea)
            # 求检测区域边界，【暂时使用最大矩形边界】
            CurOneArea_bbox = TransPointsToBbox(TempSensorDeteArea)
            TempSensorDeteArea = CalAboveHeightValidArea(CurOneArea_bbox, MultiSensorEdgeArea, MultiSensorEdgeAloneDist) # 
            TempSensorDeteArea = TransBboxToPoints(TempSensorDeteArea)
            # 多边形区域内
            TempSensorDeteArea = np.array(TempSensorDeteArea)
            CurOneArea = TempSensorDeteArea.reshape([int(len(TempSensorDeteArea)/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
            
            plt.plot(CurOneArea[:,0], CurOneArea[:,1], color=ColorGroup[i%len(ColorGroup)], linewidth=0.5, linestyle='--')
            plt.plot([CurOneArea[-1,0], CurOneArea[0,0]], [CurOneArea[-1,1], CurOneArea[0,1]], color=ColorGroup[i%len(ColorGroup)], linewidth=0.5, linestyle='--')

            
#        # 画多传感器边界区域
#        for i_edge in range(MultiSensorEdgeArea.shape[0]):
#            MultiSensorEdgeAreaOne = MultiSensorEdgeArea[i_edge] # [xmin,xmax,ymin,ymax]
#            plt.gca().add_patch(plt.Rectangle((MultiSensorEdgeAreaOne[0],MultiSensorEdgeAreaOne[2]),MultiSensorEdgeAreaOne[1]-MultiSensorEdgeAreaOne[0],MultiSensorEdgeAreaOne[3]-MultiSensorEdgeAreaOne[2], fill=False, edgecolor='k', linewidth=3))

            
        HumanAloneIndex = AlarmIDIndex.Alarm_Alone
        InternalSupervisorIndex = AlarmIDIndex.Alarm_InternalSupervisor
        AboveHeightIndex = AlarmIDIndex.Alarm_AboveHeight
        NoLyingIndex = AlarmIDIndex.Alarm_NoLying
        WrongLyingIndex = AlarmIDIndex.Alarm_WrongLying
        
        HumanAloneShowFlag = True
        InternalSupervisorShowFlag = True
        AboveHeightShowFlag = False
        NoLyingIndexFlag = False
        WrongLyingIndexFlag = False

        # 显示多传感器输入信息        
        CurHumanInfo = CurFusionInputInfo[HumanAloneIndex] # 目标检测人员信息
        CurInternalSupervisorHumanInfo = CurFusionInputInfo[InternalSupervisorIndex] # 内部监管人员信息
        CurAboveHeightInfo = CurFusionInputInfo[AboveHeightIndex] # 超高
        if NoLyingIndexFlag:
            CurNoLyingInfo = CurFusionInputInfo[NoLyingIndex] # 休息时间不休息
        if WrongLyingIndexFlag:
            CurWrongLyingInfo = CurFusionInputInfo[WrongLyingIndex] # 非休息时间休息
        
        if HumanAloneShowFlag == True:
            for sensor_dete in CurHumanInfo:
                TempSensorDeteInfoSrc = CurHumanInfo[sensor_dete]
    #            print(sensor_dete, TempSensorDeteInfoSrc)
                for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                    if not sensor_dete in MultiSensorName:
                        print('  HumanAlone sensor_dete error.')
                        continue
                    PlotColorIdx = MultiSensorName.index(sensor_dete)
                    plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = 'o')
        if InternalSupervisorShowFlag == True:
            for sensor_dete in CurInternalSupervisorHumanInfo:
                TempSensorDeteInfoSrc = CurInternalSupervisorHumanInfo[sensor_dete]
                for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                    if not sensor_dete in MultiSensorName:
                        print('  InternalSupervisor sensor_dete error.')
                        continue
                    PlotColorIdx = MultiSensorName.index(sensor_dete)
                    plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = '*')
        if AboveHeightShowFlag == True:
            for sensor_dete in CurAboveHeightInfo:
                TempSensorDeteInfoSrc = CurAboveHeightInfo[sensor_dete]
                for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                    if not sensor_dete in MultiSensorName:
                        print('  InternalSupervisor sensor_dete error.')
                        continue
                    PlotColorIdx = MultiSensorName.index(sensor_dete)
                    plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = '^')
        if NoLyingIndexFlag == True:
            for sensor_dete in CurNoLyingInfo:
                TempSensorDeteInfoSrc = CurNoLyingInfo[sensor_dete]
                for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                    if not sensor_dete in MultiSensorName:
                        print('  InternalSupervisor sensor_dete error.')
                        continue
                    PlotColorIdx = MultiSensorName.index(sensor_dete)
                    plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = '+')
        if WrongLyingIndexFlag == True:
            for sensor_dete in CurWrongLyingInfo:
                TempSensorDeteInfoSrc = CurWrongLyingInfo[sensor_dete]
                for i_obj in range(TempSensorDeteInfoSrc.shape[0]):
                    if not sensor_dete in MultiSensorName:
                        print('  InternalSupervisor sensor_dete error.')
                        continue
                    PlotColorIdx = MultiSensorName.index(sensor_dete)
                    plt.plot(TempSensorDeteInfoSrc[i_obj,1], TempSensorDeteInfoSrc[i_obj,2], color = ColorGroup[PlotColorIdx%len(ColorGroup)], marker = 'x')
                
    
        # 显示多传感器聚类检测
        for i_alarm in CurFrameMultiSensorAlarmInfo:
            CurFrameMultiSensorAlarmInfoOne = CurFrameMultiSensorAlarmInfo[i_alarm]['DeteAreaHumanInfo'] # 'DeteAreaHumanInfo'
            if len(CurFrameMultiSensorAlarmInfoOne) == 0:
                continue
            # HumanAloneIndex
            if i_alarm == HumanAloneIndex and HumanAloneShowFlag==True:
                for i_area in CurFrameMultiSensorAlarmInfoOne:
                    CurAlarmAreaOne = CurFrameMultiSensorAlarmInfoOne[i_area]['ObjInfo']
                    if len(CurAlarmAreaOne) == 0:
                        continue
                    for i_obj in range(CurAlarmAreaOne.shape[0]):
                        CurAlarmAreaOneObjOne = CurAlarmAreaOne[i_obj,:]
                        plt.scatter(CurAlarmAreaOneObjOne[0], CurAlarmAreaOneObjOne[1], color = '', marker = 'o', edgecolors='r', s = 100)

            # InternalSupervisorIndex
            if i_alarm == InternalSupervisorIndex and InternalSupervisorShowFlag==True:
                for i_area in CurFrameMultiSensorAlarmInfoOne:
                    CurAlarmAreaOne = CurFrameMultiSensorAlarmInfoOne[i_area]['ObjInfo']
                    if len(CurAlarmAreaOne) == 0:
                        continue
                    for i_obj in range(CurAlarmAreaOne.shape[0]):
                        CurAlarmAreaOneObjOne = CurAlarmAreaOne[i_obj,:]
                        plt.scatter(CurAlarmAreaOneObjOne[0], CurAlarmAreaOneObjOne[1], color = '', marker = '*', edgecolors='r', s = 100)

            # AboveHeightIndex
            if i_alarm == AboveHeightIndex and AboveHeightShowFlag==True:
                for i_area in CurFrameMultiSensorAlarmInfoOne:
                    CurAlarmAreaOne = CurFrameMultiSensorAlarmInfoOne[i_area]['ObjInfo']
                    if len(CurAlarmAreaOne) == 0:
                        continue
                    for i_obj in range(CurAlarmAreaOne.shape[0]):
                        CurAlarmAreaOneObjOne = CurAlarmAreaOne[i_obj,:]
                        plt.scatter(CurAlarmAreaOneObjOne[0], CurAlarmAreaOneObjOne[1], color = '', marker = '^', edgecolors='r', s = 100)
            # NoLying
            if i_alarm == NoLyingIndex and NoLyingIndexFlag==True:
                for i_area in CurFrameMultiSensorAlarmInfoOne:
                    CurAlarmAreaOne = CurFrameMultiSensorAlarmInfoOne[i_area]['ObjInfo']
                    if len(CurAlarmAreaOne) == 0:
                        continue
                    for i_obj in range(CurAlarmAreaOne.shape[0]):
                        CurAlarmAreaOneObjOne = CurAlarmAreaOne[i_obj,:]
                        plt.scatter(CurAlarmAreaOneObjOne[0], CurAlarmAreaOneObjOne[1], color = '', marker = '+', edgecolors='r', s = 100)
            # WrongLying
            if i_alarm == WrongLyingIndex and WrongLyingIndexFlag==True:
                for i_area in CurFrameMultiSensorAlarmInfoOne:
                    CurAlarmAreaOne = CurFrameMultiSensorAlarmInfoOne[i_area]['ObjInfo']
                    if len(CurAlarmAreaOne) == 0:
                        continue
                    for i_obj in range(CurAlarmAreaOne.shape[0]):
                        CurAlarmAreaOneObjOne = CurAlarmAreaOne[i_obj,:]
                        plt.scatter(CurAlarmAreaOneObjOne[0], CurAlarmAreaOneObjOne[1], color = '', marker = 'x', edgecolors='r', s = 100)

                        
        # 显示多传感器输出信息
        AloneValidFlag = -1
        SupervisorValidFlag = -1
        AboveHeightValidFlag = -1
        NoLyingValidFlag = -1
        WrongLyingValidFlag = -1
        for i_alarm in CurMultiFrameMultiSensorAlarmInfo:
            CurMultiFrameMultiSensorAlarmInfoOne = CurMultiFrameMultiSensorAlarmInfo[i_alarm]
            if len(CurMultiFrameMultiSensorAlarmInfoOne) == 0:
                continue
            # HumanAloneIndex
            if i_alarm == HumanAloneIndex and HumanAloneShowFlag==True:
                for i_obj in CurMultiFrameMultiSensorAlarmInfoOne:
                    CurAlarmObjOne = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                    if CurAlarmObjOne['AlarmState'] == 1:
                        AloneValidFlag = 1
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = 'o', edgecolors='g', s = 350)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
                    else:
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = 'o', edgecolors='g', s = 200)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
            # InternalSupervisorIndex
            if i_alarm == InternalSupervisorIndex and InternalSupervisorShowFlag==True:
                for i_obj in CurMultiFrameMultiSensorAlarmInfoOne:
                    CurAlarmObjOne = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                    if CurAlarmObjOne['AlarmState'] == 1:
                        SupervisorValidFlag = 1
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '*', edgecolors='g', s = 350)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
                    else:
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '*', edgecolors='g', s = 200)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
            # AboveHeightIndex
            if i_alarm == AboveHeightIndex and AboveHeightShowFlag==True:
                for i_obj in CurMultiFrameMultiSensorAlarmInfoOne:
                    CurAlarmObjOne = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                    if CurAlarmObjOne['AlarmState'] == 1:
                        AboveHeightValidFlag = 1
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '^', edgecolors='g', s = 350)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
                    else:
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '^', edgecolors='g', s = 200)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
            # NoLyingIndex
            if i_alarm == NoLyingIndex and NoLyingIndexFlag==True:
                for i_obj in CurMultiFrameMultiSensorAlarmInfoOne:
                    CurAlarmObjOne = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                    if CurAlarmObjOne['AlarmState'] == 1:
                        NoLyingValidFlag = 1
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '+', edgecolors='g', s = 350)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
                    else:
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = '+', edgecolors='g', s = 200)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
            # WrongLyingIndex
            if i_alarm == WrongLyingIndex and WrongLyingIndexFlag==True:
                for i_obj in CurMultiFrameMultiSensorAlarmInfoOne:
                    CurAlarmObjOne = CurMultiFrameMultiSensorAlarmInfoOne[i_obj]
                    if CurAlarmObjOne['AlarmState'] == 1:
                        WrongLyingValidFlag = 1
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = 'x', edgecolors='g', s = 350)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')
                    else:
                        plt.scatter(CurAlarmObjOne['ObjX'], CurAlarmObjOne['ObjY'], color = '', marker = 'x', edgecolors='g', s = 200)
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']+0.05, int(CurAlarmObjOne['SumFrame']), color ='g')
                        plt.text(CurAlarmObjOne['ObjX']+0.05, CurAlarmObjOne['ObjY']-0.1, int(CurAlarmObjOne['AlarmID']), color ='g')

        # title
        if len(CurFusionInputStrInfo) > 50:
            CurFusionInputStrInfoShow = CurFusionInputStrInfo[:50]
        else:
            CurFusionInputStrInfoShow = CurFusionInputStrInfo
        CurFrmTitle = 'Frm = {}, Time = {} \n {} \n Alone = {}, Supervisor = {}, AboveHeightValidFlag = {}'.format(str(i_frame).zfill(5), CurFrameWorldTime, CurFusionInputStrInfoShow, AloneValidFlag, SupervisorValidFlag, AboveHeightValidFlag)
        plt.title(CurFrmTitle)

        if FigAxisXYRange is not None:
            plt.axis('equal')
            plt.xlim([FigAxisXYRange[0], FigAxisXYRange[1]])
            plt.ylim([FigAxisXYRange[2], FigAxisXYRange[3]])
        else:
            plt.axis('equal')
                    
        plt.show()
        # save figure
        if os.path.exists(PlotImageSaveDirName):
            CurFrameSaveName = os.path.join(PlotImageSaveDirName, str(i_frame).zfill(5) + '.png')
            plt.savefig(CurFrameSaveName, dpi=300)
            
            
#        if i_frame > 100:
#            break
#        break
    
    
    return 0
    
    
if __name__ == '__main__':
    print('Start')
    
    TestReadMultiSensorDeteFusionInfoFlag = 0
    TestPlotMultiSensorDeteFusionInfoFlag = 1
    
    if TestReadMultiSensorDeteFusionInfoFlag == 1:
        LogFileName = 'ZT_20200413_test.txt'  
#        LogFileName = 'ZT_20200413.txt'    
        
        # 读取 log 文件信息
        MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName)
        print('MultiSensorDeteFusionInputInfo = {}'.format(MultiSensorDeteFusionInputInfo))
        print('MultiSensorDeteFusionOutputInfo = {}'.format(MultiSensorDeteFusionOutputInfo))
        
    if TestPlotMultiSensorDeteFusionInfoFlag == 1:

        LogFileType = 3 # if LogFileType == 1, 20200418 之前的log 文件格式
                        # if LogFileType == 2, 20200418 之后的log 文件格式
                        # if LogFileType == 3, 20200509 之后的msg_log 文件格式
                        
        if LogFileType == 1:
            # 节点 log 文件地址
    #        LogFileName = 'ZT_20200413_test.txt'  
    #        LogFileName = 'ZT_20200413.txt'
            LogFileName = 'ZT_20200417.txt'
            # 服务器中配置文件
            if LogFileName == 'ZT_20200417.txt':
                # 20200417
                Area1 = [2.5260, 0.0720, 4.0510, 0.0720, 4.0510, 1.7810, 2.5260, 1.7810]
                Area2 = [0.1910, 0.9630, 0.3520, 0.9630, 0.3520, 1.3430, 0.1910, 1.3430]
                Area3 = [0.3880, 1.7740, 2.5340, 1.7740, 2.5340, 3.1180, 0.3880, 3.1180]
                Area4 = [2.5260, 1.7740, 4.0370, 1.7740, 4.0370, 3.1320, 2.5260, 3.1320]
                Area5 = [0.3960, 0.0720, 2.5340, 0.0720, 2.5340, 1.7810, 0.3960, 1.7810]
            else:
                # 20200416
                Area1 = [2.3960, 0.0070, 4.2800, 0.0070, 4.2800, 1.4970, 2.3960, 1.4970]
                Area2 = [-0.6040, 0.6040, 0.0700, 0.6040, 0.0700, 1.5960, -0.6040, 1.5960]
                Area3 = [0.5690, 1.4970, 2.3960, 1.4970, 2.3960, 2.9590, 0.5690, 2.9590]
                Area4 = [2.4030, 1.4970, 4.2800, 1.4970, 4.2800, 2.9730, 2.4030, 2.9730]
                Area5 = [0.5550, 0.0070, 2.3960, 0.0070, 2.3960, 1.4970, 0.5550, 1.4970]
            MultiSensorArea = [Area1, Area2, Area3, Area4, Area5]
            MultiSensorName = ['10001', '10002', '10003', '10004', '10005']
            # 新建 log 显示结果地址
            PlotImageSaveName = LogFileName.split('.txt')[0]
            PlotImageSaveDirName = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\MultiSensorDeteResult', PlotImageSaveName)
            if not os.path.exists(PlotImageSaveDirName):
                os.mkdir(PlotImageSaveDirName)
            # read log info
            MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = LogFileType)

        elif LogFileType == 2:
            # 节点 log 文件地址
            LogFileName = os.path.join('log','202005081400.txt')
            # 服务器中配置文件
            if True:
                # 20200417
                Area1 = [2.5260, 0.0720, 4.0510, 0.0720, 4.0510, 1.7810, 2.5260, 1.7810]
                Area2 = [0.1910, 0.9630, 0.3520, 0.9630, 0.3520, 1.3430, 0.1910, 1.3430]
                Area3 = [0.3880, 1.7740, 2.5340, 1.7740, 2.5340, 3.1180, 0.3880, 3.1180]
                Area4 = [2.5260, 1.7740, 4.0370, 1.7740, 4.0370, 3.1320, 2.5260, 3.1320]
                Area5 = [0.3960, 0.0720, 2.5340, 0.0720, 2.5340, 1.7810, 0.3960, 1.7810]
            MultiSensorArea = [Area1, Area2, Area3, Area4]
            MultiSensorName = ['10001', '10002', '10003', '10004']
            # 新建 log 显示结果地址
            PlotImageSaveName = os.path.basename(LogFileName).split('.txt')[0]
            PlotImageSaveDirName = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\MultiSensorDeteResult', PlotImageSaveName)
            if not os.path.exists(PlotImageSaveDirName):
                os.mkdir(PlotImageSaveDirName)
            # read log info
            MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = LogFileType)

        elif LogFileType == 3:
            SelectCaseName = 'ZT' # 选择场景数据， ZT/LG
            
            if SelectCaseName == 'ZT':
                # 节点 log 文件地址
                LogFileName = os.path.join('log','zt_msg_log_20200509.txt') # gbk
#                LogFileName = os.path.join('log','zt_msg_log_test.txt') # utf-8

                # 服务器中配置文件
                if True:
                    # 20200417
                    Area1 = [2.5260, 0.0720, 4.0510, 0.0720, 4.0510, 1.7810, 2.5260, 1.7810]
                    Area2 = [0.1910, 0.9630, 0.3520, 0.9630, 0.3520, 1.3430, 0.1910, 1.3430]
                    Area3 = [0.3880, 1.7740, 2.5340, 1.7740, 2.5340, 3.1180, 0.3880, 3.1180]
                    Area4 = [2.5260, 1.7740, 4.0370, 1.7740, 4.0370, 3.1320, 2.5260, 3.1320]
                    Area5 = [0.3960, 0.0720, 2.5340, 0.0720, 2.5340, 1.7810, 0.3960, 1.7810]
                MultiSensorArea = [Area1, Area2, Area3, Area4]
                MultiSensorName = ['10001', '10002', '10003', '10004']
                # 新建 log 显示结果地址
                PlotImageSaveName = os.path.basename(LogFileName).split('.txt')[0]
                PlotImageSaveDirName = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\MultiSensorDeteResult', PlotImageSaveName)
                if not os.path.exists(PlotImageSaveDirName):
                    os.mkdir(PlotImageSaveDirName)
            elif SelectCaseName == 'LG':
                # 节点 log 文件地址
                LogFileName = os.path.join('log','lg_msg_log_20200508.txt')
                # 服务器中配置文件
                if True:
                    # 20200417
                    Area1 = [2.5260, 0.0720, 4.0510, 0.0720, 4.0510, 1.7810, 2.5260, 1.7810]
                    Area2 = [0.1910, 0.9630, 0.3520, 0.9630, 0.3520, 1.3430, 0.1910, 1.3430]
                    Area3 = [0.3880, 1.7740, 2.5340, 1.7740, 2.5340, 3.1180, 0.3880, 3.1180]
                    Area4 = [2.5260, 1.7740, 4.0370, 1.7740, 4.0370, 3.1320, 2.5260, 3.1320]
                    Area5 = [0.3960, 0.0720, 2.5340, 0.0720, 2.5340, 1.7810, 0.3960, 1.7810]
                MultiSensorArea = [Area1, Area2, Area3, Area4]
                MultiSensorName = ['10001', '10002', '10003', '10004']
                # 新建 log 显示结果地址
                PlotImageSaveName = os.path.basename(LogFileName).split('.txt')[0]
                PlotImageSaveDirName = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\MultiSensorDeteResult', PlotImageSaveName)
                if not os.path.exists(PlotImageSaveDirName):
                    os.mkdir(PlotImageSaveDirName)
            # read log info
            MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo = ReadMultiSensorDeteFusionInfo(LogFileName, LogFileType = LogFileType)

        # print MultiSensorDeteFusion info
        print('MultiSensorDeteFusionInputInfo size = {}'.format(len(MultiSensorDeteFusionInputInfo)))
        print('MultiSensorDeteFusionOutputInfo size = {}'.format(len(MultiSensorDeteFusionOutputInfo)))
        print(MultiSensorDeteFusionInputInfo[1])
#        print(MultiSensorDeteFusionOutputInfo[1])

        # plot log info
        if len(MultiSensorDeteFusionOutputInfo) > 0:
            PlotMultiSensorDeteFusionInfo(MultiSensorArea, MultiSensorName, MultiSensorDeteFusionInputInfo, MultiSensorDeteFusionOutputInfo, PlotImageSaveDirName)
        
        
    
    print('End')
    
    