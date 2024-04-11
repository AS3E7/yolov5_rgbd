# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:14:29 2019

@author: Administrator
"""

import io_hsf
import time
import random

DataPath = "//192.168.10.251/AllUsers/RangeData/FWS/SrcDepth/ZH_Sensor1"
DataName = "Color2019-11-11-111454"
DataName = "Depth2019-11-11-111454"
DeHsfPath = DataPath + "/" + DataName + ".HSF.HSF"
HsfPath = DataPath + "/" + DataName + ".HSF"
CsvPath = DataPath + "/" + DataName + ".csv"
DataType = DataName[:5]

FileInfo = io_hsf.GetHsfFileInfo(HsfPath, DataFormat=DataType)
DeFileInfo = io_hsf.GetHsfFileInfo(DeHsfPath, DataFormat=DataType)

assert FileInfo['FrameNum'] == DeFileInfo['FrameNum']
FrameSum = FileInfo['FrameNum']

write_data = '----------' + DataType + '----------\n'
write_data += 'FileName : ' + FileInfo['FileName'][:-4] + '\n'


    
TestCase = 2 #1：顺序测试 2：单帧测试 3：倒序测试 4：乱序测试

#---------顺序读取效率测试---------
if TestCase == 1:
    
    BeginTime = time.time()
    for i in range(FrameSum):
        CurFrmData, TempFrmTime = io_hsf.GetHSFDataFrames(HsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "顺序读取压缩数据总用时：%.04f\n"%(EndTime-BeginTime)
    
    BeginTime = time.time()
    for i in range(FrameSum):
        DeCurFrmData, DeTempFrmTime = io_hsf.GetHSFDataFrames(DeHsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "顺序读取解压数据总用时：%.04f\n"%(EndTime-BeginTime)
    
#---------单帧读取速率测试-------------
if TestCase == 2:
    FrameNum = random.randint(0, FrameSum)
    
    BeginTime = time.time()
    CurFrmData, TempFrmTime = io_hsf.GetHSFDataFrames(HsfPath, FrameNum, DataFormat=DataType)
    EndTime = time.time()
    write_data += "单帧读取压缩数据总用时：%.04f\n"%(EndTime-BeginTime)
    
    BeginTime = time.time()
    DeCurFrmData, DeTempFrmTime = io_hsf.GetHSFDataFrames(DeHsfPath, FrameNum, DataFormat=DataType)
    EndTime = time.time()
    write_data += "单帧读取解压数据总用时：%.04f\n"%(EndTime-BeginTime)
    
    assert TempFrmTime == DeTempFrmTime

#----------逆序读取效率测试------------
if TestCase == 3:
    FrameNumList = [x for x in range(FrameSum)]
                    
    BeginTime = time.time()
    for i in reversed(FrameNumList):
        CurFrmData, TempFrmTime = io_hsf.GetHSFDataFrames(HsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "逆序读取压缩数据总用时：%.04f\n"%(EndTime-BeginTime)
    
    BeginTime = time.time()
    for i in reversed(FrameNumList):
        DeCurFrmData, DeTempFrmTime = io_hsf.GetHSFDataFrames(DeHsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "逆序读取解压数据总用时：%.04f\n"%(EndTime-BeginTime)
    
#----------乱序读取效率测试-----------
if TestCase == 4:
    FrameNumList = [x for x in range(FrameSum)]
    random.shuffle(FrameNumList)
    
    BeginTime = time.time()
    for i in reversed(FrameNumList):
        CurFrmData, TempFrmTime = io_hsf.GetHSFDataFrames(HsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "乱序读取压缩数据总用时：%.04f\n"%(EndTime-BeginTime)
    
    BeginTime = time.time()
    for i in reversed(FrameNumList):
        DeCurFrmData, DeTempFrmTime = io_hsf.GetHSFDataFrames(DeHsfPath, i, DataFormat=DataType)
    EndTime = time.time()
    write_data += "乱序读取解压数据总用时：%.04f\n"%(EndTime-BeginTime)
    
write_data += '-------------------------\n'
with open("TestResult.txt", "a+") as fp:
    fp.write(write_data)