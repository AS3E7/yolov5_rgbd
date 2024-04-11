# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 16:44:42 2020

@author: HYD
"""

import numpy as np
import time
import os
import csv
import argparse
import torch
import cv2
import copy
#from chainercv.utils import read_image

import logging as lgmsg
import re
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler

import detecter
from detect_alarm.AlarmString import CreatDetecter
from util.CloudPointFuns import ReadKinect2FromPly
#
from config import one_sensor_params, online_offline_type, log_info, alarm_level

"""
# log 信息
ConfigDebugInfo = log_info['debug_info']
OnOffLineType = online_offline_type
if OnOffLineType == 'OffLine':
    if int(ConfigDebugInfo[0]) > 0:
        
        SaveLogFolderName = 'mylog'
        #日志打印格式
        log_fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = lgmsg.Formatter(log_fmt)
        #创建TimedRotatingFileHandler对象
    #    time_str = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    #    file_name = time_str +'_data'
    
#        file_name = 'msg_log_' + time.strftime('%Y%m%d%H', time.localtime(time.time()))
        file_name = 'msg_log'
        file_full_name = os.path.join(SaveLogFolderName,file_name)
        
        log_file_handler = TimedRotatingFileHandler(filename=file_full_name, when="D", interval=1, backupCount=7)
        log_file_handler.suffix = "%Y-%m-%d_%H-%M_%S.log"
        log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_\d{2}.log$")
        log_file_handler.setFormatter(formatter)
        
    #    log_file_handler.setLevel(logging.DEBUG)
        
    #    log_file_stream_handler = logging.StreamHandler()  
    #    log_file_stream_handler.level = logging.FATAL
        
#        lgmsg.basicConfig(level=lgmsg.DEBUG)
    
    
#        lgmsg.root.setLevel(lgmsg.DEBUG)
##        lgmsg.root.setLevel(lgmsg.INFO)
#        log = lgmsg.getLogger()
#        log.addHandler(log_file_handler)
#    #    log.addHandler(log_file_stream_handler)
#        # 开始打印信息
#        log.info('开始导入算法模块')
#        log.error('开始导入算法模块2')
#        print('aaaaaaaaa')
##        logging.removeHandler(log_file_handler)



##        lgmsg.basicConfig(level=lgmsg.INFO)
#        lgmsg.root.setLevel(lgmsg.DEBUG)
#        log = lgmsg.getLogger()
#        log.addHandler(log_file_handler)




        
#        lgmsg.root.setLevel(lgmsg.DEBUG)
#        log = lgmsg.getLogger()
#        log_file_handler.setLevel(lgmsg.DEBUG)
#        log.addHandler(log_file_handler)
    

#        lgmsg.root.setLevel(lgmsg.DEBUG)
##        lgmsg.basicConfig(level=lgmsg.DEBUG)
#        log = lgmsg.getLogger()
#        log.addHandler(log_file_handler)
        
#        lgmsg.basicConfig(level=lgmsg.DEBUG)
        lgmsg.root.setLevel(lgmsg.DEBUG)
        log = lgmsg.getLogger()
        log.addHandler(log_file_handler)
        
        # 开始打印信息
        log.debug('开始导入算法模块')
        log.info('开始导入算法模块2')
        log.warning('开始导入算法模块3')
        log.error('开始导入算法模块4')
        
        lgmsg.debug('开始导入算法模块')
        lgmsg.info('开始导入算法模块2')
        lgmsg.warning('开始导入算法模块3')
        lgmsg.error('开始导入算法模块4')
        
        print('aaaaaaaaabb')

    if 0:
        # 打印文件保存地址
        SaveLogFolderName = 'mylog'
        #日志打印格式
        log_fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
        formatter = lgmsg.Formatter(log_fmt)
        #创建TimedRotatingFileHandler对象
    #    time_str = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    #    file_name = time_str +'_data'
    
#        file_name = 'msg_log_' + time.strftime('%Y%m%d%H', time.localtime(time.time()))
        file_name = 'msg_log'
        file_full_name = os.path.join(SaveLogFolderName,file_name)
        
        log_file_handler = TimedRotatingFileHandler(filename=file_full_name, when="D", interval=1, backupCount=7)
        log_file_handler.suffix = "%Y-%m-%d_%H-%M_%S.log"
        log_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_\d{2}.log$")
        log_file_handler.setFormatter(formatter)

        
    #    log_file_stream_handler = logging.StreamHandler()  
    #    log_file_stream_handler.level = logging.FATAL
        
    #    logging.basicConfig(level=logging.INFO)
        log = lgmsg.getLogger()
#        lgmsg.root.setLevel(lgmsg.DEBUG)
        log_file_handler.setLevel(lgmsg.INFO)
        log.addHandler(log_file_handler)
    #    log.addHandler(log_file_stream_handler)
        
        # 开始打印信息
        log.info('开始导入算法模块2')
        log.warning('开始导入算法模块3')
    #    log.info('开始导入算法模块')
        print('aaaaaaaaabbb')
    #    log.removeHandler(log_file_handler)
        
    #    log_warning.write_warning()
        
     
"""
    
def callPythonONAndOFFLine(DepthData, DepthWidth, DepthHeight, Pts0, WorldTime, DeteStr, FileName, RGBData = None, RGBWidth = 1920, RGBHeight = 1080):
    # inputs:
    #       Img0: depth data
    #       KintWidth: depth image width size
    #       KintHeight: depth image height size
    #       Pts0: point cloud data
    #       WTime: world time
    #       DetecterStr: detect result, string format
    #       FileName: file name, offline data format
    # outputs: 
    #       result: string format
    
    # [RGBData, RGBWidth, RGBHeight, DepthData, Pts, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr]
    # DataTypeFlag
    DataTypeFlag = [1,1,1,0,0,0,0,0,0,0]
    # ImageData
    if RGBData is None:
        RGBData = np.zeros([1920,1080,3]) 
        RGBWidth = 1920
        RGBHeight = 1080

    # DataTypeFlag
    if DataTypeFlag[0] == 1 and DataTypeFlag[1] == 1: 
        # 输入数据模式1：点云 + 深度
        
        # OnLine
        if len(FileName)==0:
            result = detecter.callPython(RGBData, RGBWidth, RGBHeight, DepthData, Pts0, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr)
        # OffLine
        elif len(FileName)>0:
            result = detecter.callPythonOffLine(RGBData, RGBWidth, RGBHeight, DepthData, Pts0, DepthWidth, DepthHeight, DataTypeFlag, WorldTime, DeteStr, FileName)


            
    return result
    

if __name__ == '__main__':

    print('Start')
#    lgmsg.root.setLevel(lgmsg.DEBUG)
    # image size
    KintWidth = 512
    KintHeight = 424
    
    TestCase = 72 # TestCase = 63, 测试LG数据，测试单个图像，无标注信息
                  # TestCase = 7, test multi image, NanShanData, 20181025, 171-177
                  # TestCase = 72, 监仓认证版, vvvvvv
                  # TestCase = 73, 龙岗新仓测试
                  # TestCase = 8, yolo测试 RGB 数据
                  # TestCase = 10, 测试image 数据
                  
                  
    if TestCase == 63: # 测试LG数据，测试单个图像，无标注信息
        # world time
        WTime = time.time()

        ######### 是否保存 图片结果 ##################
        SaveDeteImgResultFlag = 1 # if SaveDeteImgResultFlag == 1, save detect result; else, not save
        # LG-100 数据
#        SaveFolderSrc = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\NASPoseDetect_OneClass\result'
        # LG select 数据
        SaveFolderSrc = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ResultImage\LG_100'

        
        ######### TestModelName ##################
#        TestModelName = '20200326_01' # 
#        TestModelName = '20200326_02_NJ' # 
        TestModelName = '20200326_04_resume'
#        TestModelName = '20200326_03_resume_low'
        
#        TestModelIterName = 'ssd300_pose_9999.pth'
        TestModelIterName = 'v2.pth'
        
        MoldelFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\NASPoseDetect_OneClass\weights'
        ModelFileName = os.path.join(MoldelFolderName, TestModelName, TestModelIterName)
        # SaveFolderName
        SaveFolderName = os.path.join(SaveFolderSrc, TestModelName)
        if not os.path.exists(SaveFolderName):
            os.mkdir(SaveFolderName)

        ######### eval result ##################
        Predcsvname = 'result\\eval_result_colormap_' + TestModelName + '.csv' # colormap image test 2, model:           
        Predout = open(Predcsvname,"w",newline='')
        Predwriter = csv.writer(Predout,dialect = "excel")
        firstrow = ['filename','result']
        Predwriter.writerow(firstrow)

        ######### test images folder ##################
        # LG-100 数据
#        DepthImageFloderDir = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\perosn-PascalVOC-export_Depth\JPEGImages'
         # LG select 数据
        DepthImageFloderDir = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\Calib\Colormap'
       

        ######### model name ##################
        parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')

        parser.add_argument('--trained_model', default= ModelFileName, type=str, help='Trained state_dict file path to open') # 
        parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
        parser.add_argument('--visual_threshold', default=0.4, type=float, help='Final confidence threshold') # init: 0.5
        parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
        parser.add_argument('--pose_root', default='', help='Location of POSE root directory')
        args = parser.parse_args()
        
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
            
        if args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            
        # load net
        num_classes = len(POSE_CLASSES) + 1 # +1 background
        net = build_ssd('test', 300, num_classes) # initialize SSD
        #net = net.cuda()
        
        net.load_state_dict(torch.load(args.trained_model,  map_location='cpu')) # use only cpus
        if args.cuda:
            net.load_state_dict(torch.load(args.trained_model))
            net = net.cuda()
        
        net.eval()
        print('Finished loading model: {}!'.format(TestModelName))
            
        # images mean value
        #means = (122, 156, 98)
        means = (104, 117, 123)
        
#        ######### test images loop ################## 
#        for root, dirs, files in os.walk(DepthImageFloderDir):
#            for filename in files:
#                if filename.endswith('png'):
#                    # depth data
#                    OneDepthDir = os.path.join(DepthImageFloderDir, filename)
#                    Img0 = np.zeros([1,217088])
#                    print(filename)
#                    img_file =os.path.join(DepthImageFloderDir, filename[:-3] + 'png')
#
#                    # test callPython
#                    ori_img = read_image(img_file, color=True) # read img_file
#                    
##                    pred_bbox, pred_labels, pred_scores = detecter.test_one_depth_image(args.save_folder, net, args.cuda, ori_img, BaseTransform(net.size, (104, 117, 123)), thresh=args.visual_threshold)
#                    pred_bbox, pred_labels, pred_scores = detecter.test_one_depth_image_src(args.save_folder, net, args.cuda, ori_img, BaseTransform(net.size, (104, 117, 123)), thresh=args.visual_threshold)
#
#                    #for prdh in range(pred_scores.shape[0]): # save result
#                    for prdh in range(len(pred_scores)): # save result
#                        tempPrdHum1 = [filename[:-4], str((pred_bbox[prdh][1])),str((pred_bbox[prdh][0])),str((pred_bbox[prdh][3])),str((pred_bbox[prdh][2])),str((pred_labels[prdh])),str((pred_scores[prdh]))]               
#                        Predwriter.writerow(tempPrdHum1)
##                    pred_bbox_Cmp = np.dot(pred_bbox[0], np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]))
#                    pred_bbox_Cmp = pred_bbox
#                    
#                    pred_labels_Cmp = pred_labels
#                    pred_scores_Cmp = pred_scores
#                    # save result in Category Folders
#                    if SaveDeteImgResultFlag == 1:
##                        detecter.SaveImgDeteResult(img_file, filename[:-3], pred_bbox_Cmp, pred_labels_Cmp, pred_scores_Cmp)
#                        detecter.SaveImgDeteResult_2(SaveFolderName, img_file, filename[:-3], pred_bbox_Cmp, pred_labels_Cmp, pred_scores_Cmp)
#                    
#        # close
#        Predout.close()
        
        
    elif TestCase == 7: # test multi image, NanShanData, 20181025, 171-177
        # sensor name
        SensorName = '171'
        
        # read sensor config file
        SensorConfigFileName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\SensorConfig.ini' # SensorName
        
        # CreatDetecter, Initial State
        DetecterStr = CreatDetecter(SensorConfigFileName)
        print(DetecterStr)
        
        # read depth data folder
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-230000' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-200000' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-200000_0' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\176\\Depth2018-10-14-230000' # 176
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\176\\Depth2018-10-14-200000' # 176

        DepthFloderDir = 'V:\\PoseEstimate\\PlyData\\NSDAT\\' + SensorName + '\\Depth2018-10-20-120000' # SensorName
        
        for root, dirs, files in os.walk(DepthFloderDir):
            for filename in files:
                if filename.endswith('kdepth'): 
                    # select one frame data to test
#                    if not filename == 'Depth2018-10-14-200000(188).kdepth':
#                        continue
                    
                    # depth data
                    OneDepthDir = os.path.join(DepthFloderDir, filename)
                    with open(OneDepthDir, 'rb') as fid:
                        data_array = np.fromfile(fid, np.int16)
                    Img0 = data_array
                    # Cloud Point 
                    print(filename)
                    PointCloudName = filename[:-6] + 'kply'
                    OnePlyDir = os.path.join(DepthFloderDir, PointCloudName)
                    Pts0 = ReadKinect2FromPly(OnePlyDir) # 3*N
                    # SaveFileName
                    SaveFileName = SensorName + '_' + filename[:-7]
                    
                    # current time, world time 
                    WTime = time.time()

                    # test callPython
                    callPython_start = time.time()
                    
#                    result = callPython(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr) # new callPython function, 20181016
                    result = callPythonONAndOFFLine(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr, SaveFileName) # new callPython function, 20181030
                    
                    
                    DetecterStr = result
                    print("callPython result : ")
                    print(result)
                    
                    callPython_end = time.time()
                    print('callPython time = %f'%(callPython_end - callPython_start))
                    
                    # debug
                    break
                
    elif TestCase == 71: # 测试展厅监仓数据
        # sensor name
#        SensorName = '15'
#        SensorName = '17'
        SensorName = '18'
        
        # read sensor config file
        SensorConfigFileName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\SensorConfig.ini' # SensorName
        
        # CreatDetecter, Initial State
        DetecterStr = CreatDetecter(SensorConfigFileName)
        print(DetecterStr)
        
        # read depth data folder
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-230000' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-200000' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\171\\Depth2018-10-14-200000_0' # 171
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\176\\Depth2018-10-14-230000' # 176
#        DepthFloderDir = 'X:\\PoseEstimate\\PlyData\\NSDAT\\176\\Depth2018-10-14-200000' # 176

        # ZT 20200409 数据
#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\Ply', SensorName) # SensorName
        # ZT 20200411 数据
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data', SensorName) # SensorName
        # ZT 20200416 数据
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200416', SensorName) # SensorName

        # ZT 20200421 数据，窗户问题
#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LG_ZT\Data_20200421', SensorName) # SensorName

#        # ZT 20200423 数据， lying 数据
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200423', SensorName) # SensorName
        # ZT 20200429 数据， lying 数据
        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200429', SensorName) # SensorName

        
        for root, dirs, files in os.walk(DepthFloderDir):
            for filename in files:
                if filename.endswith('ply'): 
                    # select one frame data to test
                    
                    CurFrameIdx = int(filename.split('.ply')[0].split('_')[0])
                    
#                    if not (CurFrameIdx>1400 and CurFrameIdx<1450): # 超高
#                        continue

#                    if not filename == '700.ply': # 17
#                        continue
                    
                    # lying
#                    if not filename == '1185.ply': # 15
#                        continue
#                    if not (CurFrameIdx>1180 and CurFrameIdx<1190): # 15
#                        continue
#                    if not (CurFrameIdx>300 and CurFrameIdx<310): # 15
#                        continue
#                    if not (CurFrameIdx>296 and CurFrameIdx<307): # 15
#                        continue
                    
#                    if not filename == '1.ply': # 20200421, sensor-17
#                        continue
#                    if not (CurFrameIdx<5): # 超高
#                        continue

#                    if not (filename == '123_78.ply' or filename == '123_79.ply' or filename == '123_80.ply'): # 超高
#                        continue

#                    if not (filename == '123_77.ply' or filename == '123_78.ply' or filename == '123_79.ply' or filename == '123_80.ply' or filename == '123_82.ply' or filename == '123_83.ply'): # 超高
#                        continue

                    # Cloud Point 
                    print(CurFrameIdx, ',  ', filename)
                    PointCloudName = filename
                    OnePlyDir = os.path.join(DepthFloderDir, PointCloudName)
                    Pts0 = ReadKinect2FromPly(OnePlyDir) # 3*N
                    # SaveFileName
                    SaveFileName = SensorName + '_' + filename.split('.ply')[0]
                    
                    # depth data
#                    OneDepthDir = os.path.join(DepthFloderDir, filename)
#                    with open(OneDepthDir, 'rb') as fid:
#                        data_array = np.fromfile(fid, np.int16)
#                    Img0 = data_array

                    Img0 = Pts0[2,:]*1000

                    # current time, world time 
                    WTime = time.time()

                    # test callPython
                    callPython_start = time.time()
                    
#                    result = callPython(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr) # new callPython function, 20181016
                    result = callPythonONAndOFFLine(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr, SaveFileName) # new callPython function, 20181030
                    
                    
                    DetecterStr = result
                    print("callPython result : ")
                    print(result)
                    
                    callPython_end = time.time()
                    print('callPython time = %f'%(callPython_end - callPython_start))
                    
                    # debug
                    break
                
                
    elif TestCase == 72: # 测试展厅监仓认证版
        # sensor name
#        SensorName = '17'
        SensorName = '18'
#        SensorName = '19'

        lgmsg.info('111111')
        
        # read sensor config file
        SensorConfigFileName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\SensorConfig.ini' # SensorName
        
        # CreatDetecter, Initial State
        DetecterStr = CreatDetecter(SensorConfigFileName)
#        print(DetecterStr)
        
        # read depth data folder

#        # ZT 20200522 数据， lying 数据
#        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\ZTTest\19\Depth2020-05-22-190000') # SensorName

#        # ZT 20200522 数据， lying 数据
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200522', SensorName) # SensorName

        # ZT 20200618 数据, OVERHEIGHT
#        DepthFloderDir = os.path.join(r'C:\software\LGDete\demo\test_image') # SensorName

#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\temp\zt_19') # SensorName

        # LG-166
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\NewLG_Bbox_Data\Select_Images\CollectFrameData\20200727_C102\SrcData') # SensorName

        # ZT-18
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200728\18') # SensorName
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\ZT_Bbox_Data\SrcImage\Data\20200728\17') # SensorName
       
#        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\A_103\65')

        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\C_102_2\167\2020-07-29-171000')
#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\temp\167')

#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\temp\zt_19')

#        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\YanTao\LGData\20200828\高度异常配置数据\A102\ply')

        
        
        for root, dirs, files in os.walk(DepthFloderDir):
            for filename in files:
                if filename.endswith('ply'): 
                    # select one frame data to test
                    
                    CurFrameIdx = int(filename.split('.ply')[0].split('_')[-1])

#                    CurFrameIdx = int(filename.split('.ply')[0].split('_')[0])
#                    CurFrameIdx = int(filename.split('.ply')[0])
#                    print('CurFrameIdx = ', CurFrameIdx)

#                    CurFrameIdx = 0


                    # lying
#                    if not (CurFrameIdx>150 and CurFrameIdx<160): # 15
#                        continue

#                    if not (CurFrameIdx>52 and CurFrameIdx<62): # 15
#                        continue
                    
                    if not (CurFrameIdx==904): # 901
                        continue
#                    if not (CurFrameIdx>897 and CurFrameIdx<904): # 15
#                        continue
                    
#                    if not (CurFrameIdx==310): # 901
#                        continue
#                    if not (CurFrameIdx>300 and CurFrameIdx<320): # 15
#                        continue
#                    if not (CurFrameIdx>0 and CurFrameIdx<2): # 15
#                        continue
                    
#                    time.sleep(0.1)


#                    if not (CurFrameIdx==10): # 超高
#                        continue
#                    if not (CurFrameIdx>10 and CurFrameIdx<15): # 超高
#                        continue

                    # Cloud Point 
                    print(CurFrameIdx, ',  ', filename)
                    PointCloudName = filename
                    OnePlyDir = os.path.join(DepthFloderDir, PointCloudName)
                    Pts0 = ReadKinect2FromPly(OnePlyDir) # 3*N
                    # SaveFileName
                    SaveFileName = SensorName + '_' + filename.split('.ply')[0]
                    
                    # depth data
#                    OneDepthDir = os.path.join(DepthFloderDir, filename)
#                    with open(OneDepthDir, 'rb') as fid:
#                        data_array = np.fromfile(fid, np.int16)
#                    Img0 = data_array

                    Img0 = Pts0[2,:]*1000

                    # current time, world time 
                    WTime = time.time()

                    # test callPython
                    callPython_start = time.time()
                    
                    result = callPythonONAndOFFLine(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr, SaveFileName) # new callPython function, 20181030
                    
                    
                    DetecterStr = result
#                    print("callPython result : ")
#                    print(result)
                    
                    callPython_end = time.time()
                    print('callPython time = %f'%(callPython_end - callPython_start))
                    
                    # debug
                    break
        

    elif TestCase == 73: # 测试龙岗新仓数据
        # sensor name
#        SensorName = '101'
#        SensorName = '168'
#        SensorName = '166'
        SensorName = '167'
        
        # read sensor config file
        SensorConfigFileName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\SensorConfig.ini' # SensorName
        
        # CreatDetecter, Initial State
        DetecterStr = CreatDetecter(SensorConfigFileName)
        print(DetecterStr)
        
        # read depth data folder
        # NewLG
#        DepthFloderDir = os.path.join(r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox_20200318\NewLG_Bbox_Data\Select_Images\FrameData\20200617', SensorName) # SensorName

#        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\C_102_2', SensorName, '2020-07-10-153000') # SensorName
#        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\C_102_2', SensorName, '2020-07-10-113000') # SensorName
        DepthFloderDir = os.path.join(r'X:\PlyData\LGKSS\C_102_2', SensorName, '2020-07-01-134000') # SensorName

        
        for root, dirs, files in os.walk(DepthFloderDir):
            for filename in files:
                if filename.endswith('ply'): 
                    # select one frame data to test
                    
                    CurFrameIdx = int(filename.split('.ply')[0].split('_')[-1])
#                    CurFrameIdx = int(filename.split('.ply')[0].split('_')[0])
                    
                    # select data
#                    if not (CurFrameIdx==355): # 
#                        continue

#                    if not (CurFrameIdx==515): # 
#                        continue
                    
#                    if not (CurFrameIdx==798): # 
#                        continue

                    if not (CurFrameIdx==1542): # 
                        continue



#                    if not (CurFrameIdx>0 and CurFrameIdx<50): # 15
#                        continue
#                    if not (CurFrameIdx>49 and CurFrameIdx<100): # 15
#                        continue

                    # Cloud Point 
                    print(CurFrameIdx, ',  ', filename)
                    PointCloudName = filename
                    OnePlyDir = os.path.join(DepthFloderDir, PointCloudName)
                    Pts0 = ReadKinect2FromPly(OnePlyDir) # 3*N
                    # SaveFileName
                    SaveFileName = SensorName + '_' + filename.split('.ply')[0]
                    
                    # depth data
                    Img0 = Pts0[2,:]*1000

                    # current time, world time 
                    WTime = time.time()

                    # test callPython
                    callPython_start = time.time()
                    
#                    result = callPython(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr) # new callPython function, 20181016
                    result = callPythonONAndOFFLine(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr, SaveFileName) # new callPython function, 20181030
                    
                    
                    DetecterStr = result
                    print("callPython result : ")
                    print(result)
                    
                    callPython_end = time.time()
                    print('callPython time = %f'%(callPython_end - callPython_start))
                    
                    # debug
                    break
        
        
    elif TestCase == 8: # 测试展厅监仓认证版
        # sensor name
        SensorName = '18'
        
        # read sensor config file
        SensorConfigFileName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\SensorConfig.ini' # SensorName
        
        # CreatDetecter, Initial State
        DetecterStr = CreatDetecter(SensorConfigFileName)
        print(DetecterStr)
        
        # read depth data folder
        DepthFloderDir = os.path.join(r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\LGPoseDete\test_image') # SensorName

        RGBWidth = 480
        RGBHeight = 270
        
        for root, dirs, files in os.walk(DepthFloderDir):
            for filename in files:
                if filename.endswith('.ply'): 
                    # select one frame data to test

                    # Cloud Point 
                    PointCloudName = filename
                    OnePlyDir = os.path.join(DepthFloderDir, PointCloudName)
                    Pts0 = ReadKinect2FromPly(OnePlyDir) # 3*N
                    # SaveFileName
                    SaveFileName = SensorName + '_' + filename.split('.png')[0]
                    
                    # depth data
                    Img0 = Pts0[2,:]*1000
                    
                    # RGBData
                    OneRGBDir = os.path.join(DepthFloderDir, filename.replace('.ply', '.png'))
                    RGBDataSrc = cv2.imread(OneRGBDir) # (270, 480, 3)
                    RGBDataSrc = RGBDataSrc[:,:,[2,1,0]]
                    RGBData = RGBDataSrc.flatten()
                    print('RGBData size ', RGBData.shape)
                    
#                    im_rgb = np.reshape(RGBData, (RGBHeight, RGBWidth, 3))
#                    im_rgb_plt = copy.deepcopy(im_rgb)
#                    RGBImageData = cv2.cvtColor(im_rgb_plt, cv2.COLOR_BGR2RGB)
#                    cv2.imwrite('test_rgb.png', RGBImageData)
                    
                    
                    # current time, world time 
                    WTime = time.time()

                    # test callPython
                    callPython_start = time.time()
                    
#                    result = callPython(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr) # new callPython function, 20181016
                    result = callPythonONAndOFFLine(Img0, KintWidth, KintHeight, Pts0, WTime, DetecterStr, SaveFileName, RGBData = RGBData, RGBWidth = RGBWidth, RGBHeight = RGBHeight) # new callPython function, 20181030
                    
                    
                    DetecterStr = result
                    print("callPython result : ")
                    print(result)
                    
                    callPython_end = time.time()
                    print('callPython time = %f'%(callPython_end - callPython_start))    
    

    elif TestCase == 10: # 测试图片数据
        from detect_bbox.Bbox_Detecter import Bbox_Detecter
        from detect_bbox.SSD_Detect.Eval import PlotDeteResult
    
        print('TestCase = {}'.format(TestCase))
        TestImageFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\temp\NewLG\Calib\Colormap'
        SaveImageFolderName = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\temp\NewLG\Calib\Colormap_Result'
        
        for root, dirs, files in os.walk(TestImageFolderName):
            for filename in files:
                if filename.endswith('.png'): 
                    print('filename = ', filename)

                    # read data
                    CurImageFullFileName = os.path.join(TestImageFolderName, filename)
                    ImageData = cv2.imread(CurImageFullFileName)
#                    print('ImageData size = ', ImageData.shape)
                    # (424, 512, 3) --> (3, 424, 512)
                    ProcessDepth = np.transpose(ImageData,(2,0,1)) # (424, 512, 3),[RGR]
                    ProcessDepth_1 = ProcessDepth[[2,1,0],:,:] # (424, 512, 3),[RGB]
                    ProcessDepth = np.array(ProcessDepth_1)
#                    print('ProcessDepth ', ProcessDepth.shape)
                    
                    # 图像数据检测
                    SelectDeteMethodFlag = 1
                    CurImgDeteResult = dict()
                    if SelectDeteMethodFlag == 1:
                        PredBbox, PredLabels, PredScores = Bbox_Detecter(ProcessDepth)
                        CurImgDeteResult['Bbox'] = PredBbox
                        CurImgDeteResult['Label'] = PredLabels
                        CurImgDeteResult['Score'] = PredScores
                    CurImgDeteResult['Data'] = ProcessDepth
                    
                    # save image
                    ImgDeteResult = CurImgDeteResult
                    ResultSaveFileName = os.path.join(SaveImageFolderName, 'result_' + filename)
                    SaveImageData = np.array(ImgDeteResult['Data'])
                    PlotDeteResult(SaveImageData, ImgDeteResult['Bbox'], ImgDeteResult['Label'], ImgDeteResult['Score'], ResultSaveFileName)

                    
                    
    print('End.')
    
    
    
    

    