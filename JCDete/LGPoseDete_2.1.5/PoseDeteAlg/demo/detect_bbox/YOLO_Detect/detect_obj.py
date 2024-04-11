# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 20:52:08 2020

@author: huqiugen
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
from numpy import random
import glob
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImage_set
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,set_logging)
from utils.torch_utils import select_device
from utils.datasets import trans_rgb2depth


def detect(img0):

    # Set Dataloader
    dataset = LoadImage_set(img0, img_size=imgsz)

    img = torch.from_numpy(dataset).to(device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred, train_out = model(img, augment=opt.augment)
    
    if len(train_out)==3:
        pred = pred
    elif len(train_out)==5:
        # multi output
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
        # src rgbd
        pred1 = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
        pred2 = pred[:, inf_out_num_bbox_type1:inf_out_num_bbox_type1+inf_out_num_bbox_type2, :] # 416: [bs, 429, 6]
        pred3 = pred[:, -inf_out_num_bbox_type3:, :] # 416: [bs, 429, 6]
        pred = pred1
    
    elif len(train_out)==9:
        # multi output
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2_1 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type2_2 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3]
        inf_out_num_bbox_type2_3 = train_out[5].shape[1] * train_out[5].shape[2] * train_out[5].shape[3]
        inf_out_num_bbox_type2 = inf_out_num_bbox_type2_1 + inf_out_num_bbox_type2_2 + inf_out_num_bbox_type2_3
        inf_out_num_bbox_type3_1 = train_out[6].shape[1] * train_out[6].shape[2] * train_out[6].shape[3] # output3
        inf_out_num_bbox_type3_2 = train_out[7].shape[1] * train_out[7].shape[2] * train_out[7].shape[3]
        inf_out_num_bbox_type3_3 = train_out[8].shape[1] * train_out[8].shape[2] * train_out[8].shape[3]
        inf_out_num_bbox_type3 = inf_out_num_bbox_type3_1 + inf_out_num_bbox_type3_2 + inf_out_num_bbox_type3_3
        # src rgbd
        pred1 = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
        pred2 = pred[:, inf_out_num_bbox_type1:inf_out_num_bbox_type1+inf_out_num_bbox_type2, :] # 416: [bs, 9009, 6]
        pred3 = pred[:, -inf_out_num_bbox_type3:, :] # 416: [bs, 9009, 6]
        pred = pred1
        
    else:
        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
        pred = pred[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
        
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            return det
        else:
            return []
        
        
if __name__ == '__main__':
    
    ##########################################################
    # RGBD
    ##########################################################
    TestCase = 2 
    

    SelectWeightNameModel = r'weights\c3_2branch_ch6_07032100.pt'
    SelectInputDataChannelNum = 6
    

    if TestCase == 1: # 从图片开始测试
        ImageFolderName = r'data\test_data\test_images'
        ResultFolderName = r'data\test_data\result'
        TransMatrixFileName = r'data/rgb2depth_param.txt'
    elif TestCase == 2: # 从原始深度和RGB开始测试
        ImageFolderName = r'D:\xiongbiao\Code\LGPoseDete\RGBDHumanAnalysis\Train\data\test_data\test_ply'
        ResultFolderName = r'D:\xiongbiao\Code\LGPoseDete\RGBDHumanAnalysis\Train\data\test_data\result'
        # TransMatrixFileName = r'data/rgb2depth_param.txt'
        TransMatrixFileName = r'D:\xiongbiao\Code\LGPoseDete\RGBDHumanAnalysis\Train\data\RGB2DepthRotParams\ZT\15/rot_param.txt'
    
    SaveTwoPlotResultFlag = True
    OutputFolderName = ResultFolderName
    OutputVisualFolderName = ResultFolderName
    WeightsFileName = SelectWeightNameModel
    SourceFolderName = ImageFolderName
    
    if not os.path.exists(OutputFolderName):
        os.makedirs(OutputFolderName)
    if not os.path.exists(OutputVisualFolderName):
        os.makedirs(OutputVisualFolderName)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=WeightsFileName, help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
    
    parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--source', type=str, default=SourceFolderName, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=OutputFolderName, help='output-1 folder')  # output-1 folder

    parser.add_argument('--inputdata_channel_num', type=str, default=SelectInputDataChannelNum, help='output-1 folder')  # 输入数据通道数:3/6
    parser.add_argument('--trans_matrix_file', type=str, default=TransMatrixFileName, help='output-1 folder')  # 输入数据通道数:3/6


    opt = parser.parse_args()
    
    weights = opt.weights
    imgsz = opt.img_size
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # Eval模式
    model.to(device).eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # inputdata_channel_num
    inputdata_channel_num = opt.inputdata_channel_num
    trans_matrix_file = opt.trans_matrix_file
    path_visual = OutputVisualFolderName
    
    with torch.no_grad():
        # path_img = './data/samples'
        # path_visual = './data/visual'
        
        path_img = opt.source
        path_visual = path_visual
        
        if not os.path.exists(path_visual):
            os.makedirs(path_visual)
        path = str(Path(path_img))
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]
        for i in files:
            if i.endswith('_RGB.png'):
                continue
            print('i:',i)
            
            save_path = str(Path(opt.output) / Path(i).name) # save file name
            if os.path.isfile(save_path + '.txt'):
                os.remove(save_path + '.txt')
            
            folder_path, file_name = os.path.split(i)
            filename, extension = os.path.splitext(file_name)
                  
            if inputdata_channel_num <= 3:
                img0 = cv2.imread(i)
            else:
                # depth image
                img0 = cv2.imread(i)
                # rgb image
                # path_2 = i.replace('.png', '_RGB.png') 
                path_2 = i.replace('Depth', 'Color') # 对应的 RGB文件名
                img_2 = cv2.imread(path_2) # 读取图像文件
                dest_image_width = img0.shape[1]
                dest_image_height = img0.shape[0]
                if not img0.shape[0] == img_2.shape[0]: # 如果img_2 图像已经变换了，则不需要 trans_rgb2depth 操作
                    t11 = time.time()
                    img_2 = trans_rgb2depth(img_2, trans_matrix_file, dest_image_width, dest_image_height)# 转换RGB对应Depth关系
                    t12 = time.time()
                # [424,512,3] --> [424,512,6]
                img0 = np.concatenate((img0, img_2),2)
                
                # img0 = img0.astype(np.uint8)
                plot_img0 = img0[:,:,:3] # [424,512,3]
                plot_img0 = plot_img0.astype(np.uint8)
                plot_img1 = img0[:,:,3:] # [424,512,3]
                plot_img1 = plot_img1.astype(np.uint8)
                
            # detect 
            t1 = time.time()
            y = detect(img0)
            t2 = time.time()
            
            if len(y) == 0:
                continue
            
            bbox = [resut.cpu().detach().numpy() for i, resut in enumerate(y)]
            for *xyxy, conf, cls in reversed(bbox):
                # save predict bbox info
                if opt.save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                # plot predict bbox info
                if img0.shape[2] <= 3:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    cv2.rectangle(img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), (0,255,0), 3)
                    tl = line_thickness=None or round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                    if label:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.putText(img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)
                else:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    cv2.rectangle(plot_img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), (0,255,0), 3)
                    tl = line_thickness=None or round(0.002 * (plot_img0.shape[0] + plot_img0.shape[1]) / 2) + 1  # line/font thickness
                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                    if label:
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                        cv2.putText(plot_img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)
                        
                    if SaveTwoPlotResultFlag==True:
                        cv2.rectangle(plot_img1, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), (0,255,0), 3)
                        tl = line_thickness=None or round(0.002 * (plot_img1.shape[0] + plot_img1.shape[1]) / 2) + 1  # line/font thickness
                        # color = colors or [random.randint(0, 255) for _ in range(3)]
                        c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                        if label:
                            tf = max(tl - 1, 1)  # font thickness
                            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.putText(plot_img1, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                        lineType=cv2.LINE_AA)
                    
            save_name = os.path.join(path_visual, filename + '.jpg')
            save_name1 = os.path.join(path_visual, filename + '_RGB.jpg')
            if img0.shape[2] <= 3:
                cv2.imwrite(save_name, img0)
            else:
                cv2.imwrite(save_name, plot_img0)
                if SaveTwoPlotResultFlag==True:
                    cv2.imwrite(save_name1, plot_img1)
            
            print(y)
            print('detect time :',t2-t1)
            # print('rgb2depth :',t12-t11)
            
            # break
            
            
            
