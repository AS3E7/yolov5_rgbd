# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:02:17 2021

@author: Administrator
"""
import argparse
import numpy as np
import os
import torch
import glob
import cv2
import time
import copy
from tqdm import tqdm
from shutil import copyfile
from pathlib import Path

from models.experimental import attempt_load
from utils.datasets import LoadImage_set
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,set_logging)
from utils.torch_utils import select_device
from utils.datasets import trans_rgb2depth
from utils.cal_pc import compute_ious, match_ious_result

class DetectDatasetImages():
    def __init__(self, dataset_dir, weights_dir, save_dir, opt):
        self.dataset_dir = dataset_dir
        self.weights_dir = weights_dir
        self.save_dir = save_dir
        self.opt = opt
        
    def detect_dataset_images(self, dataset_dir, weights_dir, save_dir):
        """
        功能：检测数据图像结果
        数据：train/val
        """
        # all accuracy info
        AllAccInfo = dict()
        AllAccInfo['frame'] = dict()
        AllAccInfo['object_3Class'] = dict()
        AllAccInfo['object_2Class'] = dict()
        
        opt = self.opt
        # save detect result info
        SaveResultAccFielName = os.path.join(opt.output, str(round(time.time()))+'.txt')
        FpSaveResultAccFielName = open(SaveResultAccFielName, 'w')
        
        # load model
        model = attempt_load(weights_dir)  # load FP32 model
        device = select_device(opt.device)
        model.to(device).eval()
        cls_names = model.module.names if hasattr(model, 'module') else model.names
        
        # train
        trainval_type = 'train' + '2014'
        dataset_image_train = os.path.join(dataset_dir, 'images', trainval_type)
        dataset_label_train = os.path.join(dataset_dir, 'labels', trainval_type)
        dataset_label_train_txt = os.path.join(os.path.dirname(dataset_dir), trainval_type+'.txt')
        dataset_save_dir = os.path.join(save_dir)
        SumDeteTime = DetectDatasetImages.detect_images(dataset_image_train, dataset_label_train, model, txt_dir=dataset_label_train_txt, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, dataset_dir=self.dataset_dir)
        print('{}, each frame time = {}'.format(trainval_type, SumDeteTime[0,0]/SumDeteTime[0,1]), file=FpSaveResultAccFielName)
        # frame acc
        cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_train, dataset_label_train, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_train_txt)
        print('{}, frame_accuracy = {}, {}, {}'.format(trainval_type, cur_i, cur_l, cur_acc), file=FpSaveResultAccFielName)
        AllAccInfo['frame']['train'] = cur_acc
        # obj acc, [lie,sit,stand]
        cur_label_obj, cur_all_label_obj, cur_obj_acc = DetectDatasetImages.calc_dataset_acc_obj(dataset_image_train, dataset_label_train, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_train_txt, cls_names=cls_names, cls_trans=len(cls_names))
        print('    bbox_iou_match = \n{}'.format(cur_label_obj), file=FpSaveResultAccFielName)
        print('    bbox_iou_match_sum = \n{}'.format(cur_all_label_obj), file=FpSaveResultAccFielName)
        print('    overall precision = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 1]]))), file=FpSaveResultAccFielName)
        print('               recall = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 2]]))), file=FpSaveResultAccFielName)
        print('             accuracy = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj))), file=FpSaveResultAccFielName)   
        AllAccInfo['object_3Class']['train'] = cur_obj_acc
        # obj acc, [lie,nolie]
        cls_trans = 2
        cur_label_obj, cur_all_label_obj, cur_obj_acc = DetectDatasetImages.calc_dataset_acc_obj(dataset_image_train, dataset_label_train, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_train_txt, cls_names=cls_names, cls_trans=cls_trans)
        print('    cls = {}, bbox_iou_match = \n{}'.format(cls_trans, cur_label_obj), file=FpSaveResultAccFielName)
        print('    bbox_iou_match_sum = \n{}'.format(cur_all_label_obj), file=FpSaveResultAccFielName)
        print('    overall precision = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 1]]))), file=FpSaveResultAccFielName)
        print('               recall = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 2]]))), file=FpSaveResultAccFielName)
        print('             accuracy = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj))), file=FpSaveResultAccFielName)   
        AllAccInfo['object_2Class']['train'] = cur_obj_acc
        
        # val
        trainval_type = 'val' + '2014'
        dataset_image_val = os.path.join(dataset_dir, 'images', trainval_type)
        dataset_label_val = os.path.join(dataset_dir, 'labels', trainval_type)
        dataset_label_val_txt = os.path.join(os.path.dirname(dataset_dir), trainval_type+'.txt')
        dataset_save_dir = os.path.join(save_dir)
        SumDeteTime = DetectDatasetImages.detect_images(dataset_image_val, dataset_label_val, model, txt_dir=dataset_label_val_txt, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, dataset_dir=self.dataset_dir)
        print('{}, each frame time = {}'.format(trainval_type, SumDeteTime[0,0]/SumDeteTime[0,1]), file=FpSaveResultAccFielName)
        # frame acc
        cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_val, dataset_label_val, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_val_txt)
        print('{}, frame_accuracy = {}, {}, {}'.format(trainval_type, cur_i, cur_l, cur_acc), file=FpSaveResultAccFielName)
        AllAccInfo['frame']['val'] = cur_acc
        # obj acc, [lie,sit,stand]
        cur_label_obj, cur_all_label_obj, cur_obj_acc = DetectDatasetImages.calc_dataset_acc_obj(dataset_image_val, dataset_label_val, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_val_txt, cls_names=cls_names, cls_trans=len(cls_names))
        print('    bbox_iou_match = \n{}'.format(cur_label_obj), file=FpSaveResultAccFielName)
        print('    bbox_iou_match_sum = \n{}'.format(cur_all_label_obj), file=FpSaveResultAccFielName)
        print('    overall precision = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 1]]))), file=FpSaveResultAccFielName)
        print('               recall = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 2]]))), file=FpSaveResultAccFielName)
        print('             accuracy = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj))), file=FpSaveResultAccFielName)  
        AllAccInfo['object_3Class']['val'] = cur_obj_acc
        # obj acc, [lie,nolie]
        cls_trans = 2
        cur_label_obj, cur_all_label_obj, cur_obj_acc = DetectDatasetImages.calc_dataset_acc_obj(dataset_image_val, dataset_label_val, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type, txt_dir=dataset_label_val_txt, cls_names=cls_names, cls_trans=cls_trans)
        print('    cls = {}, bbox_iou_match = \n{}'.format(cls_trans, cur_label_obj), file=FpSaveResultAccFielName)
        print('    bbox_iou_match_sum = \n{}'.format(cur_all_label_obj), file=FpSaveResultAccFielName)
        print('    overall precision = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 1]]))), file=FpSaveResultAccFielName)
        print('               recall = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj[[0, 2]]))), file=FpSaveResultAccFielName)
        print('             accuracy = %.6f'%(cur_all_label_obj[0] / (np.sum(cur_all_label_obj))), file=FpSaveResultAccFielName)  
        AllAccInfo['object_2Class']['val'] = cur_obj_acc

        # print result
        print('frame_train, frame_val, object_3Class_train, object_3Class_val, object_2Class_train, object_2Class_val: ', file=FpSaveResultAccFielName)
        print('{:>4} {:>4} {:>4} {:>4} {:>4} {:>4}'.format(round(AllAccInfo['frame']['train'], 4), round(AllAccInfo['frame']['val'], 4), \
                                                           round(AllAccInfo['object_3Class']['train'], 4), round(AllAccInfo['object_3Class']['val'], 4), \
                                                           round(AllAccInfo['object_2Class']['train'], 4), round(AllAccInfo['object_2Class']['val'], 4)), file=FpSaveResultAccFielName)
        FpSaveResultAccFielName.close()
        
        print('  detect_dataset_images end')

        return 0
    
    # @classmethod
    def detect_images(image_dir, label_dir, model, txt_dir=None, save_dir=None, opt=None, trainval_type='train',dataset_dir=None):
        """
        功能：检测文件中目标图像,包括 与label 关系对比
        """
        # output image
        if not save_dir==None:
            path_visual = os.path.join(save_dir, 'visual', trainval_type)
            path_labels = os.path.join(save_dir, 'labels', trainval_type)
            if not os.path.exists(path_visual):
                os.makedirs(path_visual)
            if not os.path.exists(path_labels):
                os.makedirs(path_labels)
        
        # read filename
        files = []
        if (txt_dir == None) or (not os.path.exists(txt_dir)):
            if os.path.isdir(image_dir):
                files = sorted(glob.glob(os.path.join(image_dir, '*.*')))
            elif os.path.isfile(image_dir):
                files = [image_dir]
        else:
            fp = open(txt_dir, 'r') # read txt file
            files_lines = fp.readlines()
            fp.close()
            files = []
            for i_file in files_lines:    
                if dataset_dir == None:
                    cur_i_file = i_file.replace('./', 'data/').replace('.png\n', '.png') # trans current file name
                else:
                    cur_i_file_1 = i_file.replace('./coco/', '').replace('.png\n', '.png') # trans current file name
                    cur_i_file = os.path.join(dataset_dir, cur_i_file_1)
                files.append(cur_i_file)
                
        # loop images
        SaveTwoPlotResultFlag = opt.output_two_type_result
        names = model.module.names if hasattr(model, 'module') else model.names
        # sum detect time
        SumDeteTime = np.zeros([1, 2]) # [sum_time, sum_frame]
        with torch.no_grad():
            for i in files:
                # select data type
                if not opt.data_type == 'rgb':
                    if i.endswith('_RGB.png'): # exclude rgb images
                        continue
                # print('i:',i)
                
    
                folder_path, file_name = os.path.split(i)
                filename, extension = os.path.splitext(file_name)
                      
                if opt.data_channel <= 3:
                    img0 = cv2.imread(i)
                else:
                    # depth image
                    img0 = cv2.imread(i)
                    # rgb image
                    path_2 = i.replace('.png', '_RGB.png') # 对应的 RGB文件名
                    img_2 = cv2.imread(path_2) # 读取图像文件
                    dest_image_width = img0.shape[1]
                    dest_image_height = img0.shape[0]
                    if not img0.shape[0] == img_2.shape[0]: # 如果img_2 图像已经变换了，则不需要 trans_rgb2depth 操作
                        img_2 = trans_rgb2depth(img_2, opt.trans_matrix_file, dest_image_width, dest_image_height)# 转换RGB对应Depth关系
                    # [424,512,3] --> [424,512,6]
                    img0 = np.concatenate((img0, img_2),2)
                    
                    # img0 = img0.astype(np.uint8)
                    plot_img0 = img0[:,:,:3] # [424,512,3]
                    plot_img0 = plot_img0.astype(np.uint8)
                    plot_img1 = img0[:,:,3:] # [424,512,3]
                    plot_img1 = plot_img1.astype(np.uint8)
                    
                # detect 
                t1 = time.time()
                
                # h0, w0 = img0.shape[0], img0.shape[1]
                # r = opt.img_size / max(h0, w0)
                # interp = cv2.INTER_LINEAR
                # img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
                
                y = DetectDatasetImages.detect_one_image(img0, model, opt=opt)
                t2 = time.time()
                SumDeteTime[0, 0] = SumDeteTime[0, 0] + (t2-t1)
                SumDeteTime[0, 1] = SumDeteTime[0, 1] + 1
                
                if not save_dir==None:
                    # remove save_path
                    save_path = str(Path(path_labels) / Path(i).name) # save file name
                    if os.path.isfile(save_path + '.txt'):
                        os.remove(save_path + '.txt')
                    # plot image
                    if isinstance(y, dict): # 三种类型数据结果
                        if len(y['rgbd']) == 0 and len(y['rgb']) == 0 and len(y['depth']) == 0:
                            continue
                        
                        # rgbd
                        y_pred = y['rgbd']
                        cur_color = (0,255,0)
                        bbox = [resut.cpu().detach().numpy() for i, resut in enumerate(y_pred)]
                        for *xyxy, conf, cls in reversed(bbox):
                            # save predict bbox info
                            if opt.save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                            # plot predict bbox info
                            if img0.shape[2] <= 3:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                cv2.rectangle(img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 5)
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
                                cv2.rectangle(plot_img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 5)
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
                                    cv2.rectangle(plot_img1, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 5)
                                    tl = line_thickness=None or round(0.002 * (plot_img1.shape[0] + plot_img1.shape[1]) / 2) + 1  # line/font thickness
                                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                    if label:
                                        tf = max(tl - 1, 1)  # font thickness
                                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                        cv2.putText(plot_img1, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                                    lineType=cv2.LINE_AA)
                                        
                        # rgb
                        y_pred = y['rgb']
                        cur_color = (0,150,255)
                        bbox = [resut.cpu().detach().numpy() for i, resut in enumerate(y_pred)]
                        for *xyxy, conf, cls in reversed(bbox):
                            # save predict bbox info
                            # if opt.save_txt:  # Write to file
                            #     with open(save_path + '.txt', 'a') as file: #
                            #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                            # plot predict bbox info
                            if img0.shape[2] <= 3:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                cv2.rectangle(img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 2)
                                tl = line_thickness=None or round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
                                # color = colors or [random.randint(0, 255) for _ in range(3)]
                                c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                if label:
                                    tf = max(tl - 1, 1)  # font thickness
                                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                    cv2.putText(img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [cur_color[0], cur_color[1], cur_color[2]], thickness=tf,
                                                lineType=cv2.LINE_AA)
                            else:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                cv2.rectangle(plot_img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 2)
                                tl = line_thickness=None or round(0.002 * (plot_img0.shape[0] + plot_img0.shape[1]) / 2) + 1  # line/font thickness
                                # color = colors or [random.randint(0, 255) for _ in range(3)]
                                c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                if label:
                                    tf = max(tl - 1, 1)  # font thickness
                                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                    cv2.putText(plot_img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [cur_color[0], cur_color[1], cur_color[2]], thickness=tf,
                                                lineType=cv2.LINE_AA)
                                    
                                if SaveTwoPlotResultFlag==True:
                                    cv2.rectangle(plot_img1, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 2)
                                    tl = line_thickness=None or round(0.002 * (plot_img1.shape[0] + plot_img1.shape[1]) / 2) + 1  # line/font thickness
                                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                    if label:
                                        tf = max(tl - 1, 1)  # font thickness
                                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                        cv2.putText(plot_img1, label, (c1[0], c1[1] - 2), 0, tl / 4, [cur_color[0], cur_color[1], cur_color[2]], thickness=tf,
                                                    lineType=cv2.LINE_AA)
                                        
                        # depth
                        y_pred = y['depth']
                        cur_color = (0,0,255)
                        bbox = [resut.cpu().detach().numpy() for i, resut in enumerate(y_pred)]
                        for *xyxy, conf, cls in reversed(bbox):
                            # # save predict bbox info
                            # if opt.save_txt:  # Write to file
                            #     with open(save_path + '.txt', 'a') as file:
                            #         file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                            # plot predict bbox info
                            if img0.shape[2] <= 3:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                cv2.rectangle(img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 2)
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
                                cv2.rectangle(plot_img0, (int([*xyxy][0]),int([*xyxy][1])), (int([*xyxy][2]),int([*xyxy][3])), cur_color, 2)
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
                                    cv2.rectangle(plot_img1, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 2)
                                    tl = line_thickness=None or round(0.002 * (plot_img1.shape[0] + plot_img1.shape[1]) / 2) + 1  # line/font thickness
                                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                    if label:
                                        tf = max(tl - 1, 1)  # font thickness
                                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                        cv2.putText(plot_img1, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                                    lineType=cv2.LINE_AA)
                    
                    else: # 一种类型数据结果
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
                    if opt.output_images ==True: # save plot images
                        if img0.shape[2] <= 3:
                            cv2.imwrite(save_name, img0)
                        else:
                            cv2.imwrite(save_name, plot_img0)
                            if SaveTwoPlotResultFlag==True:
                                cv2.imwrite(save_name1, plot_img1)
                    t2 = time.time()
        return SumDeteTime

    def detect_one_image(img0, model, opt=None):
        """
        功能：前向计算一张图像数据
        """
        # Set Dataloader
        dataset = LoadImage_set(img0, img_size=opt.img_size)
    
        img = torch.from_numpy(dataset).to(select_device(opt.device))
        # img = img.float()
        # img /= 255.0
    
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        pred, train_out = model(img, augment=opt.augment)
        
        # # save test img
        # test_save_filename = 'detect_images_one.txt'
        # fp=open(test_save_filename, 'w')
        # img_save = img.reshape(6, img.shape[2]*img.shape[3])
        # for i_line in range(img_save.shape[1]):
        #     cur_line = img_save[:,i_line].cpu().numpy()
        #     fp.writelines([str(round(cur_line[0], 4)), ' ', str(round(cur_line[1], 4)), ' ', str(round(cur_line[2], 4)), ' ', str(round(cur_line[3], 4)), ' ', str(round(cur_line[4], 4)), ' ', str(round(cur_line[5], 4)), '\n'])
        # fp.close()
        
        if len(train_out)>3: # Detect 6-output
            # 3Label pred    
            pred1 = [] # src rgbd
            pred2 = [] # RGB
            pred3 = [] # Depth
            det_all = dict()
            det_all['rgbd'] = []
            det_all['rgb'] = []
            det_all['depth'] = []
            
            # if 
            if len(train_out)==5:
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
    
            # src rgbd
            # Apply NMS
            # pred1 = non_max_suppression(pred1, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, merge=False)
            pred1 = non_max_suppression(pred1, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=False)

            # Process detections
            for i, det in enumerate(pred1):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    det_all['rgbd'] = det
                else:
                    det_all['rgbd'] = []
                    
            # RGB
            # Apply NMS
            pred2 = non_max_suppression(pred2, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred2):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    det_all['rgb'] = det
                else:
                    det_all['rgb'] = []
                    
            # Depth
            # Apply NMS
            pred3 = non_max_suppression(pred3, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred3):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    det_all['depth'] = det
                else:
                    det_all['depth'] = []
                    
            return det_all
                
        else:
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
        
        return 0
    
    def calc_dataset_acc(image_dir, label_dir, save_dir=None, opt=None, trainval_type='train', txt_dir=None):
        """
        功能：计算数据集检测结果的性能
        """
        # path_mistake
        path_visual = os.path.join(save_dir, 'visual', trainval_type)
        path_labels = os.path.join(save_dir, 'labels', trainval_type)
        path_mistake = os.path.join(save_dir, 'mistake', trainval_type)
        if not os.path.exists(path_mistake):
            os.makedirs(path_mistake)
        # calculate
        i =0
        l =0
        coco_file = os.path.join(label_dir, '*.txt')
        yolov3_file = os.path.join(path_labels, '*.png.txt')
        coco_file_dirs = glob.glob(coco_file)
        yolov3_file_dirs = glob.glob(yolov3_file)
        if len(glob.glob(coco_file)) > 1.5*len(glob.glob(yolov3_file)):
            if not txt_dir==None:
                coco_file_dirs = []
                yolov3_file_dirs = []
                # read txt_dir
                fp = open(txt_dir, 'r') # read txt file
                files_lines = fp.readlines()
                fp.close()
                files = []
                for i_file in files_lines:    
                    cur_i_file = os.path.basename(i_file).replace('.png\n', '.txt') # trans current file name
                    coco_file_dirs.append(os.path.join(label_dir, cur_i_file)) # coco_file
                    cur_i_file_pred = cur_i_file.replace('.txt', '.png.txt')
                    yolov3_file_dirs.append(os.path.join(path_labels, cur_i_file_pred)) # yolov3_file
        
        coco = os.path.dirname(os.path.dirname(label_dir))
        yolov3 = save_dir
        train2014 = trainval_type
        images= r'images'
        labels = r'labels'
        output = path_mistake
        
        # for seq_dets_fn, seq_dets_fm in zip(coco_file_dirs, yolov3_file_dirs):
        for ii, seq_dets_fn in enumerate(coco_file_dirs):
            seq_dets_fm = yolov3_file_dirs[ii]
            
            seq = seq_dets_fn[coco_file.find('*'):].split('/')[0]
            
            if not opt.data_type == 'rgb':
                if seq.endswith('_RGB.txt'):
                    continue
            if seq.endswith('_Depth.txt'):
                continue
                
            img_seq = seq.split('.')[0]
            # print('img_seq:',img_seq)
            
            # if 'Depth-2020-10-29-180755_10003' == img_seq:
            #     print('img_seq')
                
            i += 1
    
            seq_det = []
            seq_dep = []
            fp = open(os.path.join(coco, labels, train2014, img_seq+'.txt'), 'r')
            if os.path.exists(yolov3+'/'+labels+'/'+train2014+'/'+img_seq + '.png.txt'):
                ft = open(os.path.join(yolov3, labels, train2014, img_seq + '.png.txt'), 'r')
            elif not os.path.exists(yolov3+'/'+labels+'/'+train2014+'/'+img_seq + '.png.txt'):
                count_fp = len(fp.readlines())
                # img = cv2.imread(yolov3+'/'+images+'/'+train2014+'/'+img_seq + '.png')
                img = cv2.imread(coco+'/'+images+'/'+train2014+'/'+img_seq + '.png')
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(count_fp), (20, 40), font, 1, (0, 0, 255), 5)
                cv2.imwrite(output+'/'+img_seq + '.png', img)
                
                if os.path.exists(coco+'/'+images+'/'+train2014+'/'+img_seq + '_RGB.png'):
                    img = cv2.imread(coco+'/'+images+'/'+train2014+'/'+img_seq + '_RGB.png')
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(count_fp), (20, 40), font, 1, (0, 0, 255), 5)
                    cv2.imwrite(output+'/'+img_seq + '_RGB.png', img)
                
                l += 1
                # i += 1
                continue
            # print(os.path.join(coco, labels, train2014, img_seq+'.txt'))
            # print(os.path.join(yolov3, labels, train2014, img_seq + '.png.txt'))
            count_fp = len(fp.readlines())
            count_ft = len(ft.readlines())
            # print('count_fp:',count_fp)
            # print('count_ft:', count_ft)
            
    
            
            if count_fp!=count_ft:
                # if '167_Depth2020-07-10-135000_02010' == img_seq:
                #     print('img_seq')
                
                # img = cv2.imread(yolov3 + '/' + images + '/' + train2014 + '/' + img_seq + '.png')
                img = cv2.imread(coco + '/' + images + '/' + train2014 + '/' + img_seq + '.png')
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(count_fp), (20, 40), font, 1, (0, 0, 255), 5)
                # print(output + '/' + img_seq + '.png')
                if os.path.exists(path_visual):
                    copyfile(os.path.join(path_visual, img_seq + '.jpg'), output + '/' + img_seq + '.png')
                    if os.path.exists(os.path.join(path_visual, img_seq + '_RGB.jpg')):
                        copyfile(os.path.join(path_visual, img_seq + '_RGB.jpg'), output + '/' + img_seq + '_RGB.png')
                else:
                    cv2.imwrite(output + '/' + img_seq + '.png', img)
                    if os.path.exists(coco + '/' + images + '/' + train2014 + '/' + img_seq + '_RGB.png'):
                        img = cv2.imread(coco + '/' + images + '/' + train2014 + '/' + img_seq + '_RGB.png')
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(img, str(count_fp), (20, 40), font, 1, (0, 0, 255), 5)
                        cv2.imwrite(output + '/' + img_seq + '_RGB.png', img)
    
                l += 1
        # print('i:',i)
        # print('l:',l)
        # print('accuracy:',(i-l)/i)
        if i==0:
            acc = 0
        else:
            acc = (i-l)/i
            
        return i, l, acc
    
    def calc_dataset_acc_obj(image_dir, label_dir, save_dir=None, opt=None, trainval_type='train', txt_dir=None, cls_names=['person'], cls_trans=3):
        """
        功能：计算数据集检测结果的性能，按目标个数计算
        """
        mistake_obj_folder_name = 'mistake_obj'
        iou_thresh = opt.iou_thres # 0.35/0.5/0.45
        label_name = cls_names # ['lie', 'sit', 'stand']
        if len(cls_names) == 3:
            if cls_trans==2:
                label_name = ['lie', 'nolie']
                label_name_trans_idx = [0, 1, 1]
                mistake_obj_folder_name = mistake_obj_folder_name + '_' + str(cls_trans)
                # print('{}, {}'.format(cls_trans, label_name))
            elif cls_trans==1:
                label_name = ['person']
                label_name_trans_idx = [0, 0, 0]
                mistake_obj_folder_name = mistake_obj_folder_name + '_' + str(cls_trans)
                # print('{}, {}'.format(cls_trans, label_name))
            else:
                label_name_trans_idx = [i for i in range(len(cls_names))]
        else:
            label_name_trans_idx = [i for i in range(len(cls_names))]
        
        iou_match_res = np.zeros([len(label_name), 3]) # objects ious
        image_iou_match_res = np.zeros([3]) # image ious
        
        # path_mistake
        path_visual = os.path.join(save_dir, 'visual', trainval_type)
        path_labels = os.path.join(save_dir, 'labels', trainval_type)
        path_mistake = os.path.join(save_dir, mistake_obj_folder_name, trainval_type)
        if not os.path.exists(path_mistake):
            os.makedirs(path_mistake)
        # calculate
        coco_file = os.path.join(label_dir, '*.txt')
        yolov3_file = os.path.join(path_labels, '*.png.txt')
        coco_file_dirs = glob.glob(coco_file)
        yolov3_file_dirs = glob.glob(yolov3_file)
        if len(glob.glob(coco_file)) > 1.5*len(glob.glob(yolov3_file)):
            if not txt_dir==None:
                coco_file_dirs = []
                yolov3_file_dirs = []
                # read txt_dir
                fp = open(txt_dir, 'r') # read txt file
                files_lines = fp.readlines()
                fp.close()
                for i_file in files_lines:    
                    cur_i_file = os.path.basename(i_file).replace('.png\n', '.txt') # trans current file name
                    coco_file_dirs.append(os.path.join(label_dir, cur_i_file)) # coco_file
                    cur_i_file_pred = cur_i_file.replace('.txt', '.png.txt')
                    yolov3_file_dirs.append(os.path.join(path_labels, cur_i_file_pred)) # yolov3_file
        
        coco = os.path.dirname(os.path.dirname(label_dir))
        yolov3 = save_dir
        train2014 = trainval_type
        images= r'images'
        labels = r'labels'
        output = path_mistake
        
        # for seq_dets_fn, seq_dets_fm in zip(coco_file_dirs, yolov3_file_dirs):
        for ii, seq_dets_fn in enumerate(coco_file_dirs):
            # seq_dets_fm = yolov3_file_dirs[ii]
            
            seq = seq_dets_fn[coco_file.find('*'):].split('/')[0]
            
            if not opt.data_type == 'rgb':
                if seq.endswith('_RGB.txt'):
                    continue
            if seq.endswith('_Depth.txt'):
                continue
                
            img_seq = seq.split('.')[0]
            # print('img_seq:',img_seq)
            
            # if 'Depth-2020-10-29-180755_10003' == img_seq:
            #     print('img_seq')

            
            # read truth image
            cur_image_truth_name = os.path.join(coco, images, train2014, img_seq+'.png')
            cur_image_truth = cv2.imread(cur_image_truth_name)
            cur_image_truth_w = cur_image_truth.shape[1]
            cur_image_truth_h = cur_image_truth.shape[0]
            
            # read truth label info, [label, xc, yc, w, h]
            cur_image_truth_label_all_objs = []
            cur_image_truth_label_name = os.path.join(coco, labels, train2014, img_seq+'.txt')
            OneLabelInfoSrc = []
            fp_src = open(cur_image_truth_label_name, 'r')
            OneLabelInfoLines = fp_src.readlines()
            for i_OneLabelInfo in OneLabelInfoLines:
                for i_data in i_OneLabelInfo.split():
                    OneLabelInfoSrc.append(float(i_data))
            OneLabelInfoSrc = np.array(OneLabelInfoSrc)
            OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5]) 
            fp_src.close()
            cur_image_truth_label_info = dict() # [lie, sit, stand]
            for i_label_name in range(len(label_name)):
                cur_image_truth_label_info[i_label_name] = []
            for i_bbox in range(OneLabelInfoSrc.shape[0]):
                cur_obj_label_src = int(OneLabelInfoSrc[i_bbox,0]) # label name
                cur_obj_label = label_name_trans_idx[cur_obj_label_src]
                BboxXmin = int(cur_image_truth_w*(OneLabelInfoSrc[i_bbox, 1]-OneLabelInfoSrc[i_bbox, 3]/2))
                BboxYmin = int(cur_image_truth_h*(OneLabelInfoSrc[i_bbox, 2]-OneLabelInfoSrc[i_bbox, 4]/2))
                BboxXmax = int(cur_image_truth_w*(OneLabelInfoSrc[i_bbox, 1] + OneLabelInfoSrc[i_bbox, 3]/2))
                BboxYmax = int(cur_image_truth_h*(OneLabelInfoSrc[i_bbox, 2] + OneLabelInfoSrc[i_bbox, 4]/2))
                cur_obj_info_bbox = [BboxXmin, BboxYmin, BboxXmax, BboxYmax] # [x1, y1, x2, y2]
                cur_image_truth_label_info[cur_obj_label].append(cur_obj_info_bbox)
                cur_image_truth_label_all_objs.append([BboxXmin, BboxYmin, BboxXmax, BboxYmax, cur_obj_label, 1])

            # read pred label info, [x1, y1, x2, y2, label, score]
            cur_image_pred_label_all_objs = []
            cur_image_pred_label_name = os.path.join(yolov3, labels, train2014, img_seq+'.png.txt')
            if os.path.exists(cur_image_pred_label_name): # exist pred info txt
                OneLabelInfoSrc = []
                fp_src = open(cur_image_pred_label_name, 'r')
                OneLabelInfoLines = fp_src.readlines()
                for i_OneLabelInfo in OneLabelInfoLines:
                    for i_data in i_OneLabelInfo.split():
                        OneLabelInfoSrc.append(float(i_data))
                OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/6),6]) 
                fp_src.close()
                cur_image_pred_label_info = dict() # [lie, sit, stand]
                for i_label_name in range(len(label_name)):
                    cur_image_pred_label_info[i_label_name] = []
                for i_bbox in range(OneLabelInfoSrc.shape[0]): 
                    cur_obj_label_src = int(OneLabelInfoSrc[i_bbox,4]) # label name
                    cur_obj_label = label_name_trans_idx[cur_obj_label_src]
                    BboxXmin = OneLabelInfoSrc[i_bbox, 0]
                    BboxYmin = OneLabelInfoSrc[i_bbox, 1]
                    BboxXmax = OneLabelInfoSrc[i_bbox, 2]
                    BboxYmax = OneLabelInfoSrc[i_bbox, 3]
                    cur_obj_info_bbox = [BboxXmin, BboxYmin, BboxXmax, BboxYmax] # [x1, y1, x2, y2]
                    cur_image_pred_label_info[cur_obj_label].append(cur_obj_info_bbox)
                    cur_image_pred_label_all_objs.append([BboxXmin, BboxYmin, BboxXmax, BboxYmax, cur_obj_label, OneLabelInfoSrc[i_bbox,5]])
            else: # no exist pred info txt
                # print('pred file not exist: ', cur_image_pred_label_name)
                cur_image_pred_label_info = dict() # [lie, sit, stand]
                for i_label_name in range(len(label_name)):
                    cur_image_pred_label_info[i_label_name] = []
            
            # compare truth and pred object info
            curr_match_res = [0, 0, 0]
            for i_keys in cur_image_truth_label_info.keys():
                i_label_name = i_keys
                cur_label_truth_obj_info = cur_image_truth_label_info[i_label_name]
                cur_label_pred_obj_info = cur_image_pred_label_info[i_label_name]
                # match
                if len(cur_label_truth_obj_info)>0:
                    if len(cur_label_pred_obj_info)>0:
                        ious = compute_ious(np.array(cur_label_truth_obj_info), np.array(cur_label_pred_obj_info))
                        curr_match_res = match_ious_result(ious, iou_thresh) # [正，漏，误]
                        iou_match_res[i_label_name] += curr_match_res
                    elif len(cur_label_pred_obj_info)==0:
                        curr_match_res = [0, len(cur_label_truth_obj_info), 0] # [正，漏，误]
                        iou_match_res[i_label_name] += curr_match_res
                
            # save image
            img0 = cur_image_truth
            for cur_image_truth_obj_info in cur_image_truth_label_all_objs: # image truth label
                cur_color = (0,255,0)
                # save image truth label
                label = '%s %.2f' % (label_name[int(cur_image_truth_obj_info[4])], cur_image_truth_obj_info[5])
                x_1 = int(cur_image_truth_obj_info[0])
                y_1 = int(cur_image_truth_obj_info[1])
                x_2 = int(cur_image_truth_obj_info[2])
                y_2 = int(cur_image_truth_obj_info[3])
                cv2.rectangle(img0, (x_1, y_1), (x_2, y_2), cur_color, 6)
                tl = line_thickness=None or round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
                # color = colors or [random.randint(0, 255) for _ in range(3)]
                c1, c2 = (x_1, y_1), (x_2, y_2)
                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.putText(img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)
            for cur_image_pred_obj_info in cur_image_pred_label_all_objs: # image pred label
                cur_color = (0,0,255)
                # save image truth label
                label = '%s %.2f' % (label_name[int(cur_image_pred_obj_info[4])], cur_image_pred_obj_info[5])
                x_1 = int(cur_image_pred_obj_info[0])
                y_1 = int(cur_image_pred_obj_info[1])
                x_2 = int(cur_image_pred_obj_info[2])
                y_2 = int(cur_image_pred_obj_info[3])
                cv2.rectangle(img0, (x_1, y_1), (x_2, y_2), cur_color, 2)
                tl = line_thickness=None or round(0.002 * (img0.shape[0] + img0.shape[1]) / 2) + 1  # line/font thickness
                # color = colors or [random.randint(0, 255) for _ in range(3)]
                c1, c2 = (x_1, y_1), (x_2, y_2)
                if label:
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                    cv2.putText(img0, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                lineType=cv2.LINE_AA)
            # save image
            if curr_match_res[1]>0 or curr_match_res[2]>0:
                cur_save_name = output+'/' + img_seq + '.png'
                cv2.imwrite(cur_save_name, img0)
                    
        # sum 
        image_iou_match_res = np.sum(iou_match_res, axis=0)
        acc = image_iou_match_res[0]/np.sum(image_iou_match_res)

        return iou_match_res, image_iou_match_res, acc
    
    
    
    
class CalcValDeteAccParams():
    def __init__(self):
        self.output_images = []
        self.save_txt = []
        self.data_channel = []
        self.output_two_type_result = []
        self.img_size = []
        self.augment = []
        self.conf_thres = []
        self.iou_thres = []
        self.classes = []
        self.agnostic_nms = []
        self.device = []
        
        
def CalcValDeteAcc(dataset_dir, save_dir, model, device='', dataloader=None):
    """
    功能：计算验证集检测精度
    """
    trainval_type = 'val' + '2014'
    dataset_image_val = os.path.join(dataset_dir, 'images', trainval_type)
    dataset_label_val = os.path.join(dataset_dir, 'labels', trainval_type)
    dataset_label_val_txt = os.path.join(os.path.dirname(dataset_dir), trainval_type+'.txt')
    dataset_save_dir = os.path.join(save_dir)
    
    # parser = argparse.ArgumentParser()
    # opt = parser.parse_args()
    # opt.output_images = True
    # opt.save_txt = True
    # opt.data_channel = 6
    # opt.output_two_type_result = False
    # opt.img_size = 416
    # opt.augment = False
    # opt.conf_thres = 0.6
    # opt.iou_thres = 0.5
    # opt.classes=1
    # opt.agnostic_nms = True
    # opt.device = device
    
    opt = CalcValDeteAccParams()
    opt.output_images = True
    opt.save_txt = True
    opt.data_channel = 6
    opt.output_two_type_result = False
    opt.img_size = 416
    opt.augment = False
    opt.conf_thres = 0.6
    opt.iou_thres = 0.5
    opt.classes=1
    opt.agnostic_nms = True
    opt.device = device
    
    # load model
    # model = attempt_load(model_dir)  # load FP32 model
    device = select_device(opt.device)
    # model.to(device).eval()
    model.eval()
    
    SelectMethodFlag = 3 # SelectMethodFlag = 1,使用遍历单帧数据，测试结果
                         # SelectMethodFlag = 2,使用dataloader,batchsize=1,遍历单帧数据，测试结果
                         # SelectMethodFlag = 3,使用dataloader,batchsize=1,遍历单帧数据，不保存结果文件，计算精度 
    
    CalcDeteAccCompareFolderName = os.path.join(r'output\coco_958_compare', 'labels', trainval_type)
    
    if SelectMethodFlag == 1:
        # 方法1
        DetectDatasetImages.detect_images(dataset_image_val, dataset_label_val, model, txt_dir=dataset_label_val_txt, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type)
        # calc_dataset_acc
        dataset_label_val_select = CalcDeteAccCompareFolderName
        # cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_val, dataset_label_val, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type)
        cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_val, dataset_label_val_select, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type)

    elif SelectMethodFlag == 2:
        # 方法2
        # output image
        if not save_dir==None:
            path_visual = os.path.join(save_dir, 'visual', trainval_type)
            path_labels = os.path.join(save_dir, 'labels', trainval_type)
            if not os.path.exists(path_visual):
                os.makedirs(path_visual)
            if not os.path.exists(path_labels):
                os.makedirs(path_labels)
        # input
        SaveTwoPlotResultFlag = opt.output_two_type_result
        names = model.module.names if hasattr(model, 'module') else model.names
        half = False
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        
            filename = os.path.basename(paths[0])
            filename_fullname = os.path.join(dataset_image_val, filename)
            # depth image
            img0 = cv2.imread(filename_fullname)
            # rgb image
            path_2 = filename_fullname.replace('.png', '_RGB.png') # 对应的 RGB文件名
            img_2 = cv2.imread(path_2) # 读取图像文件
            dest_image_width = img0.shape[1]
            dest_image_height = img0.shape[0]
            if not img0.shape[0] == img_2.shape[0]: # 如果img_2 图像已经变换了，则不需要 trans_rgb2depth 操作
                img_2 = trans_rgb2depth(img_2, opt.trans_matrix_file, dest_image_width, dest_image_height)# 转换RGB对应Depth关系
            # [424,512,3] --> [424,512,6]
            img0 = np.concatenate((img0, img_2),2)
            
            # img0 = img0.astype(np.uint8)
            plot_img0 = img0[:,:,:3] # [424,512,3]
            plot_img0 = plot_img0.astype(np.uint8)
            plot_img1 = img0[:,:,3:] # [424,512,3]
            plot_img1 = plot_img1.astype(np.uint8)
        
            # Disable gradients
            with torch.no_grad():
                # Run model
                inf_out, train_out = model(img, augment=False)  # inference and training outputs
                # 网络output 数，Detect 3-output/6-output
                if len(train_out)>3: # Detect 6-output
                    if len(train_out) == 5: # Detect(P3, P4, P5, Branch1, Branch2)
                        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
                        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
                        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
                        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
                        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
                        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
                        inf_out = inf_out[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
                    elif len(train_out) == 9: # Detect(P3, P4, P5, Branch1_1,Branch1_2,Branch1_3, Branch2_1,Branch2_2,Branch2_3)
                        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
                        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3] # output1
                        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3] # output1
                        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
                        inf_out_num_bbox_type2_1 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
                        inf_out_num_bbox_type2_2 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output2
                        inf_out_num_bbox_type2_3 = train_out[5].shape[1] * train_out[5].shape[2] * train_out[5].shape[3] # output2
                        inf_out_num_bbox_type2 = inf_out_num_bbox_type2_1 + inf_out_num_bbox_type2_2 + inf_out_num_bbox_type2_3 # output2
                        inf_out_num_bbox_type3_1 = train_out[6].shape[1] * train_out[6].shape[2] * train_out[6].shape[3] # output3
                        inf_out_num_bbox_type3_2 = train_out[7].shape[1] * train_out[7].shape[2] * train_out[7].shape[3] # output3
                        inf_out_num_bbox_type3_3 = train_out[8].shape[1] * train_out[8].shape[2] * train_out[8].shape[3] # output3
                        inf_out_num_bbox_type3 = inf_out_num_bbox_type3_1 + inf_out_num_bbox_type3_2 + inf_out_num_bbox_type3_3 # output3
                        inf_out = inf_out[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
    
                # Run NMS
                pred1 = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=False)
        
                # Process detections
                det_all = dict()
                det_all['rgbd'] = []
                det_all['rgb'] = []
                det_all['depth'] = []
                for i, det in enumerate(pred1):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                        det_all['rgbd'] = det
                    else:
                        det_all['rgbd'] = []
        
                # save info 
                if not save_dir==None:
                    y = det_all
                    # remove save_path
                    save_path = str(Path(path_labels) / Path(filename).name) # save file name
                    if os.path.isfile(save_path + '.txt'):
                        os.remove(save_path + '.txt')
                    # plot image
                    if isinstance(y, dict):
                        if len(y['rgbd']) == 0 and len(y['rgb']) == 0 and len(y['depth']) == 0:
                            continue
                        
                        # rgbd
                        y_pred = y['rgbd']
                        cur_color = (0,255,0)
                        bbox = [resut.cpu().detach().numpy() for i, resut in enumerate(y_pred)]
                        for *xyxy, conf, cls in reversed(bbox):
                            # save predict bbox info
                            if opt.save_txt:  # Write to file
                                with open(save_path + '.txt', 'a') as file:
                                    file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                            # plot predict bbox info
                            if img0.shape[2] <= 3:
                                label = '%s %.2f' % (names[int(cls)], conf)
                                cv2.rectangle(img0, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 5)
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
                                cv2.rectangle(plot_img0, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 5)
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
                                    cv2.rectangle(plot_img1, ([*xyxy][0],[*xyxy][1]), ([*xyxy][2],[*xyxy][3]), cur_color, 5)
                                    tl = line_thickness=None or round(0.002 * (plot_img1.shape[0] + plot_img1.shape[1]) / 2) + 1  # line/font thickness
                                    # color = colors or [random.randint(0, 255) for _ in range(3)]
                                    c1, c2 = (int([*xyxy][0]), int([*xyxy][1])), (int([*xyxy][2]), int([*xyxy][3]))
                                    if label:
                                        tf = max(tl - 1, 1)  # font thickness
                                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                        cv2.putText(plot_img1, label, (c1[0], c1[1] - 2), 0, tl / 4, [255, 255, 255], thickness=tf,
                                                    lineType=cv2.LINE_AA)
        
                    save_name = os.path.join(path_visual, filename.split('.')[0] + '.jpg')
                    save_name1 = os.path.join(path_visual, filename.split('.')[0] + '_RGB.jpg')
                    if opt.output_images ==True: # save plot images
                        if img0.shape[2] <= 3:
                            cv2.imwrite(save_name, img0)
                        else:
                            cv2.imwrite(save_name, plot_img0)
                            if SaveTwoPlotResultFlag==True:
                                cv2.imwrite(save_name1, plot_img1)
               
        # calc_dataset_acc
        dataset_label_val_select = CalcDeteAccCompareFolderName
        # cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_val, dataset_label_val, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type)
        cur_i, cur_l, cur_acc = DetectDatasetImages.calc_dataset_acc(dataset_image_val, dataset_label_val_select, save_dir=dataset_save_dir, opt=opt, trainval_type=trainval_type)

    elif SelectMethodFlag == 3:
        # 方法3                            
        # output image
        if not save_dir==None:
            path_visual = os.path.join(save_dir, 'visual', trainval_type)
            path_labels = os.path.join(save_dir, 'labels', trainval_type)
            if not os.path.exists(path_visual):
                os.makedirs(path_visual)
            if not os.path.exists(path_labels):
                os.makedirs(path_labels)
        # input
        SaveTwoPlotResultFlag = opt.output_two_type_result
        names = model.module.names if hasattr(model, 'module') else model.names
        half = False
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        # calculate image accuracy
        acc_i =0 # dete image num
        acc_l =0 # dete wrong image num
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
            acc_i = acc_i + 1
            # image
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # target
            if isinstance(targets, dict):
                cur_image_label = targets['rgbd']
            else:
                cur_image_label = targets
                
                
            # filename
            filename = os.path.basename(paths[0])
            filename_fullname = os.path.join(dataset_image_val, filename)
            
            # # depth image
            # img0 = cv2.imread(filename_fullname)
            # # rgb image
            # path_2 = filename_fullname.replace('.png', '_RGB.png') # 对应的 RGB文件名
            # img_2 = cv2.imread(path_2) # 读取图像文件
            # dest_image_width = img0.shape[1]
            # dest_image_height = img0.shape[0]
            # if not img0.shape[0] == img_2.shape[0]: # 如果img_2 图像已经变换了，则不需要 trans_rgb2depth 操作
            #     img_2 = trans_rgb2depth(img_2, opt.trans_matrix_file, dest_image_width, dest_image_height)# 转换RGB对应Depth关系
            # # [424,512,3] --> [424,512,6]
            # img0 = np.concatenate((img0, img_2),2)
            
            # # img0 = img0.astype(np.uint8)
            # plot_img0 = img0[:,:,:3] # [424,512,3]
            # plot_img0 = plot_img0.astype(np.uint8)
            # plot_img1 = img0[:,:,3:] # [424,512,3]
            # plot_img1 = plot_img1.astype(np.uint8)
        
            # Disable gradients
            with torch.no_grad():
                # Run model
                inf_out, train_out = model(img, augment=False)  # inference and training outputs
                # 网络output 数，Detect 3-output/6-output
                if len(train_out)>3: # Detect 6-output
                    if len(train_out) == 5: # Detect(P3, P4, P5, Branch1, Branch2)
                        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
                        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3]
                        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3]
                        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
                        inf_out_num_bbox_type2 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
                        inf_out_num_bbox_type3 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output3
                        inf_out = inf_out[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
                    elif len(train_out) == 9: # Detect(P3, P4, P5, Branch1_1,Branch1_2,Branch1_3, Branch2_1,Branch2_2,Branch2_3)
                        inf_out_num_bbox_type1_1 = train_out[0].shape[1] * train_out[0].shape[2] * train_out[0].shape[3] # output1
                        inf_out_num_bbox_type1_2 = train_out[1].shape[1] * train_out[1].shape[2] * train_out[1].shape[3] # output1
                        inf_out_num_bbox_type1_3 = train_out[2].shape[1] * train_out[2].shape[2] * train_out[2].shape[3] # output1
                        inf_out_num_bbox_type1 = inf_out_num_bbox_type1_1 + inf_out_num_bbox_type1_2 + inf_out_num_bbox_type1_3
                        inf_out_num_bbox_type2_1 = train_out[3].shape[1] * train_out[3].shape[2] * train_out[3].shape[3] # output2
                        inf_out_num_bbox_type2_2 = train_out[4].shape[1] * train_out[4].shape[2] * train_out[4].shape[3] # output2
                        inf_out_num_bbox_type2_3 = train_out[5].shape[1] * train_out[5].shape[2] * train_out[5].shape[3] # output2
                        inf_out_num_bbox_type2 = inf_out_num_bbox_type2_1 + inf_out_num_bbox_type2_2 + inf_out_num_bbox_type2_3 # output2
                        inf_out_num_bbox_type3_1 = train_out[6].shape[1] * train_out[6].shape[2] * train_out[6].shape[3] # output3
                        inf_out_num_bbox_type3_2 = train_out[7].shape[1] * train_out[7].shape[2] * train_out[7].shape[3] # output3
                        inf_out_num_bbox_type3_3 = train_out[8].shape[1] * train_out[8].shape[2] * train_out[8].shape[3] # output3
                        inf_out_num_bbox_type3 = inf_out_num_bbox_type3_1 + inf_out_num_bbox_type3_2 + inf_out_num_bbox_type3_3 # output3
                        inf_out = inf_out[:, :inf_out_num_bbox_type1, :] # 416: [bs, 9009, 6]
    
                # Run NMS
                pred1 = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, merge=False)
            
            # if filename=='Depth-2020-10-29-115234_10009.png':
            #     print('pred1 = ', pred1)
            
            # image dete acc
            if pred1[0]==None:
                acc_l = acc_l + 1
            else:
                cur_image_label_obj_num = cur_image_label.shape[0]
                cur_image_dete_obj_num = pred1[0].shape[0]
                if not cur_image_label_obj_num == cur_image_dete_obj_num:
                    acc_l = acc_l + 1
                    # print('filename = ', filename)
        # test images dete acc
        cur_i = acc_i
        cur_l = acc_l
        cur_acc = (cur_i-cur_l)/cur_i
                                

    model.float()

    print('calc_dataset_acc = ', cur_i, cur_l, cur_acc)

    return cur_acc


if __name__ == "__main__":
    print('Start.')
    
    TestCase = 1
    
    if TestCase == 1:
        # TestModelName = ['Depth', 'RGB', 'RGBD-2Branch', 'RGBD-S', 'RGBD-M', 'RGBD-L']
        TestModelName = ['RGBD-2Branch', 'RGBD-S', 'RGBD-M', 'RGBD-L']
        
        TestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_zt_1133\coco' # data_class3_zt_1133
        # TestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_863_exclude_bbox\coco' # data_class3_863_exclude_bbox
        # TestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\dataset_test\coco' # RGBD_20210602_TestDepthError
        # TestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210706_NoAnnoAll\dataset_test\coco' # RGBD_20210706_NoAnnoAll
        
        
        # [lie, sit, stand], 0722
        TestModelWeightsName = [r'weights_c3_ch6_laterf_pretrain_0624\best_06241300.pt',
                                r'zt_weights_c3_ch6_0722\best_07221500.pt',
                                r'weights_2branch_fromimg_middle_0624\best_06232300.pt',
                                r'zt_weights_2branch_fromimg_large_0722\best_07221100.pt']
        SelectTestModelNameIdx = [0, 1, 0, 0]
        
        
        # # [lie, sit, stand], 0624
        # TestModelWeightsName = [r'weights_c3_ch6_laterf_pretrain_0624\best_06241300.pt',
        #                         r'weights_c3_ch6_0624\best_06240800.pt',
        #                         r'weights_2branch_fromimg_middle_0624\best_06232300.pt',
        #                         r'weights_2branch_fromimg_large_0624\best_06240000.pt']
        # SelectTestModelNameIdx = [0, 0, 1, 0]
        
        
        # [lie, sit, stand], [data_class3_863_exclude_bbox]
        # TestModelWeightsName = [r'weights_c3_ch6_laterf_pretrain_0630\best_07032100.pt',
        #                         r'weights_c3_ch6_0630\best_07031100.pt',
        #                         r'weights_2branch_fromimg_middle_0630\best_07031200.pt',
        #                         r'weights_2branch_fromimg_large_0630\best_07051800.pt']
        # SelectTestModelNameIdx = [1, 1, 1, 1]
        
        
        # 修改网络结构, 20210705
        # TestModelWeightsName = [r'weights_2Branch_Combine\best_07062200_concate-0-3-6.pt',
        #                         r'',
        #                         r'',
        #                         r'']
        # SelectTestModelNameIdx = [1, 0, 0, 0]
        
        
        TestDatasetClassNum = 3 # 3Class/1Class
        TestSaveResultFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD_2Branch_3Label\output\all_model_detect_result' # save folder name
        for i_model, CurTestModelName in enumerate(TestModelName):
            print('CurTestModelName = ', CurTestModelName)
            DataImageType = 'rgbd'
            InputDataChannelNum = 6
            if SelectTestModelNameIdx[i_model] == 0:
                print('  SelectTestModelNameIdx = ', SelectTestModelNameIdx[i_model])
                continue
            # CurModelProjectFolderName
            if CurTestModelName == 'Depth':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD'
                DataImageType = 'depth'
                InputDataChannelNum = 3
            if CurTestModelName == 'RGB':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD_RGB'
                DataImageType = 'rgb'
                InputDataChannelNum = 3
            if CurTestModelName == 'RGBD-2Branch':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD_2Branch_3Label'
                
                # CurModelProjectFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD_2Branch_3Label_Pretrain_CombineOutputs' # 修改网络结构
                
                DataImageType = 'rgbd'
                InputDataChannelNum = 6
            if CurTestModelName == 'RGBD-S':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD'
                DataImageType = 'rgbd'
                InputDataChannelNum = 6
            if CurTestModelName == 'RGBD-M':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD_2Branch'
                DataImageType = 'rgbd'
                InputDataChannelNum = 6
            if CurTestModelName == 'RGBD-L':
                CurModelProjectFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD_2Branch'
                DataImageType = 'rgbd'
                InputDataChannelNum = 6
            # weights folder name
            CurModelWeightsName = TestModelWeightsName[i_model]
            # file name
            SelectDatasetName = TestDatasetFolderName
            SelectWeightNameModel = os.path.join(CurModelProjectFolderName, 'runs\\evolve', CurModelWeightsName)
            if not os.path.exists(SelectWeightNameModel):
                print('  model weights={} not exist'.format(SelectWeightNameModel))
                continue

            SelectImageType = DataImageType
            TestTrainvalName = 'train2014'
            SelectInputDataChannelNum = InputDataChannelNum
            SelectConfThres = 0.6
            SelectIouThres = 0.5
            SavePlotResultImageFlag = True
            SavePlotResultLabelFlag = True
            SaveTwoPlotResultFlag = True # False, 保存两种图像结果
            # CurSelectDatasetAndModelName
            CurSelectDatasetAndModelName = ''
            CurSelectModelName = os.path.basename(SelectWeightNameModel).split('.')[0]
            CurSelectDatasetName = os.path.dirname(SelectDatasetName).split('\\')[-1]
            if CurSelectDatasetName.find('dataset_test')>-1: # 测试数据集，非dataset数据集
                CurSelectDatasetName = os.path.dirname(SelectDatasetName).split('\\')[-2]
            CurSelectDatasetAndModelName = CurSelectDatasetName + '_' + CurSelectModelName
            if len(CurSelectDatasetAndModelName) == 0:
                SelectOutputFolderName = os.path.join(TestSaveResultFolderName, CurTestModelName, os.path.dirname(SelectWeightNameModel).split('\\')[-1])
            else:
                SelectOutputFolderName = os.path.join(TestSaveResultFolderName, CurTestModelName, os.path.dirname(SelectWeightNameModel).split('\\')[-1], CurSelectDatasetAndModelName)
            if not os.path.exists(SelectOutputFolderName):
                os.makedirs(SelectOutputFolderName)
    
            # argparse
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', type=str, default=SelectWeightNameModel, help='initial weights path')
            parser.add_argument('--conf-thres', type=float, default=SelectConfThres, help='object confidence threshold') # 0.6
            parser.add_argument('--iou-thres', type=float, default=SelectIouThres, help='IOU threshold for NMS') # 0.5
            parser.add_argument('--trainval', type=str, default=TestTrainvalName, help='initial weights path')
            parser.add_argument('--data_channel', type=str, default=SelectInputDataChannelNum, help='initial weights path')
            parser.add_argument('--dataset_name', type=str, default=SelectDatasetName, help='initial weights path')
            parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
            parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
            parser.add_argument('--augment', default=False, action='store_true', help='augmented inference')
            parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
            parser.add_argument('--agnostic-nms', default=True, action='store_true', help='class-agnostic NMS')
            parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
            parser.add_argument('--output', type=str, default=SelectOutputFolderName, help='output-1 folder')  # output-1 folder
            parser.add_argument('--output_images', type=str, default=SavePlotResultImageFlag, help='output-1 folder')  # output-1 folder
            parser.add_argument('--output_labels', type=str, default=SavePlotResultLabelFlag, help='output-1 folder')  # output-1 folder
            parser.add_argument('--output_two_type_result', type=str, default=SaveTwoPlotResultFlag, help='output-1 folder')  # output-1 folder
            parser.add_argument('--data_type', type=str, default=SelectImageType, help='data type: rgb, depth, rgbd')
            
            opt = parser.parse_args()
            
            # input params
            DatasetName = opt.dataset_name
            WeightsFileName = opt.weights
            OutputFolderName = opt.output
            if len(OutputFolderName)>0 and (not os.path.exists(OutputFolderName)):
                os.makedirs(OutputFolderName)
    
            # detect_dataset_images
            CurDetectDatasetImages = DetectDatasetImages(DatasetName, WeightsFileName, OutputFolderName, opt)
            CurDetectDatasetImages.detect_dataset_images(DatasetName, WeightsFileName, OutputFolderName)
    
    
    