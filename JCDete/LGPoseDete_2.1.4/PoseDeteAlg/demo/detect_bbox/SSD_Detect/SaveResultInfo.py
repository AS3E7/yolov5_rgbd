# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:35:52 2019

@author: JiangZhuang XiongBiao

说明：使用标注过的数据测试模型得到结果文件（result.csv和测试结果图片.png）
"""

from __future__ import print_function
import os
import cv2
import csv
import time
import torch
import argparse
#import detecter
#import datetime
#from ssd import build_ssd
#from data import data_root
#from chainercv.utils import read_image
#from data import BaseTransform, POSE_CLASSES

def draw_bbox(ori_img, pred_bbox, pred_labels, pred_scores, pose_bbox_label_names):
    '''
    参数
    ----------
    pred_bbox(list) : 点的坐标（y1, x1, y2, x2）
    pred_labels(list) : 标签（第几类）
    pred_scores(list) : 类别相应的概率
    pose_bbox_label_names(tuple) : 标签名称
    
    return
    ----------
    ori_img(array) : 图片数据
    '''
    ori_img = ori_img.copy()
    
    for num in range(len(pred_bbox)):
        point = pred_bbox[num]
        label = pred_labels[num]
        score = pred_scores[num]
        font = cv2.FONT_HERSHEY_SIMPLEX#cv2的默认字体
        first_point = (int(point[1]), int(point[0]))#点1的坐标
        last_point = (int(point[3]), int(point[2]))#点2的坐标
        draw_point = (int(point[3]), int(point[0]-15))#标注框的坐标
        text_point = (int(point[1]+2), int(point[0]-6))#标注文字的坐标
        cv2.rectangle(ori_img, first_point, last_point, (255, 0, 255), 4)#画label框
#        cv2.rectangle(ori_img, first_point, draw_point, (255, 0, 0), -1)#画标注框
        cv2.rectangle(ori_img, first_point, draw_point, (255, 255, 255), -1)#画标注框
        label_test = pose_bbox_label_names[label]
        text = label_test + ':' + str(round(float(score), 3))
#        cv2.putText(ori_img,text, text_point,font, 0.4, (255, 255, 255), 1)#在标注框写文本
        cv2.putText(ori_img,text, text_point,font, 0.35, (0, 0, 0), 1)#在标注框写文本
    
    return ori_img
    
def write_result(csv_url_name, filename, pred_bbox, pred_labels, pred_scores, pose_bbox_label_names, *TestModelName):
    '''
    参数
    ----------
    csv_url_name(str) : csv文件路径及名称
    filename(str) : 文件路径
    pred_bbox(list) : 点的坐标（y1, x1, y2, x2）
    pred_labels(list) : 标签（第几类）
    pred_scores(list) : 类别相应的概率
    pose_bbox_label_names(tuple) : 标签名称
    TestModelName(str) : 模型名字
    '''
    Predout = open(csv_url_name, "a+", newline='')
    Predwriter = csv.writer(Predout, dialect = "excel")
    for num in range(len(pred_bbox)):
        csv_list = [filename, str(len(pred_bbox)), str(pred_bbox[num][0]), str(pred_bbox[num][1]), 
                    str(pred_bbox[num][2]), str(pred_bbox[num][3]), str(round(float(pred_scores[num]), 3)), 
                    str(pred_labels[num]), pose_bbox_label_names[pred_labels[num]]]
        Predwriter.writerow(csv_list)
    Predout.close()
    
def read_result(csv_url_name):
    '''
    功能: 读取检测结果信息文件
        csv 文件格式: [filename,box_num,y0,x0,y1,x1,score,label_num,label_str]
    输入:
        csv_url_name(str) : csv文件路径及名称
    输出:
        MultiObjInfo: 每个object 信息
    '''
    MultiObjInfo = []
    first_line = True
    with open(csv_url_name) as fr:
        for line in fr:
            OneObjInfo = {}
            if first_line:
                first_line = False
                continue
            if len(line.strip().split(',')) == 9:
                filename = line.strip().split(',')[0]
                label_str = line.strip().split(',')[-1]
                box_num,ymin,xmin,ymax,xmax,score,label_num, = map(float, line.strip().split(',')[1:-1])
                # file name
                filenameIdx = filename.rfind('.')
                filename = filename[0:filenameIdx]
                # OneObjInfo info
                OneObjInfo['filename'] = filename
                OneObjInfo['box_num'] = box_num
                OneObjInfo['ymin'] = ymin
                OneObjInfo['xmin'] = xmin
                OneObjInfo['ymax'] = ymax
                OneObjInfo['xmax'] = xmax
                OneObjInfo['score'] = score
                OneObjInfo['label'] = label_num
                OneObjInfo['label_str'] = label_str
            else:
                continue
            
            if len(OneObjInfo)>0:
                MultiObjInfo.append(OneObjInfo) 
    return MultiObjInfo

if __name__ == '__main__':
    number = 0
    # world time
    WTime = time.time()
    result_url = r''#eg:  'D:\algorithm project\Jiancang_about\ErKan_about\NASPoseDetect'
    test_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    ######### 是否保存 图片结果 ##################
    SaveDeteImgResultFlag = 1 # if SaveDeteImgResultFlag == 1, save detect result; else, not save
    SaveTrainResultFlag = 1 # 是否保存Train结果的csv
    SaveTestResultFlag = 1 # 是否保存Test结果的csv

    ######### TestModelName ##################
    TestModelName = 'ssd300_pose_9999' # new, vvv

    ######### test images folder ##################
    # 模拟测试数据
    if result_url == '':
        DepthImageFloderDir = r'result\test\pred\JPEGImages'
    else:
        DepthImageFloderDir = result_url + '\\' + r'result\test\pred\JPEGImages'
    
    ######### model name ##################
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--trained_model', default='weights/' + TestModelName + '.pth',
                        type=str, help='Trained state_dict file path to open') # 模型路径
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='Dir to save results')
    parser.add_argument('--visual_threshold', default=0.5, type=float,
                        help='Final confidence threshold') # init: 0.6
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use cuda to train model')
    parser.add_argument('--pose_root', default=data_root, help='Location of POSE root directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
        
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device('cpu')
        
    num_classes = len(POSE_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    
    net.load_state_dict(torch.load(args.trained_model,  map_location=device)) # use only cpus
    if args.cuda:
        net.load_state_dict(torch.load(args.trained_model))
        net = net.cuda()
    
    net.eval()
    print('Finished loading model!')
    
    pose_bbox_label_names = ('nolying', 'lying')

    begin_time = time.time()
    
    if result_url == '':
        result_csv_url_name = r'result\test\pred' + '\\' + test_time + '-' + TestModelName + '.csv'
    else:
        result_csv_url_name = result_url + r'\result\test\pred' + '\\' + test_time + '-' + TestModelName + '.csv'
    Predout = open(result_csv_url_name, "a+", newline='')
    Predwriter = csv.writer(Predout, dialect = "excel")
    csv_list = ['filename', 'box_num', 'y0', 'x0', 'y1' , 'x1', 'score', 'label_num', 'label_str']
    Predwriter.writerow(csv_list)
    Predout.close()
    
    ''' test images loop '''
    for _, _, files in os.walk(DepthImageFloderDir):
        for filename in files:
            if filename.endswith('png'):
                number += 1
                # depth data
                OneDepthDir = os.path.join(DepthImageFloderDir, filename)
                print(filename)
                img_file =os.path.join(DepthImageFloderDir, filename[:-3] + 'png')
                # test callPython
                ori_img = read_image(img_file, color=True) # read img_file
                
                pred_bbox, pred_labels, pred_scores = detecter.test_one_depth_image_src(args.save_folder, net, args.cuda, ori_img, BaseTransform(net.size, (104, 117, 123)), thresh=args.visual_threshold)

                ori_img = cv2.imread(img_file)
                
                ori_image = draw_bbox(ori_img, pred_bbox, pred_labels, pred_scores, pose_bbox_label_names)
                
                if result_url == '':
                    saveFile = 'result\\test\\pred\\png_box\\' + test_time
                else:
                    saveFile = result_url + '\\result\\test\\pred\\png_box\\' + test_time
                if not os.path.exists(saveFile):
                    os.mkdir(saveFile)
                saveFile = saveFile + '\\' + filename
                if SaveDeteImgResultFlag is 1:
                    cv2.imwrite(saveFile, ori_image)
                if SaveTestResultFlag is 1:
                    write_result(result_csv_url_name, filename, pred_bbox, pred_labels, pred_scores, pose_bbox_label_names, TestModelName)

    end_time = time.time()
    print('All test image : %d'%number)
    print('Test use time : %0.2f'%(end_time - begin_time))