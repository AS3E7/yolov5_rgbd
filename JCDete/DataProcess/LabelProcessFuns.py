# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:04:53 2021

@author: Administrator
"""

import os
import cv2
import numpy as np
import random
import glob
import xml.etree.ElementTree as ET
from shutil import copyfile
from DatasetTrainValSetting import ReSetDatasetTrainvalInfo

# 标注文件 3 类转为 1 类
def LabelTransClass3To1(SrcLabelFolderName, DestLabelFolderName, LabelPostfixName='.txt'):
    """
    功能：对标注类别转换，3类转为1类
    """
    # loop file
    for rt, dirs, files in os.walk(SrcLabelFolderName):
        for f in files:
            if f.endswith(LabelPostfixName):
                print('1 ', f)
                # read file
                OneLabelFileName = os.path.join(SrcLabelFolderName, f)
                OneDestLabelFileName = os.path.join(DestLabelFolderName, f)
                #   原始标注文件
                OneLabelInfoSrc = []
                fp_src = open(OneLabelFileName, 'r')
                OneLabelInfoLines = fp_src.readlines()
                for i_OneLabelInfo in OneLabelInfoLines:
                    for i_data in i_OneLabelInfo.split():
                        OneLabelInfoSrc.append(float(i_data))
                OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5])
                fp_src.close()
                #   变换 label
                OneLabelInfoDest = OneLabelInfoSrc
                OneLabelInfoDest[:, 0] = 0
                #   保存变换后标注文件
                fp_dest = open(OneDestLabelFileName, 'w')
                for i_bbox in range(OneLabelInfoDest.shape[0]):
                    CurBboxLine = ''
                    for i_data in range(OneLabelInfoDest[i_bbox].shape[0]):
                        if i_data == 0:
                            CurBboxLine = CurBboxLine + str(int(OneLabelInfoDest[i_bbox][i_data])) + ' '
                        else:
                            CurBboxLine = CurBboxLine + str(round(OneLabelInfoDest[i_bbox][i_data], 4)) + ' '
                    fp_dest.writelines(CurBboxLine + '\n')
                fp_dest.close()
                
    return 0

# 显示标注 bbox 信息
def PlotImageAnnoInfo(SrcLabelFolderName, SrcImageFolderName, DestImageFolderName, LabelPostfixName='.txt'):
    """
    功能：显示图像标注信息
    """
    # loop file
    for rt, dirs, files in os.walk(SrcLabelFolderName):
        for f in files:
            if f.endswith(LabelPostfixName):
                print('1 ', f)
                # read file
                OneLabelFileName = os.path.join(SrcLabelFolderName, f)
                OneImageFileName = os.path.join(SrcImageFolderName, f.replace(LabelPostfixName, '.png'))
                OneDestImageFileName = os.path.join(DestImageFolderName, f.replace(LabelPostfixName, '.png'))
                if not os.path.exists(OneImageFileName):
                    OneImageFileName = os.path.join(SrcImageFolderName, f.replace(LabelPostfixName, '.jpg'))
                    OneDestImageFileName = os.path.join(DestImageFolderName, f.replace(LabelPostfixName, '.jpg'))
                if not os.path.exists(OneLabelFileName) or not os.path.exists(OneImageFileName):
                    continue
                
                #   原始标注文件
                OneLabelInfoSrc = []
                fp_src = open(OneLabelFileName, 'r')
                OneLabelInfoLines = fp_src.readlines()
                for i_OneLabelInfo in OneLabelInfoLines:
                    for i_data in i_OneLabelInfo.split():
                        OneLabelInfoSrc.append(float(i_data))
                OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5])
                fp_src.close()
                # plot image
                OneImageData = cv2.imread(OneImageFileName)
                for i_bbox in range(OneLabelInfoSrc.shape[0]):
                    BboxXmin = int(OneImageData.shape[1]*(OneLabelInfoSrc[i_bbox, 1]-OneLabelInfoSrc[i_bbox, 3]/2))
                    BboxYmin = int(OneImageData.shape[0]*(OneLabelInfoSrc[i_bbox, 2]-OneLabelInfoSrc[i_bbox, 4]/2))
                    BboxXmax = int(OneImageData.shape[1]*(OneLabelInfoSrc[i_bbox, 1] + OneLabelInfoSrc[i_bbox, 3]/2))
                    BboxYmax = int(OneImageData.shape[0]*(OneLabelInfoSrc[i_bbox, 2] + OneLabelInfoSrc[i_bbox, 4]/2))
                    Color = (0,255,0)
                    if OneLabelInfoSrc[i_bbox, 0] == 0:
                        Color = (255,255,0)
                    elif OneLabelInfoSrc[i_bbox, 0] == 1:
                        Color = (0,255,255)
                    elif OneLabelInfoSrc[i_bbox, 0] == 2:
                        Color = (255,0,255)
                    
                    cv2.rectangle(OneImageData, (BboxXmin, BboxYmin), (BboxXmax, BboxYmax), Color, 3)
                    cv2.putText(OneImageData, str(int(OneLabelInfoSrc[i_bbox, 0])), (BboxXmin, BboxYmin-5), 0, 0.7, [255, 255, 255], lineType=cv2.LINE_AA)
                cv2.imwrite(OneDestImageFileName, OneImageData)
    
    return 0

class AnnoLabelTransXmlToTxt():
    def __init__(self, SrcAnnoXmlFolderName, DestAnnoTxtFolderName, ClassName):
        self.SrcAnnoXmlFolderName = SrcAnnoXmlFolderName
        self.DestAnnoTxtFolderName = DestAnnoTxtFolderName
        self.ClassName = ClassName
        self.SrcFilePostfixName = '.xml'
        self.DestFilePostfixName = '.txt'
    
    def MultiAnnoLabelTransXmlToTxt(self):
        """
        功能：遍历文件，将vott 标注的xml 文件转为 txt 文件
        """
        for root, dirs, files in os.walk(self.SrcAnnoXmlFolderName):
            for filename in files:
                if filename.endswith(self.SrcFilePostfixName):
                    CurSrcFileName = os.path.join(self.SrcAnnoXmlFolderName, filename)
                    CurDestFileName = os.path.join(self.DestAnnoTxtFolderName, filename.replace(self.SrcFilePostfixName, self.DestFilePostfixName))
                    # TransXmlToTxt
                    AnnoLabelTransXmlToTxt.TransXmlToTxt(CurSrcFileName, CurDestFileName, self.ClassName)
        return 0

    def TransXmlToTxt(SrcFileName, DestFileName, ClassName):
        """
        功能：将vott 标注的xml 文件转为 txt 文件
        """
        # open file
        classes = ClassName
        in_file = open(SrcFileName)
        out_file = open(DestFileName, 'w')
        # read xml info
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 在一个XML中每个Object的迭代
        for obj in root.iter('object'):
            # iter()方法可以递归遍历元素/树的所有子元素

            difficult = obj.find('difficult').text
            # 找到所有的椅子
            cls = obj.find('name').text
            # 如果训练标签中的品种不在程序预定品种，或者difficult = 1，跳过此object
            if cls not in classes or int(difficult) == 1:
                continue
            # cls_id 只等于1
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            # b是每个Object中，一个bndbox上下左右像素的元组
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = AnnoLabelTransXmlToTxt.convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(round(a, 4)) for a in bb]) + '\n')
        
        return 0
    
    def convert(size, box):
        """
        功能：转换bbox 坐标，[x1,x2,y1,y2]-->[xc,yc,w,h]
        """
        dw = 1. / (size[0])
        dh = 1. / (size[1])
        x = (box[0] + box[1]) / 2.0 - 1
        y = (box[2] + box[3]) / 2.0 - 1
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

class GenTrainvalDataset_RGBD3Label():
    def __init__(self, SrcImageFolderName, SrcLabelFolderName, DestImageFolderName, DestLabelFolderName, TrainRatio=0.8):
        self.SrcImageFolderName = SrcImageFolderName
        self.SrcLabelFolderName = SrcLabelFolderName
        self.DestImageFolderName = DestImageFolderName
        self.DestLabelFolderName = DestLabelFolderName
        self.TrainRatio = TrainRatio
    def GenRGBDTrainvalDatasetBaseOnBeforeDataset(self, BeforeDataset):
        """
        功能：在原有数据集基础上，生成train/val 数据集
        """
        SaveDestImageFlag = True
        # 读取原有数据文件名
        BeforeDatasetTrainFileName = BeforeDataset['TrainFileName']
        BeforeDatasetValFileName = BeforeDataset['ValFileName']
        #    read txt file
        BeforeDatasetTrainFileNameGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(BeforeDatasetTrainFileName)
        BeforeDatasetValFileNameGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(BeforeDatasetValFileName)
        BeforeDatasetTrainvalFileNameGroup = BeforeDatasetTrainFileNameGroup + BeforeDatasetValFileNameGroup
        
        # 读取现有数据文件名
        SrcImageFolderName = self.SrcImageFolderName
        SrcLabelFolderName = self.SrcLabelFolderName
        DestImageFolderName = self.DestImageFolderName
        DestLabelFolderName = self.DestLabelFolderName
        if not os.path.exists(self.DestImageFolderName):
            os.makedirs(self.DestImageFolderName)
        if not os.path.exists(self.DestLabelFolderName):
            os.makedirs(self.DestLabelFolderName)
        TrainRatio = self.TrainRatio
        ImageType = 'RGBD'
        FileTypePostfix = 'txt'
        BeforeDatasetTrainFileNameGroupValid = []
        BeforeDatasetValFileNameGroupValid = []
        if ImageType == 'RGBD':
            FilePrefixName_1 = 'Depth'
            FilePrefixName_2 = 'Color'
            RGBFilePostfixName = '_RGB' # 默认 RGBFilePostfixName = ''
            AllImageName_1 = [] # RGBD label filename
            AllImageName_2 = [] # RGB label filename
            for rt, dirs, files in os.walk(SrcLabelFolderName):
                for f in files:
                    if f.endswith('.'+FileTypePostfix):
                        # 排除之前数据集
                        if (f.replace('.txt', '.png') in BeforeDatasetTrainvalFileNameGroup) or (f.replace('.txt', '.png').replace('_RGB.', '.') in BeforeDatasetTrainvalFileNameGroup) or (f.replace('.txt', '.png').replace('_Depth.', '.') in BeforeDatasetTrainvalFileNameGroup):
                            if f.replace('.txt', '.png') in BeforeDatasetTrainFileNameGroup:
                                BeforeDatasetTrainFileNameGroupValid.append(f.replace('.txt', '.png'))
                            elif f.replace('.txt', '.png') in BeforeDatasetValFileNameGroup:
                                BeforeDatasetValFileNameGroupValid.append(f.replace('.txt', '.png'))
                            continue
                        # 排除 RGB数据
                        if len(RGBFilePostfixName)>0 and f.find(RGBFilePostfixName)>-1: # 是否为RGB 数据
                            # 是否存在对应图像数据
                            CurImageFileName = os.path.join(SrcImageFolderName, f.replace('.'+FileTypePostfix, '.png'))
                            if not os.path.exists(CurImageFileName):
                                continue
                            CurLabelFileName = f
                            AllImageName_2.append(CurLabelFileName)
                        elif f.find('_Depth.')==-1 and f.find('_RGB.')==-1: # 是否为Depth 原始文件名
                            # 是否存在对应图像数据
                            CurImageFileName = os.path.join(SrcImageFolderName, f.replace('.'+FileTypePostfix, '.png'))
                            if not os.path.exists(CurImageFileName):
                                continue
                            CurLabelFileName = f
                            AllImageName_1.append(CurLabelFileName)
            AllImageName = dict()
            AllImageName['Depth'] = AllImageName_1
            AllImageName['Color'] = AllImageName_2
        #    新添加有效的目标
        CurAddTrainFileNameGroupValid = []
        CurAddValFileNameGroupValid = []
        CurAddAllImageNum = len(AllImageName['Depth'])
        CurAddTrianImageNum = int(CurAddAllImageNum * TrainRatio)
        CurAddAllImageIdx = [x for x in range(CurAddAllImageNum)]
        random.shuffle(CurAddAllImageIdx)
        #    train
        for i_trainval in range(CurAddTrianImageNum): # [0, TrianImageNum]
            CurAddTrainFileNameGroupValid.append(AllImageName['Depth'][CurAddAllImageIdx[i_trainval]].replace('.txt', '.png'))
        #    val
        for i_trainval in range(CurAddTrianImageNum, CurAddAllImageNum, 1): # [0, TrianImageNum]
            CurAddValFileNameGroupValid.append(AllImageName['Depth'][CurAddAllImageIdx[i_trainval]].replace('.txt', '.png'))

        # 重新生成数据集 train/val
        if len(BeforeDatasetTrainFileNameGroup) == len(BeforeDatasetTrainFileNameGroupValid) and len(BeforeDatasetValFileNameGroup) == len(BeforeDatasetValFileNameGroupValid):
            NewDatasetTrainFileNameGroupValid = BeforeDatasetTrainFileNameGroup + CurAddTrainFileNameGroupValid
            NewDatasetValFileNameGroupValid = BeforeDatasetValFileNameGroup + CurAddValFileNameGroupValid
        else:
            NewDatasetTrainFileNameGroupValid = BeforeDatasetTrainFileNameGroupValid + CurAddTrainFileNameGroupValid
            NewDatasetValFileNameGroupValid = BeforeDatasetValFileNameGroupValid + CurAddValFileNameGroupValid
        DestTrainValImageFolderName = DestImageFolderName
        DestTrainValLabelFolderName = DestLabelFolderName
        #    save train
        TrainValTxtPrefixName = './coco/images/'
        TrainFileName = os.path.join(DestTrainValLabelFolderName, 'train2014.txt')
        fp = open(TrainFileName, 'w')
        if not os.path.exists(os.path.join(DestTrainValImageFolderName, 'train2014')):
            os.makedirs(os.path.join(DestTrainValImageFolderName, 'train2014'))
        if not os.path.exists(os.path.join(DestTrainValLabelFolderName, 'train2014')):
            os.makedirs(os.path.join(DestTrainValLabelFolderName, 'train2014'))
        for i_train in range(len(NewDatasetTrainFileNameGroupValid)): # [0, TrianImageNum]
            CurSrcImage = NewDatasetTrainFileNameGroupValid[i_train]
            CurImageName = TrainValTxtPrefixName + 'train2014/' + CurSrcImage
            fp.writelines(CurImageName + '\n')
            # copy file
            if SaveDestImageFlag == True:
                copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'train2014', CurSrcImage)) # Depth image
                copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '.txt'))) # Depth label
                if (os.path.exists(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_Depth.txt')))):
                    copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_Depth.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '_Depth.txt'))) # _Depth label
                if (os.path.exists(os.path.join(SrcImageFolderName, CurSrcImage.replace('.png', '_RGB.png')))):
                    copyfile(os.path.join(SrcImageFolderName, CurSrcImage.replace('.png', '_RGB.png')), os.path.join(DestTrainValImageFolderName, 'train2014', CurSrcImage.replace('.png', '_RGB.png'))) # RGB image
                if (os.path.exists(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_RGB.txt')))):
                    copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_RGB.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '_RGB.txt'))) # RGB label
        fp.close()
        #    save val
        TrainValTxtPrefixName = './coco/images/'
        TrainFileName = os.path.join(DestTrainValLabelFolderName, 'val2014.txt')
        fp = open(TrainFileName, 'w')
        if not os.path.exists(os.path.join(DestTrainValImageFolderName, 'val2014')):
            os.makedirs(os.path.join(DestTrainValImageFolderName, 'val2014'))
        if not os.path.exists(os.path.join(DestTrainValLabelFolderName, 'val2014')):
            os.makedirs(os.path.join(DestTrainValLabelFolderName, 'val2014'))
        for i_train in range(len(NewDatasetValFileNameGroupValid)): # [0, TrianImageNum]
            CurSrcImage = NewDatasetValFileNameGroupValid[i_train]
            CurImageName = TrainValTxtPrefixName + 'val2014/' + CurSrcImage
            fp.writelines(CurImageName + '\n')
            # copy file
            if SaveDestImageFlag == True:
                copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'val2014', CurSrcImage)) # Depth image
                copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '.txt'))) # Depth label
                if (os.path.exists(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_Depth.txt')))):
                    copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_Depth.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '_Depth.txt'))) # _Depth label
                if (os.path.exists(os.path.join(SrcImageFolderName, CurSrcImage.replace('.png', '_RGB.png')))):
                    copyfile(os.path.join(SrcImageFolderName, CurSrcImage.replace('.png', '_RGB.png')), os.path.join(DestTrainValImageFolderName, 'val2014', CurSrcImage.replace('.png', '_RGB.png'))) # RGB image
                if (os.path.exists(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_RGB.txt')))):
                    copyfile(os.path.join(SrcLabelFolderName, CurSrcImage.replace('.png', '_RGB.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '_RGB.txt'))) # RGB label
        fp.close()
        return 0
        
    def GenRGBDTrainvalDataset(self):
        """
        生成各类型数据 train/val 数据集
        输入：
            ImageType: 'Depth'/'RGB'/'RGBD'
        """
        SrcImageFolderName = self.SrcImageFolderName
        SrcLabelFolderName = self.SrcLabelFolderName
        DestImageFolderName = self.DestImageFolderName
        DestLabelFolderName = self.DestLabelFolderName
        TrainRatio = self.TrainRatio
        ImageType = 'RGBD'
        FileTypePostfix = 'txt'
        if ImageType == 'RGBD':
            FilePrefixName_1 = 'Depth'
            FilePrefixName_2 = 'Color'
            RGBFilePostfixName = '_RGB' # 默认 RGBFilePostfixName = ''
            AllImageName_1 = [] # RGBD label filename
            AllImageName_2 = [] # RGB label filename
            for rt, dirs, files in os.walk(SrcLabelFolderName):
                for f in files:
                    if f.endswith('.'+FileTypePostfix):
                        # 排除 RGB数据
                        if len(RGBFilePostfixName)>0 and f.find(RGBFilePostfixName)>-1: # 是否为RGB 数据
                            # 是否存在对应图像数据
                            CurImageFileName = os.path.join(SrcImageFolderName, f.replace('.'+FileTypePostfix, '.png'))
                            if not os.path.exists(CurImageFileName):
                                continue
                            CurLabelFileName = f
                            AllImageName_2.append(CurLabelFileName)
                        elif f.find('_Depth.')==-1 and f.find('_RGB.')==-1: # 是否为Depth 原始文件名
                            # 是否存在对应图像数据
                            CurImageFileName = os.path.join(SrcImageFolderName, f.replace('.'+FileTypePostfix, '.png'))
                            if not os.path.exists(CurImageFileName):
                                continue
                            CurLabelFileName = f
                            AllImageName_1.append(CurLabelFileName)
            AllImageName = dict()
            AllImageName['Depth'] = AllImageName_1
            AllImageName['Color'] = AllImageName_2
    
            GenTrainvalDataset_RGBD3Label.GenTrainvalDataset(SrcImageFolderName, SrcLabelFolderName, AllImageName, DestImageFolderName, DestLabelFolderName, TrainRatio=TrainRatio)

    def GenTrainvalDataset(SrcImageFolderName, SrcAnnoFolderName, AllImageNameInput, DestImageFolderName, DestLabelFolderName, TrainRatio=0.8, SaveDestImageFlag=True):
        """
        生成 train/val 数据集
        """
        TrainValTxtPrefixName = './coco/images/'
        if isinstance(AllImageNameInput,list):
            AllImageName = AllImageNameInput
            AllImageNum = len(AllImageName)
            AllImageName_2 = []
        else:
            AllImageName = AllImageNameInput['Depth']
            AllImageName_2 = AllImageNameInput['Color']
            AllImageNum = len(AllImageName)
        
        DestTrainValImageFolderName = DestImageFolderName
        DestTrainValLabelFolderName = DestLabelFolderName
        if not os.path.exists(DestTrainValImageFolderName):
            os.makedirs(DestTrainValImageFolderName)
        if not os.path.exists(DestTrainValLabelFolderName):
            os.makedirs(DestTrainValLabelFolderName)
        
        # AllImageIdx
        TrianImageNum = int(AllImageNum * TrainRatio)
        AllImageIdx = [x for x in range(AllImageNum)]
        random.shuffle(AllImageIdx)
        
        # save image
        # trainval
        TrainvalFileName = os.path.join(DestTrainValLabelFolderName, 'trainval2014.txt')
        fp = open(TrainvalFileName, 'w')
        for i_trainval in range(AllImageNum): # [0, TrianImageNum]
            CurImageName = TrainValTxtPrefixName + 'trainval/' + os.path.basename(AllImageName[AllImageIdx[i_trainval]].replace('.txt','.png'))
            fp.writelines(CurImageName + '\n')
        fp.close()
        # AllImageName_2
        if len(AllImageName_2)>0:
            TrainvalFileName_2 = os.path.join(DestTrainValLabelFolderName, 'trainval2014_2.txt')
            fp_2 = open(TrainvalFileName_2, 'w')
            for i_trainval in range(AllImageNum): # [0, TrianImageNum]
                CurImageName = TrainValTxtPrefixName + 'trainval/' + os.path.basename(AllImageName_2[AllImageIdx[i_trainval]].replace('.txt','.png'))
                fp_2.writelines(CurImageName + '\n')
            fp_2.close()
        
        # train
        TrainFileName = os.path.join(DestTrainValLabelFolderName, 'train2014.txt')
        fp = open(TrainFileName, 'w')
        if not os.path.exists(os.path.join(DestTrainValImageFolderName, 'train2014')):
            os.makedirs(os.path.join(DestTrainValImageFolderName, 'train2014'))
        if not os.path.exists(os.path.join(DestTrainValLabelFolderName, 'train2014')):
            os.makedirs(os.path.join(DestTrainValLabelFolderName, 'train2014'))
        for i_train in range(TrianImageNum): # [0, TrianImageNum]
            CurSrcImage = os.path.basename(AllImageName[AllImageIdx[i_train]].replace('.txt','.png'))
            CurImageName = TrainValTxtPrefixName + 'train2014/' + CurSrcImage
            fp.writelines(CurImageName + '\n')
            # copy file
            if SaveDestImageFlag == True:
                copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'train2014', CurSrcImage)) # Depth image
                copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '.txt'))) # Depth label
                copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '_Depth.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '_Depth.txt'))) # _Depth label

        fp.close()
        # AllImageName_2
        if len(AllImageName_2)>0:
            TrainFileName_2 = os.path.join(DestTrainValLabelFolderName, 'train2014_2.txt')
            fp_2 = open(TrainFileName_2, 'w')
            for i_train in range(TrianImageNum): # [0, TrianImageNum]
                CurSrcImage = os.path.basename(AllImageName_2[AllImageIdx[i_train]].replace('.txt','.png'))
                CurImageName = TrainValTxtPrefixName + 'train2014/' + CurSrcImage
                fp_2.writelines(CurImageName + '\n')
                # copy file
                if SaveDestImageFlag == True:
                    copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'train2014', CurSrcImage)) # RGB image
                    copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'train2014', CurSrcImage.replace('.png', '.txt'))) # RGB label
            fp_2.close()
            
        # val
        ValFileName = os.path.join(DestTrainValLabelFolderName, 'val2014.txt')           
        fp = open(ValFileName, 'w')
        if not os.path.exists(os.path.join(DestTrainValImageFolderName, 'val2014')):
            os.makedirs(os.path.join(DestTrainValImageFolderName, 'val2014'))
        if not os.path.exists(os.path.join(DestTrainValLabelFolderName, 'val2014')):
            os.makedirs(os.path.join(DestTrainValLabelFolderName, 'val2014'))
        for i_val in range(TrianImageNum, AllImageNum, 1): # [0, TrianImageNum]
            CurSrcImage = os.path.basename(AllImageName[AllImageIdx[i_val]].replace('.txt','.png'))
            CurImageName = TrainValTxtPrefixName + 'val2014/' + CurSrcImage
            fp.writelines(CurImageName + '\n')
            # copy file
            if SaveDestImageFlag == True:
                copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'val2014', CurSrcImage)) # Depth image
                copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '.txt'))) # Depth label
                copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '_Depth.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '_Depth.txt'))) # _Depth label

        fp.close()
        # AllImageName_2
        if len(AllImageName_2)>0:
            ValFileName_2 = os.path.join(DestTrainValLabelFolderName, 'val2014_2.txt')           
            fp_2 = open(ValFileName_2, 'w')
            for i_val in range(TrianImageNum, AllImageNum, 1): # [0, TrianImageNum]
                CurSrcImage = os.path.basename(AllImageName_2[AllImageIdx[i_val]].replace('.txt','.png'))
                CurImageName = TrainValTxtPrefixName + 'val2014/' + CurSrcImage
                fp_2.writelines(CurImageName + '\n')
                # copy file
                if SaveDestImageFlag == True:
                    copyfile(os.path.join(SrcImageFolderName, CurSrcImage), os.path.join(DestTrainValImageFolderName, 'val2014', CurSrcImage)) # RGB image
                    copyfile(os.path.join(SrcAnnoFolderName, CurSrcImage.replace('.png', '.txt')), os.path.join(DestTrainValLabelFolderName, 'val2014', CurSrcImage.replace('.png', '.txt'))) # RGB label
            fp_2.close()
            
        return 0


class AnnoLabelSelectValidCase():
    """
        功能：选择有效的标注文件信息
    """
    def __init__(self, AnnoXmlFolderName, AnnoTxtFolderName, ClassName):
        self.AnnoXmlFolderName = AnnoXmlFolderName
        self.AnnoTxtFolderName = AnnoTxtFolderName
        self.ClassName = ClassName
        self.SrcFilePostfixName = '.xml'
        self.DestFilePostfixName = '.txt'
    def DelectSameBboxFromAnnoTxt(self, SelectAnnoTxtFolderName):
        """
        从标注的txt 文件中去除重标注的目标框
        """
        AnnoTxtFolderName = self.AnnoTxtFolderName
        AnnoTxtFileGroup = glob.glob(os.path.join(AnnoTxtFolderName, '*.txt'))
        for OneAnnoTxtFile in AnnoTxtFileGroup:
            CurAnnoTxtFileName = os.path.basename(OneAnnoTxtFile)
            CurAnnoTxtFullFileName = OneAnnoTxtFile
            CurDestAnnoTxtFullFileName = os.path.join(SelectAnnoTxtFolderName, CurAnnoTxtFileName)
            # read txt info
            OneLabelInfoSrc = []
            fp_src = open(CurAnnoTxtFullFileName, 'r')
            OneLabelInfoLines = fp_src.readlines()
            for i_OneLabelInfo in OneLabelInfoLines:
                for i_data in i_OneLabelInfo.split():
                    OneLabelInfoSrc.append(float(i_data))
            OneLabelInfoSrc = np.array(OneLabelInfoSrc)
            OneLabelInfoSrc = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5]) # [label, x0, y0, w, h]
            fp_src.close()
            # compare bbox
            BboxRepeatFlag = False
            BboxNum = OneLabelInfoSrc.shape[0]
            for i_bbox in range(BboxNum):
                iOneBbox = OneLabelInfoSrc[i_bbox]
                for j_bbox in range(i_bbox+1, BboxNum):
                    jOneBbox = OneLabelInfoSrc[j_bbox]
                    # 2 bbox dist
                    ijBboxCompare = abs(iOneBbox[1] - jOneBbox[1]) + abs(iOneBbox[2] - jOneBbox[2]) + abs(iOneBbox[3] - jOneBbox[3]) + abs(iOneBbox[4] - jOneBbox[4])
                    if ijBboxCompare < 0.001:
                        BboxRepeatFlag = True
                        break
            # copy file
            if BboxRepeatFlag==True:
                print('  ', CurAnnoTxtFileName)
                copyfile(CurAnnoTxtFullFileName, CurDestAnnoTxtFullFileName)

        return 0
        
        


if __name__ == '__main__':
    print('Start.')
    
    TestCase = 2 # TestCase = 1，转换 label 类别 （3类 转 1类）
                 # TestCase = 2，显示 label 结果
                 # TestCase = 3，从标注的txt 文件中去除重标注的目标框
    
    # 转换 label 类别 （3类 转 1类）
    if TestCase == 1:
        TrainvalType = 'train2014'
        SrcLabelFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD\data\coco\labels'
        DestLabelFolderName = r'D:\xiongbiao\Code\LGPoseDete\YOLOv3_RGBD\data\coco\labels'
        LabelPostfixName = '.txt'
        SrcLabelFolderName = os.path.join(SrcLabelFolderName, TrainvalType)
        DestLabelFolderName = os.path.join(DestLabelFolderName, TrainvalType)
        
        LabelTransClass3To1(SrcLabelFolderName, DestLabelFolderName, LabelPostfixName=LabelPostfixName)
    
    # 显示 label 结果
    if TestCase == 2:
        # TrainvalType = 'val2014'
        # SrcLabelFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_737\coco\labels'
        # SrcImageFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_737\coco\images'
        # DestImageFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_737\coco\images_anno_plot'
        # LabelPostfixName = '.txt'
        
        # TrainvalType = 'train2014'
        # SrcLabelFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD_RGB\data\coco\labels'
        # SrcImageFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD_RGB\data\coco\images'
        # DestImageFolderName = r'D:\xiongbiao\Code\GPUServer\LGPoseDete\YOLOv3_RGBD_RGB\output\images_anno_plot'
        # LabelPostfixName = '.txt'
        
        
        # TrainvalType = 'val2014'
        # SrcLabelFolderName = r'D:\xiongbiao\Data\HumanDete\ZT\RGBD_20210720\DatasetAddHardImage\Dataset\data_class3\labels'
        # SrcImageFolderName = r'D:\xiongbiao\Data\HumanDete\ZT\RGBD_20210720\DatasetAddHardImage\Dataset\data_class3\images'
        # DestImageFolderName = r'D:\xiongbiao\Data\HumanDete\ZT\RGBD_20210720\DatasetAddHardImage\Dataset\images_anno_plot'
        # LabelPostfixName = '.txt'
        
        # RGBD_20211108
        TrainvalType = 'val2014'
        SrcLabelFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\labels'
        SrcImageFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\images'
        DestImageFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\images_anno_plot'
        LabelPostfixName = '.txt'
        
        SrcLabelFolderName = os.path.join(SrcLabelFolderName, TrainvalType)
        SrcImageFolderName = os.path.join(SrcImageFolderName, TrainvalType)
        DestImageFolderName = os.path.join(DestImageFolderName, TrainvalType)
        if not os.path.exists(DestImageFolderName):
            os.makedirs(DestImageFolderName)
            
        PlotImageAnnoInfo(SrcLabelFolderName, SrcImageFolderName, DestImageFolderName, LabelPostfixName=LabelPostfixName)
    
    # 从标注的txt 文件中去除重标注的目标框
    if TestCase == 3:
        ClassName = ['lying', 'sitting', 'standing']
        # train
        print('  train')
        SrcAnnoTxtFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\labels\train2014_src'
        ErrorAnnoTxtFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\labels\train2014_error'
        CurAnnoLabelSelectValidCase = AnnoLabelSelectValidCase('', SrcAnnoTxtFolderName, ClassName)
        CurAnnoLabelSelectValidCase.DelectSameBboxFromAnnoTxt(ErrorAnnoTxtFolderName)
        # val
        print('  val')
        SrcAnnoTxtFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\labels\val2014_src'
        ErrorAnnoTxtFolderName = r'D:\xiongbiao\Data\HumanDete\SZ2KSS\RGBD_20211108\dataset_test\coco\labels\val2014_error'
        CurAnnoLabelSelectValidCase = AnnoLabelSelectValidCase('', SrcAnnoTxtFolderName, ClassName)
        CurAnnoLabelSelectValidCase.DelectSameBboxFromAnnoTxt(ErrorAnnoTxtFolderName)
    