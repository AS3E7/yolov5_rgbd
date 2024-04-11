# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:04:53 2021

@author: Administrator
"""



import os
import cv2
import numpy as np
import random
import shutil
import xml.etree.ElementTree as ET
from shutil import copyfile


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

#生成文件名索引，利用标注好的xml文件，建立coco数据集

    xmlfilepath = r"C:\Users\Administrator\Desktop\images\train-PascalVOC-export\Annotations"
    saveBasePath = r"D:\data"

    srcimagefilepath = r'C:\Users\Administrator\Desktop\images\train-PascalVOC-export\JPEGImages'
    destimagefilepath = r'D:\data\images'

    srclabelfilepath = r'D:\data\anno_txt'
    destlabelfilepath = r'D:\data\labels'

    desttrainimagefilepath = destimagefilepath + '/train2014/'
    destvalimagefilepath = destimagefilepath + '/val2014/'

    desttrainlabelfilepath = destlabelfilepath + '/train2014/'
    destvallabelfilepath = destlabelfilepath + '/val2014/'

    desttrainimagefilepath = destimagefilepath + '/train2014/'
    destvalimagefilepath = destimagefilepath + '/val2014/'

    if not os.path.exists(desttrainimagefilepath):
        os.makedirs(desttrainimagefilepath)
    else:
        shutil.rmtree(desttrainimagefilepath)
        os.makedirs(desttrainimagefilepath)

    if not os.path.exists(destvalimagefilepath):
        os.makedirs(destvalimagefilepath)
    else:
        shutil.rmtree(destvalimagefilepath)
        os.makedirs(destvalimagefilepath)

    desttrainlabelfilepath = destlabelfilepath + '/train2014/'
    destvallabelfilepath = destlabelfilepath + '/val2014/'

    if not os.path.exists(desttrainlabelfilepath):
        os.makedirs(desttrainlabelfilepath)
    else:
        shutil.rmtree(desttrainlabelfilepath)
        os.makedirs(desttrainlabelfilepath)

    if not os.path.exists(destvallabelfilepath):
        os.makedirs(destvallabelfilepath)
    else:
        shutil.rmtree(destvallabelfilepath)
        os.makedirs(destvallabelfilepath)

    trainval_percent = 1  # 训练+验证集的比例
    train_percent = 0.9  # 训练集的比例

    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '.png' + '\n'
        srcimagename = total_xml[i][:-4] + '.png'
        srcimagefullname = srcimagefilepath + '/' + srcimagename

        srclabelname = total_xml[i][:-4] + '.txt'
        srclabelfullname = srclabelfilepath + '/' + srclabelname

        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write('./coco/images/train2014/' + name)
                destimagefullname = desttrainimagefilepath + srcimagename
                copyfile(srcimagefullname, destimagefullname)

                destlabelfullname = desttrainlabelfilepath + srclabelname
                copyfile(srclabelfullname, destlabelfullname)

            else:
                fval.write('./coco/images/val2014/' + name)
                destimagefullname = destvalimagefilepath + srcimagename
                copyfile(srcimagefullname, destimagefullname)

                destlablefullname = destvallabelfilepath + srclabelname
                copyfile(srclabelfullname, destlablefullname)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    print('Start.')



    SrcAnnoXmlFolderName = r'C:\Users\Administrator\Desktop\images\train-PascalVOC-export\Annotations'
    DestAnnoTxtFolderName = r'D:\data\anno_txt'
    ClassName = ['lie','sit','stand']


    CurAnnoLabelTransXmlToTxt = AnnoLabelTransXmlToTxt(SrcAnnoXmlFolderName, DestAnnoTxtFolderName, ClassName)
    CurAnnoLabelTransXmlToTxt.MultiAnnoLabelTransXmlToTxt()

    print('End.')