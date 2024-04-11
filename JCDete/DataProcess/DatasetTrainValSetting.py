# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:40:22 2021

1，对dataset数据集处理，每次添加数据，在之前数据集的基础上添加，不再重新随机生成trainval数据集

@author: Administrator
"""

import os
import numpy as np
import random
from shutil import copyfile

class ReSetDatasetTrainvalInfo():
    def __init__(self, SrcBaseDatasetFolderName, SrcDatasetFolderName, DestDatasetFolderName, AddImageFolderName):
        self.SrcBaseDatasetFolderName = SrcBaseDatasetFolderName
        self.SrcDatasetFolderName = SrcDatasetFolderName
        self.DestDatasetFolderName = DestDatasetFolderName
        self.AddImageFolderName = AddImageFolderName

    def ReSetDatasetTrainvalProcess(self):
        """
        功能：重新设置trainval处理过程
        """
        
        # 获取新标注数据名称
        FilePostfixName = ['.png']
        ExcludePostfixName = ['_Depth.png', '_RGB.png']
        AddImageFileNameGroup = self.GenFoderFileNameGroup(self.AddImageFolderName, FilePostfixName=FilePostfixName, ExcludePostfixName=ExcludePostfixName)
        # 获取前数据集名称
        PreDatasetTrainval = dict()
        CurDatasetTrainValTxtFileGroup, CurDatasetTrainTxtFileGroup, CurDatasetValTxtFileGroup = self.GenDatasetTrainvalFileName(self.SrcDatasetFolderName)
        PreDatasetTrainval['trainval'] = CurDatasetTrainValTxtFileGroup
        PreDatasetTrainval['train'] = CurDatasetTrainTxtFileGroup
        PreDatasetTrainval['val'] = CurDatasetValTxtFileGroup
        # 获取新使用的数据名称，少于新标注的数据
        AddDatasetTrainval, OnlyPreDatasetTrainval = self.GenAddDatasetTraivalFileName(PreDatasetTrainval, AddImageFileNameGroup, TrainRatio=0.8)
        # 获取原始基础数据集信息
        BaseDatasetTrainval = dict()
        BaseDatasetTrainValTxtFileGroup, BaseDatasetTrainTxtFileGroup, BaseDatasetValTxtFileGroup = self.GenDatasetTrainvalFileName(self.SrcBaseDatasetFolderName)
        BaseDatasetTrainval['trainval'] = BaseDatasetTrainValTxtFileGroup
        BaseDatasetTrainval['train'] = BaseDatasetTrainTxtFileGroup
        BaseDatasetTrainval['val'] = BaseDatasetValTxtFileGroup
        # 获取新数据集文件
        self.GenNewDatasetTraivalFileName(BaseDatasetTrainval, AddDatasetTrainval, DestDatasetFolderName)
        # 获取新数据集图片数据
        SrcFileFolderNameGroup = []
        SrcFileFolderNameGroup.append(os.path.join(self.SrcDatasetFolderName, 'coco', 'images', 'train2014'))
        SrcFileFolderNameGroup.append(os.path.join(self.SrcDatasetFolderName, 'coco', 'images', 'val2014'))
        SrcFileFolderNameGroup.append(os.path.join(self.SrcDatasetFolderName, 'coco', 'labels', 'train2014'))
        SrcFileFolderNameGroup.append(os.path.join(self.SrcDatasetFolderName, 'coco', 'labels', 'val2014'))
        self.GenNewDatasetTraivalFile(BaseDatasetTrainval, AddDatasetTrainval, DestDatasetFolderName, SrcFileFolderNameGroup)
        
        return 0
    
    def GenNewDatasetTraivalFile(self, PreDatasetTrainval, AddDatasetTrainval, DestDatasetFolderName, SrcFileFolderNameGroup):
        """
        功能：获取新数据图像/label文件
        """
        ImagePosefixName = '.png'
        ImageRGBPosefixName = '_RGB.png'
        LabelPosefixName = '.txt'
        LabelRGBPosefixName = '_RGB.txt'
        LabelDepthPosefixName = '_Depth.txt'

        DestDatasetTrainImageFolderName = os.path.join(DestDatasetFolderName, 'coco', 'images', 'train2014')
        DestDatasetTrainLabelFolderName = os.path.join(DestDatasetFolderName, 'coco', 'labels', 'train2014')
        DestDatasetValImageFolderName = os.path.join(DestDatasetFolderName, 'coco', 'images', 'val2014')
        DestDatasetValLabelFolderName = os.path.join(DestDatasetFolderName, 'coco', 'labels', 'val2014')
        if not os.path.exists(DestDatasetTrainImageFolderName):
            os.makedirs(DestDatasetTrainImageFolderName)
        if not os.path.exists(DestDatasetTrainLabelFolderName):
            os.makedirs(DestDatasetTrainLabelFolderName)
        if not os.path.exists(DestDatasetValImageFolderName):
            os.makedirs(DestDatasetValImageFolderName)
        if not os.path.exists(DestDatasetValLabelFolderName):
            os.makedirs(DestDatasetValLabelFolderName)
        # copyfile
        NewDatasetTrain = PreDatasetTrainval['train'] + AddDatasetTrainval['train']
        NewDatasetVal = PreDatasetTrainval['val'] + AddDatasetTrainval['val']
        # train
        for i_train in NewDatasetTrain: # [0, train]
            SelectFileName = i_train
            SrcImageFileName = SelectFileName
            SrcImageRGBFileName = SelectFileName.replace(ImagePosefixName, ImageRGBPosefixName)
            SrcLabelFileName = SelectFileName.replace(ImagePosefixName, LabelPosefixName)
            SrcLabelRGBFileName = SelectFileName.replace(ImagePosefixName, LabelRGBPosefixName)
            SrcLabelDepthFileName = SelectFileName.replace(ImagePosefixName, LabelDepthPosefixName)
            for i_SrcFileFolderNameGroup in SrcFileFolderNameGroup:
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcImageFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcImageFileName), os.path.join(DestDatasetTrainImageFolderName, SrcImageFileName))
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcImageRGBFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcImageRGBFileName), os.path.join(DestDatasetTrainImageFolderName, SrcImageRGBFileName))
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelFileName), os.path.join(DestDatasetTrainLabelFolderName, SrcLabelFileName))       
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelRGBFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelRGBFileName), os.path.join(DestDatasetTrainLabelFolderName, SrcLabelRGBFileName))         
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelDepthFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelDepthFileName), os.path.join(DestDatasetTrainLabelFolderName, SrcLabelDepthFileName))         
        # val
        for i_val in NewDatasetVal: # [0, train]
            SelectFileName = i_val
            SrcImageFileName = SelectFileName
            SrcImageRGBFileName = SelectFileName.replace(ImagePosefixName, ImageRGBPosefixName)
            SrcLabelFileName = SelectFileName.replace(ImagePosefixName, LabelPosefixName)
            SrcLabelRGBFileName = SelectFileName.replace(ImagePosefixName, LabelRGBPosefixName)
            SrcLabelDepthFileName = SelectFileName.replace(ImagePosefixName, LabelDepthPosefixName)
            for i_SrcFileFolderNameGroup in SrcFileFolderNameGroup:
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcImageFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcImageFileName), os.path.join(DestDatasetValImageFolderName, SrcImageFileName))
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcImageRGBFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcImageRGBFileName), os.path.join(DestDatasetValImageFolderName, SrcImageRGBFileName))
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelFileName), os.path.join(DestDatasetValLabelFolderName, SrcLabelFileName))       
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelRGBFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelRGBFileName), os.path.join(DestDatasetValLabelFolderName, SrcLabelRGBFileName))         
                if os.path.exists(os.path.join(i_SrcFileFolderNameGroup, SrcLabelDepthFileName)):
                    copyfile(os.path.join(i_SrcFileFolderNameGroup, SrcLabelDepthFileName), os.path.join(DestDatasetValLabelFolderName, SrcLabelDepthFileName))         

        return 0
    
    def GenNewDatasetTraivalFileName(self, PreDatasetTrainval, AddDatasetTrainval, DestDatasetFolderName):
        """
        功能：根据前数据集和新加数据集，生成新的数据集
        """
        DestDatasetTrainvalTxtFileName = os.path.join(DestDatasetFolderName, 'trainval2014.txt')
        DestDatasetTrainTxtFileName = os.path.join(DestDatasetFolderName, 'train2014.txt')
        DestDatasetValTxtFileName = os.path.join(DestDatasetFolderName, 'val2014.txt')
        TrainValTxtPrefixName = './coco/images/'
        NewDatasetTrainval = PreDatasetTrainval['trainval'] + AddDatasetTrainval['trainval']
        NewDatasetTrain = PreDatasetTrainval['train'] + AddDatasetTrainval['train']
        NewDatasetVal = PreDatasetTrainval['val'] + AddDatasetTrainval['val']
        # trainval
        fp = open(DestDatasetTrainvalTxtFileName, 'w')
        for i_trainval in NewDatasetTrainval: # [0, trainval]
            CurImageName = TrainValTxtPrefixName + 'trainval2014/' + i_trainval
            fp.writelines(CurImageName + '\n')
        fp.close()
        # train
        fp = open(DestDatasetTrainTxtFileName, 'w')
        for i_train in NewDatasetTrain: # [0, train]
            CurImageName = TrainValTxtPrefixName + 'train2014/' + i_train
            fp.writelines(CurImageName + '\n')
        fp.close()
        # val
        fp = open(DestDatasetValTxtFileName, 'w')
        for i_val in NewDatasetVal: # [0, val]
            CurImageName = TrainValTxtPrefixName + 'val2014/' + i_val
            fp.writelines(CurImageName + '\n')
        fp.close()
        
        return 0
    
    def GenAddDatasetTraivalFileName(self, PreDatasetTrainval, AddImageFileNameGroup, TrainRatio=0.8):
        """
        功能：获取添加新数据后，dataset trainval 信息
        """
        AddDatasetTrainval = dict()
        OnlyPreDatasetTrainval = dict()
        # 获取新的有效图像数据
        AddImageFileNameVaild = []
        OnlyPreImageFileNameVaild = []
        PreDatasetTrainvalAll = PreDatasetTrainval['trainval']
        for i_filename in PreDatasetTrainvalAll:
            if i_filename in AddImageFileNameGroup:
                AddImageFileNameVaild.append(i_filename)
            else:
                OnlyPreImageFileNameVaild.append(i_filename)
        AddDatasetTrainval['trainval'] = AddImageFileNameVaild
        OnlyPreDatasetTrainval['trainval'] = OnlyPreImageFileNameVaild
        # 对新获取的数据生成trainval
        AddImageFileNameVaildNum = len(AddImageFileNameVaild)
        TrianImageNum = int(AddImageFileNameVaildNum * TrainRatio)
        AddImageFileNameVaildRandIdx = [x for x in range(AddImageFileNameVaildNum)]
        random.shuffle(AddImageFileNameVaildRandIdx)
        # add dataset train
        AddDatasetTrainval['train'] = AddImageFileNameVaild[:TrianImageNum]
        # add dataset val
        AddDatasetTrainval['val'] = AddImageFileNameVaild[TrianImageNum:]
        # OnlyPreDatasetTrainval, train/val
        OnlyPreImageFileNameVaildTrain = []
        for i_filename in PreDatasetTrainval['train']:
            if not i_filename in AddImageFileNameGroup:
                OnlyPreImageFileNameVaildTrain.append(i_filename)
        OnlyPreDatasetTrainval['train'] = OnlyPreImageFileNameVaildTrain
        OnlyPreImageFileNameVaildVal = []
        for i_filename in PreDatasetTrainval['val']:
            if not i_filename in AddImageFileNameGroup:
                OnlyPreImageFileNameVaildVal.append(i_filename)
        OnlyPreDatasetTrainval['val'] = OnlyPreImageFileNameVaildVal
        return AddDatasetTrainval, OnlyPreDatasetTrainval
    
    def GenDatasetTrainvalFileName(self, DatasetFolderName):
        """
        功能：获取当前文件夹trainval信息
        """
        CurDatasetTrainTxtFileName = os.path.join(DatasetFolderName, 'train2014.txt')
        CurDatasetValTxtFileName = os.path.join(DatasetFolderName, 'val2014.txt')
        # read txt info
        CurDatasetTrainTxtFileGroup = self.ReadTrainvalTxt(CurDatasetTrainTxtFileName)
        CurDatasetValTxtFileGroup = self.ReadTrainvalTxt(CurDatasetValTxtFileName)
        CurDatasetTrainValTxtFileGroup = CurDatasetTrainTxtFileGroup + CurDatasetValTxtFileGroup
        
        return CurDatasetTrainValTxtFileGroup, CurDatasetTrainTxtFileGroup, CurDatasetValTxtFileGroup
    
    @classmethod        
    def ReadTrainvalTxt(self, txt_dir):
        """
        功能：读取tainval txt 文件信息
        """
        fp = open(txt_dir, 'r') # read txt file
        files_lines = fp.readlines()
        fp.close()
        files = []
        for i_file in files_lines:            
            cur_i_file = i_file.split('/')[-1].strip() # trans current file name
            files.append(cur_i_file)   
        return files
    
    @classmethod 
    def WriteTrainvalTxt(self, file_list, txt_dir, TrainValName='train', PrefixName = './coco/images/'):
        """
        功能：写tainval txt 文件信息
        """
        fp = open(txt_dir, 'w')
        for i_val in file_list: # [0, val]
            CurImageName = PrefixName + TrainValName + '2014/' + i_val
            fp.writelines(CurImageName + '\n')
        fp.close()
    
    @classmethod  
    def GenFoderFileNameGroup(self, FolderName, FilePostfixName=None, ExcludePostfixName=None):
        """
        功能：获取文件夹下文件名
        """
        FileNameGroup = []
        for root, dirs, files in os.walk(FolderName):
            for filename in files:
                # find valid filename
                FilePostfixNameValid = False
                ExcludePostfixNameValid = False
                for i_FilePostfixName in FilePostfixName:
                    if filename.find(i_FilePostfixName)>-1:
                        FilePostfixNameValid = True
                for i_ExcludePostfixName in ExcludePostfixName:
                    if filename.find(i_ExcludePostfixName)>-1:
                        ExcludePostfixNameValid = True
                if FilePostfixNameValid==True and ExcludePostfixNameValid==False:
                    FileNameGroup.append(filename) 
        return FileNameGroup
    

class GenKFoldDatasetTrainvalInfo():
    """
    根据trainval设置其他trainval数据集，验证K-fold数据集效果
    """
    def __init__(self, SrcDatasetFolderName, DestDatasetFolderName):
        self.SrcDatasetFolderName = SrcDatasetFolderName
        self.DestDatasetFolderName = DestDatasetFolderName
        
        self.SrcDatasetTrainvalTxt = os.path.join(SrcDatasetFolderName, 'trainval2014.txt') 
        self.SrcDatasetTrainTxt = os.path.join(SrcDatasetFolderName, 'train2014.txt') 
        self.SrcDatasetValTxt = os.path.join(SrcDatasetFolderName, 'val2014.txt') 
    
    def GenKFoldTrainvalFile(self):
        """
        功能：生成K-fold数据集
        """
        DestTrainval = dict()
        DestTrainval['train'] = dict()
        DestTrainval['val'] = dict()
        # read txt file
        SrcDatasetTrainvalFileNameGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(self.SrcDatasetTrainvalTxt)
        SrcDatasetTrainFileNameGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(self.SrcDatasetTrainTxt)
        SrcDatasetValFileNameGroup = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(self.SrcDatasetValTxt)
        DestTrainval['train'][0] = SrcDatasetTrainFileNameGroup
        DestTrainval['val'][0] = SrcDatasetValFileNameGroup
        # combine
        SrcTrainPartNum = 4
        SrcTrainEachPartFileNum = int(len(SrcDatasetTrainFileNameGroup)/SrcTrainPartNum)
        SrcTrainImageFileNameNum = len(SrcDatasetTrainFileNameGroup)
        SrcTrainImageFileNameIdx = [x for x in range(SrcTrainImageFileNameNum)]
        random.shuffle(SrcTrainImageFileNameIdx)
        for i in range(SrcTrainPartNum):
            CurSelectValFileName = []
            for i_select, i_filename in enumerate(SrcTrainImageFileNameIdx):
                if (i_select >= i*SrcTrainEachPartFileNum) and (i_select < (i+1)*SrcTrainEachPartFileNum):
                    CurSelectValFileName.append(SrcDatasetTrainFileNameGroup[SrcTrainImageFileNameIdx[i_filename]])
            CurSelectTrainFileName = list(set(SrcDatasetTrainvalFileNameGroup).difference(set(CurSelectValFileName)))
            DestTrainval['train'][i+1] = CurSelectTrainFileName
            DestTrainval['val'][i+1] = CurSelectValFileName
        # write txt file
        for i in range(len(DestTrainval['train'])):
            CurSaveTrainTxtFileName = os.path.join(self.DestDatasetFolderName, 'train2014_' + str(i) + '.txt')
            CurTrainTxtFileGroup = DestTrainval['train'][i]
            ReSetDatasetTrainvalInfo.WriteTrainvalTxt(CurTrainTxtFileGroup, CurSaveTrainTxtFileName, TrainValName='train')
            CurSaveTrainTxtFileName = os.path.join(self.DestDatasetFolderName, 'val2014_' + str(i) + '.txt')
            CurTrainTxtFileGroup = DestTrainval['val'][i]
            ReSetDatasetTrainvalInfo.WriteTrainvalTxt(CurTrainTxtFileGroup, CurSaveTrainTxtFileName, TrainValName='val')
        return 0

if __name__ == '__main__':
    print('start.')
    
    TestCase = 2 # 
                 # TestCase = 2, 根据trainval设置其他trainval数据集，验证K-fold数据集效果
    
    if TestCase == 1:
        SrcBaseDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_774'
        SrcDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_958_re-random'
        DestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_958'
        
        # SrcBaseDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_774'
        # SrcDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_958_re-random'
        # DestDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_958'
        
        AddImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\JC_1000\RGBDImage'
        
        SelectReSetDatasetTrainvalInfo = ReSetDatasetTrainvalInfo(SrcBaseDatasetFolderName, SrcDatasetFolderName, DestDatasetFolderName, AddImageFolderName)
        SelectReSetDatasetTrainvalInfo.ReSetDatasetTrainvalProcess()
    
    if TestCase == 2: # 根据trainval设置其他trainval数据集，验证K-fold数据集效果
        SrcTrainvalDatasetFolderName = r'D:\xiongbiao\Data\HumanDete\Dataset\data_class3_863_exclude_bbox'
        DestTrainvalDatasetFolderName = r'output\SetTrainval\data_class3_863_exclude_bbox'
    
        CurGenKFoldDatasetTrainvalInfo = GenKFoldDatasetTrainvalInfo(SrcTrainvalDatasetFolderName, DestTrainvalDatasetFolderName)
        CurGenKFoldDatasetTrainvalInfo.GenKFoldTrainvalFile()
        
        