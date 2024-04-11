# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:18:05 2022

功能：从告警数据中生成RGBD数据集

@author: Administrator
"""
import os
from utils import DirMake, DirFileNum, CopyFolderFile, CopyFileFolderToFolder

from GenNoAnnoTestDataset import GenAllNoAnnoDataset
from LabelProcessFuns import AnnoLabelTransXmlToTxt, PlotImageAnnoInfo, GenTrainvalDataset_RGBD3Label
from ImageProcessFuns import CopySameFiles, TwoModeDataTransform
from FindHardCaseImage_20210603 import RGBDLabel_FindRGBLabel, GetRGBDImageAndLabel

class DatasetRGBDProcessFromAlarmData():
    def __init__(self, SrcDataInfo, DestDataInfo, Params):
        self.SrcDataInfo = SrcDataInfo
        self.DestDataInfo = DestDataInfo
        self.Params = Params
        
        # Params
        self.Params_ClassName = self.Params['ClassName']
        self.Params_DepthImageSize = self.Params['DepthImageSize']
        self.Params_RotParamFileName = self.Params['RotParamFileName']
        self.Params_BaseDatasetDir = self.Params['BaseDatasetDir']

        # Anno dir
        self.AnnoImageVott = os.path.join(self.DestDataInfo['DestGenDataDir'], 'AnnoImageVott')
        self.AnnoImageVott_Colormap = os.path.join(self.AnnoImageVott, 'Colormap')
        self.AnnoImageVott_ColormapXml = os.path.join(self.AnnoImageVott, 'Colormap_Xml')
        self.AnnoImageVott_ColormapTxt = os.path.join(self.AnnoImageVott, 'Colormap_Txt')
        self.AnnoImageVott_RGB = os.path.join(self.AnnoImageVott, 'RGB')
        self.AnnoImageVott_RGBXml = os.path.join(self.AnnoImageVott, 'RGB_Xml')
        self.AnnoImageVott_RGBTxt = os.path.join(self.AnnoImageVott, 'RGB_Txt')
        DirMake(self.AnnoImageVott_Colormap)
        DirMake(self.AnnoImageVott_ColormapXml)
        DirMake(self.AnnoImageVott_ColormapTxt)
        DirMake(self.AnnoImageVott_RGB)
        DirMake(self.AnnoImageVott_RGBXml)
        DirMake(self.AnnoImageVott_RGBTxt)
        
        # CombineRGBDLabel
        self.DatasetAddHardImage = os.path.join(self.DestDataInfo['DestGenDataDir'], 'DatasetAddHardImage')
        self.CombineRGBDLabel = os.path.join(self.DatasetAddHardImage, 'CombineRGBDLabel')
        self.CombineRGBDLabel_Colormap = os.path.join(self.CombineRGBDLabel, 'Colormap') # Colormap
        self.CombineRGBDLabel_ColormapLabel = os.path.join(self.CombineRGBDLabel, 'ColormapLabel')
        self.CombineRGBDLabel_ColormapAnnoPlot = os.path.join(self.CombineRGBDLabel, 'ColormapAnnoPlot')
        self.CombineRGBDLabel_RGB = os.path.join(self.CombineRGBDLabel, 'RGB') # RGB
        self.CombineRGBDLabel_RGBLabel = os.path.join(self.CombineRGBDLabel, 'RGBLabel')
        self.CombineRGBDLabel_RGBAnnoPlot = os.path.join(self.CombineRGBDLabel, 'RGBAnnoPlot')
        self.CombineRGBDLabel_RGB_MatchToDepth = os.path.join(self.CombineRGBDLabel, 'RGB_MatchToDepth') # RGB_MatchToDepth
        self.CombineRGBDLabel_RGBLabel_MatchToDepth = os.path.join(self.CombineRGBDLabel, 'RGBLabel_MatchToDepth')
        self.CombineRGBDLabel_RGBAnnoPlot_MatchToDepth = os.path.join(self.CombineRGBDLabel, 'RGBAnnoPlot_MatchToDepth')
        self.CombineRGBDLabel_RGBD = os.path.join(self.CombineRGBDLabel, 'RGBD') # RGBD
        self.CombineRGBDLabel_RGBDLabel = os.path.join(self.CombineRGBDLabel, 'RGBDLabel')
        self.CombineRGBDLabel_RGBD_Select = os.path.join(self.CombineRGBDLabel, 'RGBD_Select') # Select
        self.CombineRGBDLabel_RGBDLabel_Select = os.path.join(self.CombineRGBDLabel, 'RGBDLabel_Select') # 
        DirMake(self.CombineRGBDLabel_Colormap)
        DirMake(self.CombineRGBDLabel_ColormapLabel)
        DirMake(self.CombineRGBDLabel_ColormapAnnoPlot)
        DirMake(self.CombineRGBDLabel_RGB)
        DirMake(self.CombineRGBDLabel_RGBLabel)
        DirMake(self.CombineRGBDLabel_RGBAnnoPlot)
        DirMake(self.CombineRGBDLabel_RGB_MatchToDepth)
        DirMake(self.CombineRGBDLabel_RGBLabel_MatchToDepth)
        DirMake(self.CombineRGBDLabel_RGBAnnoPlot_MatchToDepth)
        DirMake(self.CombineRGBDLabel_RGBD)
        DirMake(self.CombineRGBDLabel_RGBDLabel)
        DirMake(self.CombineRGBDLabel_RGBD_Select)
        DirMake(self.CombineRGBDLabel_RGBDLabel_Select)
        
        # dataset
        self.RGBDDataset = os.path.join(self.DatasetAddHardImage, 'Dataset')
        self.RGBDDataset_AllImage = os.path.join(self.RGBDDataset, 'images_all')
        self.RGBDDataset_AllLabel = os.path.join(self.RGBDDataset, 'labels_all')
        self.RGBDDataset_3Class = os.path.join(self.RGBDDataset, 'data_class3')
        self.RGBDDataset_3Class_Image = os.path.join(self.RGBDDataset_3Class, 'images')
        self.RGBDDataset_3Class_ImageTrain = os.path.join(self.RGBDDataset_3Class, 'images', 'train2014')
        self.RGBDDataset_3Class_ImageVal = os.path.join(self.RGBDDataset_3Class, 'images', 'val2014')
        self.RGBDDataset_3Class_Label = os.path.join(self.RGBDDataset_3Class, 'labels')
        self.RGBDDataset_3Class_LabelTrain = os.path.join(self.RGBDDataset_3Class, 'labels', 'train2014')
        self.RGBDDataset_3Class_LabelVal = os.path.join(self.RGBDDataset_3Class, 'labels', 'val2014') 
        DirMake(self.RGBDDataset_AllImage)
        DirMake(self.RGBDDataset_AllLabel)
        DirMake(self.RGBDDataset_3Class_ImageTrain)
        DirMake(self.RGBDDataset_3Class_ImageVal)
        DirMake(self.RGBDDataset_3Class_LabelTrain)
        DirMake(self.RGBDDataset_3Class_LabelVal)
    
    # GenRGBDImageFromAlarmData
    def GenRGBDImageFromAlarmData(self):
        CurGenTestErrorDataset = GenAllNoAnnoDataset(self.SrcDataInfo, self.DestDataInfo, self.Params)
        CurGenTestErrorDataset.GenTestDataset()
        return 0
        
    # GenRGBDTxtLabelFromAnno
    def GenRGBDTxtLabelFromAnno(self):
        ClassName = self.Params_ClassName
        SrcXmlLabelColormapFolderName = self.AnnoImageVott_ColormapXml
        DestXmlLabelColormapFolderName = self.AnnoImageVott_ColormapTxt
        CurAnnoLabelTransXmlToTxt = AnnoLabelTransXmlToTxt(SrcXmlLabelColormapFolderName, DestXmlLabelColormapFolderName, ClassName)
        CurAnnoLabelTransXmlToTxt.MultiAnnoLabelTransXmlToTxt()
        SrcXmlLabelRGBFolderName = self.AnnoImageVott_RGBXml
        DestXmlLabelRGBFolderName = self.AnnoImageVott_RGBTxt
        CurAnnoLabelTransXmlToTxt2 = AnnoLabelTransXmlToTxt(SrcXmlLabelRGBFolderName, DestXmlLabelRGBFolderName, ClassName)
        CurAnnoLabelTransXmlToTxt2.MultiAnnoLabelTransXmlToTxt()
        return 0
        
    # GenRGBImageLabelFromDepthInfo
    def GenRGBImageLabelFromDepthInfo(self):
        # 复制文件
        CopySameFiles(self.AnnoImageVott_Colormap, self.CombineRGBDLabel_Colormap, self.AnnoImageVott_Colormap)
        CopySameFiles(self.AnnoImageVott_ColormapTxt, self.CombineRGBDLabel_ColormapLabel, self.AnnoImageVott_ColormapTxt)
        CopySameFiles(self.AnnoImageVott_RGB, self.CombineRGBDLabel_RGB, self.AnnoImageVott_RGB)
        CopySameFiles(self.AnnoImageVott_RGBTxt, self.CombineRGBDLabel_RGBLabel, self.AnnoImageVott_RGBTxt)
        # 将RGB转换到Depth大小，包括：图像转换、label转换
        DestImageSize = self.Params_DepthImageSize
        RotParamFileName = self.Params_RotParamFileName
        CurTwoModeDataTransform = TwoModeDataTransform(self.CombineRGBDLabel_RGB, self.CombineRGBDLabel_RGBLabel, self.CombineRGBDLabel_RGB_MatchToDepth, self.CombineRGBDLabel_RGBLabel_MatchToDepth, DestImageSize, RotParamFileName)
        CurTwoModeDataTransform.MultiAnnoFrameTransform()
        # PlotImageAnnoInfo
        PlotImageAnnoInfo(self.CombineRGBDLabel_ColormapLabel, self.CombineRGBDLabel_Colormap, self.CombineRGBDLabel_ColormapAnnoPlot, LabelPostfixName='.txt')
        PlotImageAnnoInfo(self.CombineRGBDLabel_RGBLabel, self.CombineRGBDLabel_RGB, self.CombineRGBDLabel_RGBAnnoPlot, LabelPostfixName='.txt')
        PlotImageAnnoInfo(self.CombineRGBDLabel_RGBLabel_MatchToDepth, self.CombineRGBDLabel_RGB_MatchToDepth, self.CombineRGBDLabel_RGBAnnoPlot_MatchToDepth, LabelPostfixName='.txt')
        return 0
    
    # GenRGBDLabelFromTwoTypeData
    def GenRGBDLabelFromTwoTypeData(self):
        LabelFilePostfixName = '.txt'
        # RGBD Label select
        RGBDLabel_FindRGBLabel(self.CombineRGBDLabel_RGBD_Select, self.CombineRGBDLabel_RGBD_Select, self.CombineRGBDLabel_RGBLabel_MatchToDepth)
        # RGBD Label
        CopySameFiles(self.CombineRGBDLabel_ColormapLabel, self.CombineRGBDLabel_RGBDLabel_Select, self.CombineRGBDLabel_ColormapLabel)
        CopySameFiles(self.CombineRGBDLabel_RGBD_Select, self.CombineRGBDLabel_RGBDLabel_Select, self.CombineRGBDLabel_RGBD_Select, FilePostfixName=LabelFilePostfixName)
        # 生成有效的RGBD数据，包括：RGB/Depth 图像（2类图像）和 RGB/Depth/RGBD 的 label（3类label）
        GetRGBDImageAndLabel(self.CombineRGBDLabel_Colormap, self.CombineRGBDLabel_RGB_MatchToDepth, \
                         self.CombineRGBDLabel_ColormapLabel, self.CombineRGBDLabel_RGBLabel_MatchToDepth, self.CombineRGBDLabel_RGBDLabel_Select, \
                         self.CombineRGBDLabel_RGBD, self.CombineRGBDLabel_RGBDLabel)
        return 0
    
    # GenRGBDTrainvalDataset
    def GenRGBDTrainvalDataset(self):
        # 复制文件
        CopyFileFolderToFolder(self.CombineRGBDLabel_RGBD, self.RGBDDataset_AllImage)
        CopyFileFolderToFolder(self.CombineRGBDLabel_RGBDLabel, self.RGBDDataset_AllLabel)
        # GenTrainvalDataset_RGBD3Label
        CurGenTrainvalDataset_RGBD3Label = GenTrainvalDataset_RGBD3Label(self.RGBDDataset_AllImage, self.RGBDDataset_AllLabel, \
                                                                      self.RGBDDataset_3Class_Image, self.RGBDDataset_3Class_Label)
        
        # 重新生成数据集 train/val
        # CurGenTrainvalDataset_RGBD3Label.GenRGBDTrainvalDataset() # 重新生成数据集 train/val
        
        # 再以前数据集的基础上生成数据集 train/val
        BeforeDatasetInfo = dict()
        BeforeDatasetInfo['TrainFileName'] = os.path.join(self.Params_BaseDatasetDir, 'train2014.txt')
        BeforeDatasetInfo['ValFileName'] = os.path.join(self.Params_BaseDatasetDir, 'val2014.txt')
        CurGenTrainvalDataset_RGBD3Label.GenRGBDTrainvalDatasetBaseOnBeforeDataset(BeforeDatasetInfo) # 在原数据集基础上，添加生成数据集 train/val

        return 0

if __name__ == '__main__':
    print('start.')
    
    TestCase = 1 # TestCase=1,EK数据集生成
    
    if TestCase == 1:
        """
        生成数据步骤
        1，从告警数据中选择日期，生成图片 
        2，测试生成的数据 （不在此目录）
        3，手动挑选检测不好的数据，准备标注 （手动挑选）
        4，标注选择好的RGB和Depth数据 （手动标注）
        5，转换标注好的label
        6, 转换RGB数据，并显示两类数据，便于查看标注和转换是否正确
        7, 生成RGBD数据和label
        8, 生成数据集trainval（可以将此次标注的图片和之前的图片合并，进行测试和训练）
        """

        # 设置参数
        # 输入数据信息
        SrcDataInfo = dict()    
        SrcDataInfo['RGB'] = [r'E:\project\RGBD\Data\AlarmData\EK']
        SrcDataInfo['Depth'] = [r'E:\project\RGBD\Data\AlarmData\EK']
        SrcDataInfo['Colormap'] = [r'E:\project\RGBD\Data\AlarmData\EK']
        SrcDataInfo['RotParamFileName'] = dict()
        SrcDataInfo['RotParamFileName']['RGB2DepthParamsFolderNameDefault'] = [r'rgb2depth_param.txt',]
        SrcDataInfo['RotParamFileName']['RGB2DepthParamsFolderName'] = [r'',]
        # 输出数据信息E:\project\RGBD\JCDete\DataProcess\RGB2DepthRotParams\SZ2KSS
        DestDataInfo = dict()   
        DestDataInfo['RGB'] = [r'E:\project\RGBD\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\SelectHardCaseImage\RGB']
        DestDataInfo['Depth'] = [r'E:\project\RGBD\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\SelectHardCaseImage\depth']
        DestDataInfo['Colormap'] = [r'E:\project\RGBD\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\SelectHardCaseImage\Colormap']
        DestDataInfo['Dataset'] = [r'E:\project\RGBD\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\dataset_test']
        #    DestGenDataDir
        DestDataInfo['DestGenDataDir'] = r'E:\project\RGBD\JCDete\Data\Dataset_RGBD\EK_RGBD_202201'
        
        # 设置参数信息
        Params = dict()         
        Params['SelectRoom'] = ['1001', '1002'] # 选择房间编号
        Params['SelectDate'] = [                # 选择日期
                                '20211021',
                                '20211022',
                                '20211023',
                                '20211024',
                                '20211025',
                                '20211026',
                                '20211027',
                                '20211028',
                                '20211029',
                                '20211030',
                                '20211031',
                                '20211101',
                                '20211102',
                                '20211103',
                                '20211104',
                                '20211105',
                                '20211106',
                                '20211107',
                                '20211108',
                                '20211109',
                                '20211110',
                                # '20211111',
                                # '20211112',
                                # '20211113',
                                # '20211114', 
                                # '20211115',
                             ]
        Params['ClassName'] = ['lying', 'sitting', 'standing'] # ['lie', 'sit', 'stand'] / ['lying', 'sitting', 'standing']
        Params['DepthImageSize'] = [512, 424] # DepthImageSize
        Params['RotParamFileName'] = [r'E:\project\RGBD\JCDete\DataProcess\RGB2DepthRotParams\SZ2KSS',
                                        r'E:\project\RGBD\JCDete\DataProcess\RGB2DepthRotParams\ZT',]    #RGB与Depth对应关系
        Params['BaseDatasetDir'] = r'E:\project\RGBD\JCDete\RGBD_Dete\Dataset\RGBD\data_class3_lg'    #原始数据集
        
        # 选择步骤
        Step1_Flag = 0
        Step5_Flag = 0
        Step6_Flag = 0
        Step7_Flag = 0
        Step8_Flag = 1
        
        # 实例化
        CurDatasetRGBDProcessFromAlarmData = DatasetRGBDProcessFromAlarmData(SrcDataInfo, DestDataInfo, Params)
        
        # 1，从告警数据中选择日期，生成图片, [20210119, 针对EK数据，设置外仓洗漱时间数据，暂不获取]
        if Step1_Flag == 1:
            CurDatasetRGBDProcessFromAlarmData.GenRGBDImageFromAlarmData()
            
            
            
        # 5，转换标注好的label
        if Step5_Flag == 1:
            CurDatasetRGBDProcessFromAlarmData.GenRGBDTxtLabelFromAnno()
        
        # 6, 合并两类数据
        if Step6_Flag == 1:
            CurDatasetRGBDProcessFromAlarmData.GenRGBImageLabelFromDepthInfo()
    
        # 7, 生成RGBD数据和label
        if Step7_Flag == 1:
            CurDatasetRGBDProcessFromAlarmData.GenRGBDLabelFromTwoTypeData()

        # 8, 生成数据集trainval
        if Step8_Flag == 1:
            CurDatasetRGBDProcessFromAlarmData.GenRGBDTrainvalDataset()
    
    
    print('end.')

