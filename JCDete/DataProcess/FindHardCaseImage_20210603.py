# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:48:38 2021

@author: Administrator
"""

import numpy as np
import os
import cv2
from shutil import copyfile
import glob

from ImageProcessFuns import CopySemiSameFiles, ResizeImageAnnoInfo, ResizeImages, TransFileName, TwoModeDataTransform
from ColormapProcessFuns import depthFileToColormapFile
from LabelProcessFuns import PlotImageAnnoInfo, AnnoLabelTransXmlToTxt, GenTrainvalDataset_RGBD3Label, LabelTransClass3To1
from DatasetTrainValSetting import ReSetDatasetTrainvalInfo

def FindRGBDImageFromHardImage(SelectDataDateName=None):
    """
    功能：添加Hard 数据
    """
    
    SelectAnnoRGBFolderName = r"D:\xiongbiao\Data\HumanDete\NewLG\\" + SelectDataDateName + '\SelectHardCaseImage\RGB_Select\AddRoomImage\RGB_Src'
    SelectAnnoDepthFolderName = r"D:\xiongbiao\Data\HumanDete\NewLG\\" + SelectDataDateName + '\SelectHardCaseImage\RGB_Select\AddRoomImage\depth_Src'
    SrcAnnoRGBFolderName = r"D:\xiongbiao\Data\HumanDete\NewLG\\" + SelectDataDateName + '\SelectHardCaseImage\RGB_Select\AddRoomImage\RGB'
    SrcAnnoDepthFolderName = r"D:\xiongbiao\Data\HumanDete\NewLG\\" + SelectDataDateName + '\SelectHardCaseImage\RGB_Select\AddRoomImage\depth'
    
    
    # 排除文件，文件夹地址 
    ExcludeImageFolderGroupName = [r'D:\xiongbiao\Data\HumanDete\Dataset\data_class1_958\trainval2014.txt', # data_class1_958
                                   r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210602_TestDepthError\dataset_test\val2014.txt', # RGBD_20210602_TestDepthError
                                   r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210517_NoAnno\dataset_test\val2014.txt'] # RGBD_20210517_NoAnno

    # # 原始 depth 文件夹地址
    # SrcdepthFolderGroupName = [r'Y:\XbiaoData\龙岗新仓程序备份_20201109\Anno\SrcImage\ServerAlarmData\20201105\depth',
    #                             ]
    # # 原始 RGB 文件夹地址
    # SrcRGBFolderGroupName = [r'Y:\XbiaoData\龙岗新仓程序备份_20201109\Anno\SrcImage\ServerAlarmData\20201105\RGB',
    #                             ]
    # 挑选 RGB 文件夹地址
    CompareRGBImageFolderGroupName = [r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\SelectHardCaseImage\RGB_Select\AddRoomImage\RGB',
                               ]
    
    # 目标 RGB 文件夹地址
    DestRGBFolderGroupName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\SelectHardCaseImage\RGB'
    # 目标 depth 文件夹地址
    DestdepthFolderGroupName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\SelectHardCaseImage\depth'                   
    # 目标 Colormap 文件夹地址
    DestColormapFolderGroupName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\SelectHardCaseImage\Colormap'

    
    ################ 复制之前 RGB 标注的文件, RGB 文件地址 ################
    # DatasetAddHardImage_RGBSrcLabel = r'D:\xiongbiao\Data\HumanDete\NewLG\RGB\labels'
    DatasetAddHardImage_RGBSrcLabel = r'Y:\XbiaoData\龙岗新仓程序备份_20201109\huquigen\data_rgb\coco\labels'
    DatasetAddHardImage_RGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210315\DatasetAddHardImage\RGB'
    DatasetAddHardImage_RGBLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210315\DatasetAddHardImage\RGBLabel'

    ################ 新标注的文件地址 ################
    SrcXmlLabelColormapFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\AnnoImageVott\Colormap\ColormapHard_202106-PascalVOC-export\Annotations'
    DestXmlLabelColormapFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\AnnoImageVott\Colormap_Xml\Label'
    SrcXmlLabelRGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\AnnoImageVott\RGB\RGBHard_202106-PascalVOC-export\Annotations'
    DestXmlLabelRGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\AnnoImageVott\RGB_Xml\Label'
                                  
    ################ 合并HardImage 后的图像文件地址 ################
    SrcAddHardLabelColormapFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\ColormapLabel'
    SrcAddHardColormapFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\Colormap'
    DestPlotAddHardColormapFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\ColormapAnnoPlot'
    
    SrcAddHardLabelRGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBLabel'
    SrcAddHardRGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGB'
    DestPlotAddHardRGBFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBAnnoPlot'
    
    # 转换坐标后的RGB数据
    SrcAddHardLabelRGBFolderName_MatchToDepth = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBLabel_MatchToDepth'
    SrcAddHardRGBFolderName_MatchToDepth = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGB_MatchToDepth'
    DestPlotAddHardRGBFolderName_MatchToDepth = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBAnnoPlot_MatchToDepth'
    RotParamFileName = r'rgb2depth_param.txt'
    
    ################ 合并HardImage 后的RGBD 图像文件地址 ################
    SrcSelectRGBDLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBDLabel_Select'
    
    DestRGBDImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBDImage'
    DestRGBDLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBDLabel'

    ################ 合并HardImage 后的RGBD Train/val 数据地址 ################
    # 3-class
    SrcDatasetAddHardTrainvalImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\images_all'
    SrcDatasetAddHardTrainvalLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\labels_all'
    DestDatasetAddHardTrainvalImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\data_class3\images'
    DestDatasetAddHardTrainvalLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\data_class3\labels'
    # 1-class
    Dest1ClassDatasetAddHardTrainvalImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\data_class1\images'
    Dest1ClassDatasetAddHardTrainvalLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\Dataset\data_class1\labels'


    ################ 操作步骤 ################
    Step_FindSameFrameImagesFlag = 0
    Step_CopySameImagesFlag = 0 # 以前 RGB 标注的文件太少，现重新标注
    Step_TransAnnoFilesFlag = 0 # 将从xml文件转为txt文件
    Step_TransRGBMatchToDepthFlag = 0 # 将RGB转换到Depth大小，包括：图像转换、label转换
    Step_PlotImageAnnoFilesFlag = 0 # 显示标注文件，（便于查找RGBD label信息）
    Step_GenSelectRGBDImageAnnoFilesFlag = 0 # 根据挑选的RGBD label文件，生成RGBD挑选的标注文件
    Step_GenRGBDImageAnnoFilesFlag = 0 # 生成有效的RGBD数据，包括：RGB/Depth 图像（2类图像）和 RGB/Depth/RGBD 的 label（3类label）
    Step_TrainvalImagesFlag = 0 # 生成训练集/验证集，3类
    Step_1ClassTrainvalImagesFlag = 0 # 生成训练集/验证集, 1 类
    
    
    # step1：根据挑选出来的RGB图像找到对应的depth文件，并生成Colormap图像
    if Step_FindSameFrameImagesFlag == 1:
        # 选择部分文件，去除之前标注过 / 待测试的文件
        if True:
            ExcludeFileName = []
            for OneExcludeFolderName in ExcludeImageFolderGroupName:
                # read one txt file
                OneTrainvalFileName = ReSetDatasetTrainvalInfo.ReadTrainvalTxt(OneExcludeFolderName)
                for i_OneTrainvalFileName in OneTrainvalFileName:
                    ExcludeFileName.append(i_OneTrainvalFileName)
            SelectAnnoRGBFolderNameGlob = glob.glob(SelectAnnoRGBFolderName+'\\*.jpg') # RGB
            for i_SelectAnnoRGBFileName in SelectAnnoRGBFolderNameGlob:
                i_SelectAnnoRGBFileName = os.path.basename(i_SelectAnnoRGBFileName)
                if i_SelectAnnoRGBFileName.replace('Color', 'Depth').replace('jpg', 'png') in ExcludeFileName:
                    print('i_SelectAnnoRGBFileName = ', i_SelectAnnoRGBFileName)
                    continue
                else:
                    copyfile(os.path.join(SelectAnnoRGBFolderName, i_SelectAnnoRGBFileName), os.path.join(SrcAnnoRGBFolderName, i_SelectAnnoRGBFileName))
            SelectAnnoDepthFolderNameGlob = glob.glob(SelectAnnoDepthFolderName+'\\*.jpg.dep') # Depth
            for i_SelectAnnoDepthFileName in SelectAnnoDepthFolderNameGlob:
                i_SelectAnnoDepthFileName = os.path.basename(i_SelectAnnoDepthFileName)
                if i_SelectAnnoDepthFileName.replace('jpg.dep', 'png') in ExcludeFileName:
                    print('i_SelectAnnoDepthFileName = ', i_SelectAnnoDepthFileName)
                    continue
                else:
                    copyfile(os.path.join(SelectAnnoDepthFolderName, i_SelectAnnoDepthFileName), os.path.join(SrcAnnoDepthFolderName, i_SelectAnnoDepthFileName))
        
        # 复制depth 文件
        #   手动复制
        
        # 根据depth 文件生成 Colormap 图像
        depthFileToColormapFile(DestdepthFolderGroupName, DestColormapFolderGroupName, FilePostfixGroupName=['.jpg.dep', '.png'])
        # 复制RGB 文件, # 变换RGB图像尺寸
        TransFileName(SrcAnnoRGBFolderName, DestRGBFolderGroupName, ['.jpg', '.png'])
        DestRGBImageSize = [540, 960]
        ResizeImages(DestRGBFolderGroupName, DestRGBFolderGroupName, DestRGBImageSize, FileSubStrGroup = ['.jpg', '.png'])
    
    # step2：根据文件名找到对应的已有标注文件，生成未标注文件夹
    if Step_CopySameImagesFlag == 1:
        # 找到哪些文件未标注
        FilePostfixGroupName = ['.png', '.txt']# 图像/label 两种数据后缀名
        SrcLabelFolderName = os.path.join(DatasetAddHardImage_RGBSrcLabel, 'train2014')
        CopySemiSameFiles(DatasetAddHardImage_RGBFolderName, SrcLabelFolderName, DatasetAddHardImage_RGBLabelFolderName, FilePostfixGroupName = FilePostfixGroupName)
        SrcLabelFolderName = os.path.join(DatasetAddHardImage_RGBSrcLabel, 'val2014')
        CopySemiSameFiles(DatasetAddHardImage_RGBFolderName, SrcLabelFolderName, DatasetAddHardImage_RGBLabelFolderName, FilePostfixGroupName = FilePostfixGroupName)

    #（根据标注的RGB和Depth，从xml文件转为txt文件）
    # step3：根据挑选的RGBD标注文件，转换RGBD文件信息
    if Step_TransAnnoFilesFlag == 1:
        ClassName = ['lie', 'sit', 'stand'] # ['lying', 'sitting', 'standing']
        CurAnnoLabelTransXmlToTxt = AnnoLabelTransXmlToTxt(SrcXmlLabelColormapFolderName, DestXmlLabelColormapFolderName, ClassName)
        CurAnnoLabelTransXmlToTxt.MultiAnnoLabelTransXmlToTxt()
        CurAnnoLabelTransXmlToTxt2 = AnnoLabelTransXmlToTxt(SrcXmlLabelRGBFolderName, DestXmlLabelRGBFolderName, ClassName)
        CurAnnoLabelTransXmlToTxt2.MultiAnnoLabelTransXmlToTxt()
        
    # step4：将RGB转换到Depth大小，包括：图像转换、label转换
    if Step_TransRGBMatchToDepthFlag == 1:
        DestImageSize = [512, 424]
        CurTwoModeDataTransform = TwoModeDataTransform(SrcAddHardRGBFolderName, SrcAddHardLabelRGBFolderName, SrcAddHardRGBFolderName_MatchToDepth, SrcAddHardLabelRGBFolderName_MatchToDepth, DestImageSize, RotParamFileName)
        CurTwoModeDataTransform.MultiAnnoFrameTransform()
        
    # step5：显示合并之后的数据
    if Step_PlotImageAnnoFilesFlag == 1:
        PlotImageAnnoInfo(SrcAddHardLabelColormapFolderName, SrcAddHardColormapFolderName, DestPlotAddHardColormapFolderName, LabelPostfixName='.txt')
        PlotImageAnnoInfo(SrcAddHardLabelRGBFolderName, SrcAddHardRGBFolderName, DestPlotAddHardRGBFolderName, LabelPostfixName='.txt')
        PlotImageAnnoInfo(SrcAddHardLabelRGBFolderName_MatchToDepth, SrcAddHardRGBFolderName_MatchToDepth, DestPlotAddHardRGBFolderName_MatchToDepth, LabelPostfixName='.txt')

        
    #（根据标注的RGB和Depth，对比两种数据，生成RGBD标注文件）
    # step6：根据挑选的RGBD标注文件，转换RGBD文件信息,[手动复制 RGBDImage_Select 中label至 RGBDLabel_Select]
    if Step_GenSelectRGBDImageAnnoFilesFlag == 1:
        # 生成 RGBD Label 文件
        SrcRGBDImage_SelectFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\\' + SelectDataDateName + '\DatasetAddHardImage\JC_1000\RGBDImage_Select'
        RGBDLabel_FindRGBLabel(SrcRGBDImage_SelectFolderName, SrcRGBDImage_SelectFolderName, SrcAddHardLabelRGBFolderName_MatchToDepth)
        
    # step7: 生成有效的RGBD数据，包括：RGB/Depth 图像（2类图像）和 RGB/Depth/RGBD 的 label（3类label）
    if Step_GenRGBDImageAnnoFilesFlag == 1:
        GetRGBDImageAndLabel(SrcAddHardColormapFolderName, SrcAddHardRGBFolderName_MatchToDepth, \
                             SrcAddHardLabelColormapFolderName, SrcAddHardLabelRGBFolderName_MatchToDepth, SrcSelectRGBDLabelFolderName, \
                             DestRGBDImageFolderName, DestRGBDLabelFolderName)
    
    # 生成训练集/验证集 (手动复制 RGBDLabel/RGBDImage 至 labels_all/images_all)
    if Step_TrainvalImagesFlag == 1: # 生成训练集/验证集
        CurGenTrainvalDataset_RGBD3Label = GenTrainvalDataset_RGBD3Label(SrcDatasetAddHardTrainvalImageFolderName, SrcDatasetAddHardTrainvalLabelFolderName, \
                                                                         DestDatasetAddHardTrainvalImageFolderName, DestDatasetAddHardTrainvalLabelFolderName)
        CurGenTrainvalDataset_RGBD3Label.GenRGBDTrainvalDataset()
    
    # 生成训练集/验证集, 1 类
    if Step_1ClassTrainvalImagesFlag == 1: # 生成训练集/验证集, 1 类
        LabelPostfixName = '.txt'
        TrainvalType = 'train2014'
        Src3ClassLabelFolderName = os.path.join(DestDatasetAddHardTrainvalLabelFolderName, TrainvalType)
        Dest1ClassLabelFolderName = os.path.join(Dest1ClassDatasetAddHardTrainvalLabelFolderName, TrainvalType)
        LabelTransClass3To1(Src3ClassLabelFolderName, Dest1ClassLabelFolderName, LabelPostfixName=LabelPostfixName)
        TrainvalType = 'val2014'
        Src3ClassLabelFolderName = os.path.join(DestDatasetAddHardTrainvalLabelFolderName, TrainvalType)
        Dest1ClassLabelFolderName = os.path.join(Dest1ClassDatasetAddHardTrainvalLabelFolderName, TrainvalType)
        LabelTransClass3To1(Src3ClassLabelFolderName, Dest1ClassLabelFolderName, LabelPostfixName=LabelPostfixName)
    return 0



def RGBDLabel_FindRGBLabel(SrcSelectRGBImageFolderName, DestSelectRGBLabelFolderName, CompareRGBImageLabelFolderName):
    """
    功能：生成RGBDLabel，手动挑选那些Depth标注不准确的RGB图像，从RGB图像中找到对应的RGBlabel作为RGBDlabel
    """
    # 目标文件名
    SrcImagePostfixName = '.png'
    SrcLabelPostfixName = '.txt'
    SrcRGBImagePrefixName = 'Color'
    SrcDepthImagePrefixName = 'Depth'
    # 遍历图像
    for root, dirs, files in os.walk(SrcSelectRGBImageFolderName):
        for filename in files:
            if filename.endswith(SrcImagePostfixName):
                CurLabelFileName = os.path.join(CompareRGBImageLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName))
                if os.path.exists(CurLabelFileName):
                    DestCurLabelFileName = os.path.join(DestSelectRGBLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName).replace(SrcRGBImagePrefixName, SrcDepthImagePrefixName))
                    copyfile(CurLabelFileName, DestCurLabelFileName)
                else:
                    print(' label not exist: ', filename)
    return 0

def RGBDLabel_FindRGBTransLabel(SrcSelectRGBImageFolderName, DestSelectRGBLabelFolderName, CompareRGBImageLabelFolderName, CompareImageFolderName=None, InValidPixel=[0,0,128]):
    """
    功能：生成RGBDLabel，手动挑选那些Depth标注不准确的RGB图像，从RGB图像中找到对应的RGBlabel作为RGBDlabel
    添加功能：删除 RGB边界目标框对应深度数据差的数据
    """
    # 边界 bbox 目标中无效点数据个数
    InValidPixelRito = 0.85 
    # 目标文件名
    SrcImagePostfixName = '.png'
    SrcLabelPostfixName = '.txt'
    SrcRGBImagePrefixName = 'Color'
    SrcDepthImagePrefixName = 'Depth'
    # 遍历图像
    for root, dirs, files in os.walk(SrcSelectRGBImageFolderName):
        for filename in files:
            if filename.endswith(SrcImagePostfixName):
                CurLabelFileName = os.path.join(CompareRGBImageLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName))
                # 计算有效的RGB目标框
                if CompareImageFolderName==None:
                    if os.path.exists(CurLabelFileName):
                        DestCurLabelFileName = os.path.join(DestSelectRGBLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName).replace(SrcRGBImagePrefixName, SrcDepthImagePrefixName))
                        copyfile(CurLabelFileName, DestCurLabelFileName)
                    else:
                        print(' label not exist: ', filename)
                else:
                    CurDepthFileName = os.path.join(CompareImageFolderName, filename.replace(SrcRGBImagePrefixName, SrcDepthImagePrefixName))
                    if os.path.exists(CurLabelFileName) and os.path.exists(CurDepthFileName):
                        # save txt file
                        OneLabelInfoDest = []
                        DestCurLabelFileName = os.path.join(DestSelectRGBLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName).replace(SrcRGBImagePrefixName, SrcDepthImagePrefixName))
                        # read txt file
                        OneLabelInfoSrc = []
                        fp_src = open(CurLabelFileName, 'r')
                        OneLabelInfoLines = fp_src.readlines()
                        for i_OneLabelInfo in OneLabelInfoLines:
                            for i_data in i_OneLabelInfo.split():
                                OneLabelInfoSrc.append(float(i_data))
                        OneLabelInfoSrc = np.array(OneLabelInfoSrc)
                        CurDepthLabelInfo = OneLabelInfoSrc.reshape([int(OneLabelInfoSrc.shape[0]/5),5])
                        fp_src.close()
                        # read Depth image
                        SrcDepthImageData = cv2.imread(CurDepthFileName) # [424,512,3]
                        # 选择RGB边界目标框对应深度数据差的数据
                        for i_bbox in range(CurDepthLabelInfo.shape[0]):
                            BboxXmin = int(SrcDepthImageData.shape[1]*(CurDepthLabelInfo[i_bbox, 1]-CurDepthLabelInfo[i_bbox, 3]/2))
                            BboxYmin = int(SrcDepthImageData.shape[0]*(CurDepthLabelInfo[i_bbox, 2]-CurDepthLabelInfo[i_bbox, 4]/2))
                            BboxXmax = int(SrcDepthImageData.shape[1]*(CurDepthLabelInfo[i_bbox, 1] + CurDepthLabelInfo[i_bbox, 3]/2))
                            BboxYmax = int(SrcDepthImageData.shape[0]*(CurDepthLabelInfo[i_bbox, 2] + CurDepthLabelInfo[i_bbox, 4]/2))
                            # 判断目标是否在边界
                            CurObjNearEdgeFlag = False
                            if np.abs(BboxXmin-0)<15 or np.abs(BboxXmax-SrcDepthImageData.shape[1])<15:
                                CurObjNearEdgeFlag = True
                            # BBox Depth data
                            CurDepthLabelInfoBboxData = SrcDepthImageData[BboxYmin:BboxYmax, BboxXmin:BboxXmax, :]
                            InValidPixelIndex = (CurDepthLabelInfoBboxData[:,:,0]==InValidPixel[0]) & (CurDepthLabelInfoBboxData[:,:,1]==InValidPixel[1]) & (CurDepthLabelInfoBboxData[:,:,2]==InValidPixel[2])
                            CurDepthLabelInfoBboxDataInValid  = CurDepthLabelInfoBboxData[InValidPixelIndex]
                            CurDepthLabelInfoBboxDataInValidPixelRito = CurDepthLabelInfoBboxDataInValid.shape[0]/(CurDepthLabelInfoBboxData.shape[0]*CurDepthLabelInfoBboxData.shape[1])
                            # 目标有效性
                            if  CurObjNearEdgeFlag==True and CurDepthLabelInfoBboxDataInValidPixelRito>InValidPixelRito: # 无效目标
                                print('  exist valid bbox: ', filename)
                                continue
                            else: 
                                OneLabelInfoDest.append(OneLabelInfoLines[i_bbox]) # 有效目标
                        # 保存文件
                        if len(OneLabelInfoDest)==0:
                            print('no valid bbox: ', filename)
                        else:
                            fp = open(DestCurLabelFileName, 'w')
                            for i_image_name in OneLabelInfoDest: #
                                fp.writelines(i_image_name)
                            fp.close()
                            
                    else:
                        if os.path.exists(CurLabelFileName):
                            print(' depth image not exist: ', filename)
                            DestCurLabelFileName = os.path.join(DestSelectRGBLabelFolderName, filename.replace(SrcImagePostfixName, SrcLabelPostfixName).replace(SrcRGBImagePrefixName, SrcDepthImagePrefixName))
                            copyfile(CurLabelFileName, DestCurLabelFileName)
                        else:
                            print(' label not exist: ', filename)
                    
    return 0

def GetRGBDImageAndLabel(SrcDepthImageFolderName, SrcRGBImageFolderName, \
                         SrcDepthLabelFolderName, SrcRGBLabelFolderName, SrcRGBDLabelFolderName, \
                         DestRGBDImageFolderName, DestRGBDLabelFolderName):
    """
    功能：获取RGBD的Image/label
    """
    # 目标文件名
    SrcImagePostfixName = '.png'
    SrcLabelPostfixName = '.txt'
    SrcRGBImagePrefixName = 'Color'
    SrcDepthImagePrefixName = 'Depth'
    # 遍历图像
    for root, dirs, files in os.walk(SrcDepthLabelFolderName):
        for filename in files:
            if filename.endswith(SrcLabelPostfixName):
                CurDepthLabelName = os.path.join(SrcDepthLabelFolderName, filename)
                CurRGBLabelName = os.path.join(SrcRGBLabelFolderName, filename.replace(SrcDepthImagePrefixName, SrcRGBImagePrefixName))
                CurRGBDLabelName = os.path.join(SrcRGBDLabelFolderName, filename)
                CurDepthImageName = os.path.join(SrcDepthImageFolderName, filename.replace(SrcLabelPostfixName, SrcImagePostfixName))
                CurRGBImageName = os.path.join(SrcRGBImageFolderName, filename.replace(SrcLabelPostfixName, SrcImagePostfixName).replace(SrcDepthImagePrefixName, SrcRGBImagePrefixName))
                # 文件是否存在
                if not (os.path.exists(CurDepthLabelName) and os.path.exists(CurRGBLabelName) and os.path.exists(CurRGBDLabelName) and \
                        os.path.exists(CurDepthImageName) and os.path.exists(CurRGBImageName)):
                    continue
                print(filename)
                # 复制文件
                DestDepthLabelName = os.path.join(DestRGBDLabelFolderName, filename.replace(SrcLabelPostfixName, '_Depth'+SrcLabelPostfixName))
                DestRGBLabelName = os.path.join(DestRGBDLabelFolderName, filename.replace(SrcLabelPostfixName, '_RGB'+SrcLabelPostfixName))
                DestRGBDLabelName = os.path.join(DestRGBDLabelFolderName, filename)
                DestDepthImageName = os.path.join(DestRGBDImageFolderName, filename.replace(SrcLabelPostfixName, SrcImagePostfixName))
                DestRGBImageName = os.path.join(DestRGBDImageFolderName, filename.replace(SrcLabelPostfixName, '_RGB'+SrcImagePostfixName))
                # copyfile
                copyfile(CurDepthLabelName, DestDepthLabelName)
                copyfile(CurRGBLabelName, DestRGBLabelName)
                copyfile(CurRGBDLabelName, DestRGBDLabelName)
                copyfile(CurDepthImageName, DestDepthImageName)
                copyfile(CurRGBImageName, DestRGBImageName)
    return 0


if __name__ == '__main__':
    print('Start.')
    
    TestCase = 1
    
    if TestCase == 1: # 测试整个步骤 FindRGBDImageFromHardImage 
        SelectDataDateName = r'RGBD_20210603_AddData'
        FindRGBDImageFromHardImage(SelectDataDateName=SelectDataDateName)
        
    elif TestCase == 2: # 测试函数 RGBDLabel_FindRGBTransLabel
        SrcSelectRGBImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\JC_1000\RGBDImage_Select_TransRGBBbox'
        DestSelectRGBLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\JC_1000\RGBDImage_Select_TransRGBBbox'
        CompareRGBImageLabelFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\JC_1000\RGBLabel_MatchToDepth'
        CompareDepthImageFolderName = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\JC_1000\Colormap'
        
        # 更改对应的 Depth Label 文件
        SrcLabelFolderNameTrain= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class3_1000\labels\train2014'
        SrcLabelFolderNameVal = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class3_1000\labels\val2014'
        Src_1C_LabelFolderNameTrain= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class1_1000\labels\train2014'
        Src_1C_LabelFolderNameVal = r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class1_1000\labels\val2014'

        # 删除多余的图像和label文件
        SrcDatasetFolderName= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class3_1000'
        DestDatasetFolderName= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class3_895'
        Src_1C_DatasetFolderName= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class1_1000'
        Dest_1C_DatasetFolderName= r'D:\xiongbiao\Data\HumanDete\NewLG\RGBD_20210428\DatasetAddHardImage\ReSelectValidImage\NewDataset_TransRGBLabel_20210428\data_class1_895'
    
        # 从RGB-Label中重新生成RGBD-Label 信息
        if False:
            RGBDLabel_FindRGBTransLabel(SrcSelectRGBImageFolderName, DestSelectRGBLabelFolderName, CompareRGBImageLabelFolderName, CompareImageFolderName=CompareDepthImageFolderName)

        if False:
            # 遍历文件
            DestSelectRGBLabelGroup = []
            for root, dirs, files in os.walk(DestSelectRGBLabelFolderName):
                for filename in files:
                    if filename.endswith('.txt'):
                        DestSelectRGBLabelGroup.append(filename)
                        
            for root, dirs, files in os.walk(SrcLabelFolderNameTrain):
                for filename in files:
                    if filename.endswith('.txt'):
                        if filename in DestSelectRGBLabelGroup:
                            src_filename = os.path.join(DestSelectRGBLabelFolderName, filename)
                            dst_filename = os.path.join(SrcLabelFolderNameTrain, filename)
                            copyfile(src_filename, dst_filename)
            for root, dirs, files in os.walk(SrcLabelFolderNameVal):
                for filename in files:
                    if filename.endswith('.txt'):
                        if filename in DestSelectRGBLabelGroup:
                            src_filename = os.path.join(DestSelectRGBLabelFolderName, filename)
                            dst_filename = os.path.join(SrcLabelFolderNameVal, filename)
                            copyfile(src_filename, dst_filename)            
                   
        # 生成 1 类数据
        if False:
            LabelPostfixName = '.txt'
            TrainvalType = 'train2014'
            Src3ClassLabelFolderName = SrcLabelFolderNameTrain
            Dest1ClassLabelFolderName = Src_1C_LabelFolderNameTrain
            LabelTransClass3To1(Src3ClassLabelFolderName, Dest1ClassLabelFolderName, LabelPostfixName=LabelPostfixName)
            TrainvalType = 'val2014'
            Src3ClassLabelFolderName = SrcLabelFolderNameVal
            Dest1ClassLabelFolderName = Src_1C_LabelFolderNameVal
            LabelTransClass3To1(Src3ClassLabelFolderName, Dest1ClassLabelFolderName, LabelPostfixName=LabelPostfixName)       
                    
              
        # 删除多余的图像和label文件
        if True:
            # SrcDatasetFolderNameImageTrain = os.path.join(SrcDatasetFolderName, 'images', 'train2014')
            # SrcDatasetFolderNameImageVal = os.path.join(SrcDatasetFolderName, 'images', 'val2014')
            # SrcDatasetFolderNameLabelTrain = os.path.join(SrcDatasetFolderName, 'labels', 'train2014')
            # SrcDatasetFolderNameLabelVal = os.path.join(SrcDatasetFolderName, 'labels', 'val2014')                    
            # DestDatasetFolderNameImageTrain = os.path.join(DestDatasetFolderName, 'images', 'train2014')
            # DestDatasetFolderNameImageVal = os.path.join(DestDatasetFolderName, 'images', 'val2014')
            # DestDatasetFolderNameLabelTrain = os.path.join(DestDatasetFolderName, 'labels', 'train2014')
            # DestDatasetFolderNameLabelVal = os.path.join(DestDatasetFolderName, 'labels', 'val2014')
            
            SrcDatasetFolderNameImageTrain = os.path.join(Src_1C_DatasetFolderName, 'images', 'train2014')
            SrcDatasetFolderNameImageVal = os.path.join(Src_1C_DatasetFolderName, 'images', 'val2014')
            SrcDatasetFolderNameLabelTrain = os.path.join(Src_1C_DatasetFolderName, 'labels', 'train2014')
            SrcDatasetFolderNameLabelVal = os.path.join(Src_1C_DatasetFolderName, 'labels', 'val2014')                    
            DestDatasetFolderNameImageTrain = os.path.join(Dest_1C_DatasetFolderName, 'images', 'train2014')
            DestDatasetFolderNameImageVal = os.path.join(Dest_1C_DatasetFolderName, 'images', 'val2014')
            DestDatasetFolderNameLabelTrain = os.path.join(Dest_1C_DatasetFolderName, 'labels', 'train2014')
            DestDatasetFolderNameLabelVal = os.path.join(Dest_1C_DatasetFolderName, 'labels', 'val2014')
            
            if not os.path.exists(DestDatasetFolderNameImageTrain):
                os.makedirs(DestDatasetFolderNameImageTrain)
            if not os.path.exists(DestDatasetFolderNameImageVal):
                os.makedirs(DestDatasetFolderNameImageVal)
            if not os.path.exists(DestDatasetFolderNameLabelTrain):
                os.makedirs(DestDatasetFolderNameLabelTrain)
            if not os.path.exists(DestDatasetFolderNameLabelVal):
                os.makedirs(DestDatasetFolderNameLabelVal)
            # read train txt file
            SrcDatasetFolderNameImageTrainTxt= os.path.join(SrcDatasetFolderName, 'labels', 'train2014.txt')
            fp = open(SrcDatasetFolderNameImageTrainTxt, 'r')
            SrcLabelGroup = fp.readlines()
            fp.close()
            for i_image_name in SrcLabelGroup:
                cur_file_name = i_image_name.strip().split('/')[-1]
                src_image_file_name = os.path.join(SrcDatasetFolderNameImageTrain, cur_file_name)
                src_image_rgb_file_name = os.path.join(SrcDatasetFolderNameImageTrain, cur_file_name.replace('.png', '_RGB.png'))
                src_label_file_name = os.path.join(SrcDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt'))
                src_label_rgb_file_name = os.path.join(SrcDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt').replace('.txt', '_RGB.txt'))
                src_label_depth_file_name = os.path.join(SrcDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt').replace('.txt', '_Depth.txt'))
                dest_image_file_name = os.path.join(DestDatasetFolderNameImageTrain, cur_file_name)
                dest_image_rgb_file_name = os.path.join(DestDatasetFolderNameImageTrain, cur_file_name.replace('.png', '_RGB.png'))
                dest_label_file_name = os.path.join(DestDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt'))
                dest_label_rgb_file_name = os.path.join(DestDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt').replace('.txt', '_RGB.txt'))
                dest_label_depth_file_name = os.path.join(DestDatasetFolderNameLabelTrain, cur_file_name.replace('.png', '.txt').replace('.txt', '_Depth.txt'))
                if os.path.exists(src_image_file_name) and os.path.exists(src_label_file_name):
                    copyfile(src_image_file_name, dest_image_file_name)
                    copyfile(src_image_rgb_file_name, dest_image_rgb_file_name)
                    copyfile(src_label_file_name, dest_label_file_name)
                    copyfile(src_label_rgb_file_name, dest_label_rgb_file_name)
                    copyfile(src_label_depth_file_name, dest_label_depth_file_name)
            # read val txt file
            SrcDatasetFolderNameImageValTxt= os.path.join(SrcDatasetFolderName, 'labels', 'val2014.txt')
            fp = open(SrcDatasetFolderNameImageValTxt, 'r')
            SrcLabelGroup = fp.readlines()
            fp.close()
            for i_image_name in SrcLabelGroup:
                cur_file_name = i_image_name.strip().split('/')[-1]
                src_image_file_name = os.path.join(SrcDatasetFolderNameImageVal, cur_file_name)
                src_image_rgb_file_name = os.path.join(SrcDatasetFolderNameImageVal, cur_file_name.replace('.png', '_RGB.png'))
                src_label_file_name = os.path.join(SrcDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt'))
                src_label_rgb_file_name = os.path.join(SrcDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt').replace('.txt', '_RGB.txt'))
                src_label_depth_file_name = os.path.join(SrcDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt').replace('.txt', '_Depth.txt'))
                dest_image_file_name = os.path.join(DestDatasetFolderNameImageVal, cur_file_name)
                dest_image_rgb_file_name = os.path.join(DestDatasetFolderNameImageVal, cur_file_name.replace('.png', '_RGB.png'))
                dest_label_file_name = os.path.join(DestDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt'))
                dest_label_rgb_file_name = os.path.join(DestDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt').replace('.txt', '_RGB.txt'))
                dest_label_depth_file_name = os.path.join(DestDatasetFolderNameLabelVal, cur_file_name.replace('.png', '.txt').replace('.txt', '_Depth.txt'))
                if os.path.exists(src_image_file_name) and os.path.exists(src_label_file_name):
                    copyfile(src_image_file_name, dest_image_file_name)
                    copyfile(src_image_rgb_file_name, dest_image_rgb_file_name)
                    copyfile(src_label_file_name, dest_label_file_name)
                    copyfile(src_label_rgb_file_name, dest_label_rgb_file_name)
                    copyfile(src_label_depth_file_name, dest_label_depth_file_name)    
    
    
    
    
    
    
    