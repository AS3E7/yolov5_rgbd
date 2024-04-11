# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:58:27 2020

@author: HYD
"""

import numpy as np
import os
from shutil import copyfile
import matplotlib.pyplot as plt

import HSFFuns
import CloudPointFuns
import Depth2Pts
import FileHsf

def GenImageFromDepthFile(DepthFileName, DepthWidth, DepthHeight, TransDepthRange, SaveImageFileName, TransFormat='Colormap'):
    """
    从深度数据中获取深度图片数据
    """
    # read one depth file
    DepthData = HSFFuns.ReadOneFrameDepthFile(DepthFileName, DepthWidth, DepthHeight)
    
    # depth valid range
    TransDepthRange = TransDepthRange
    # save file name
    SaveImageName = SaveImageFileName # SaveImageName = ''
    # depth image format, 'Gray'/'Colormap'
    # TransFormat = 'Colormap'
    # generate depth image and save image
    DepthImageData = HSFFuns.GenImageFromDepth(DepthData, TransDepthRange, SaveImageName, TransFormat)
    
    return DepthImageData
    
def GenImageFromPlyFile(PlyFileName, DepthWidth, DepthHeight, CalibH, TransDepthRange, SaveImageFileName):
    """
    从深度数据中获取深度图片数据
    """
    # read one depth file, [3 x N]
    PointCloudData = CloudPointFuns.ReadKinect2FromPly(PlyFileName) # DepthWidth, DepthHeight
    # calibrate point cloud 
    H = CalibH
    NewPts = CloudPointFuns.Rot3D(H, PointCloudData)# [3 x N]
    NewPts_Z = NewPts[-1,:]
    NewPts_Z = H[2,3] - NewPts_Z
    NewPts_Z = np.reshape(NewPts_Z, [DepthHeight, DepthWidth]) * 1000 # new pts height, [unit: mm]
    
    # depth valid range
    TransDepthRange = TransDepthRange
    # save file name
    SaveImageName = SaveImageFileName # SaveImageName = ''
    # depth image format, 'Gray'/'Colormap'
    TransFormat = 'Colormap'
    # generate depth image and save image
    DepthImageData = HSFFuns.GenImageFromDepth(NewPts_Z, TransDepthRange, SaveImageName, TransFormat)
    
    return DepthImageData
    
def GetDepthFilesFromColorFiles(ColorLabelFolderName, DepthImagesFolderName, SaveColormapFolderName, SaveGrayFolderName, SavedepthFolderName, SaveJSONFolderName):
    """
    根据标注的RGB文件获取对应的深度图片数据: [Colormap，Gray，depth, JSON]
        如：在NJ-3000 RGB数据中找到对应的 Depth 文件数据
    """
    # loop label file
    for root, dirs, files in os.walk(ColorLabelFolderName):
        for image_file in files:
            if image_file.endswith('.json'):
                cur_label_file_name = image_file
            else:
                continue
            print('cur_label_file_name = {}'.format(cur_label_file_name))

            # sensor name, eg: 171
            cur_sensor_name = cur_label_file_name.split('_')[0]
            # depth file name, 171_Color2018-12-06-073000_00321 --> 171_Depth2018-12-06-073000_00321.png
            cur_depth_file_name = cur_label_file_name.split('.json')[0].replace('Color', 'Depth') + '.png'
            # current depth folder name, eg: V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\171
            cur_depth_folder_name = os.path.join(DepthImagesFolderName, cur_sensor_name)
            # Colormap
            if len(SaveColormapFolderName) > 0:
                if os.path.exists(os.path.join(cur_depth_folder_name, cur_depth_file_name)):
                    copyfile(os.path.join(cur_depth_folder_name, cur_depth_file_name), os.path.join(SaveColormapFolderName, cur_depth_file_name))
                else:
                    print('  {} Colormap not exist.'.format(cur_label_file_name))
            # Gray
            if len(SaveGrayFolderName) > 0:
                if os.path.exists(os.path.join(cur_depth_folder_name, 'Gray', cur_depth_file_name)):
                    copyfile(os.path.join(cur_depth_folder_name, 'Gray', cur_depth_file_name), os.path.join(SaveGrayFolderName, cur_depth_file_name))
                else:
                    print('  {} Gray not exist.'.format(cur_label_file_name))
            # depth
            if len(SavedepthFolderName) > 0:
                if os.path.exists(os.path.join(cur_depth_folder_name, 'Gray', cur_depth_file_name.replace('.png', '.depth'))):
                    copyfile(os.path.join(cur_depth_folder_name, 'Gray', cur_depth_file_name.replace('.png', '.depth')), os.path.join(SavedepthFolderName, cur_depth_file_name.replace('.png', '.depth')))
                else:
                    print('  {} depth not exist.'.format(cur_label_file_name))
            # JSON
            if len(SaveJSONFolderName) > 0:
                if os.path.exists(os.path.join(cur_depth_folder_name, cur_depth_file_name)):
                    copyfile(os.path.join(ColorLabelFolderName, cur_label_file_name), os.path.join(SaveJSONFolderName, cur_label_file_name.replace('Color', 'Depth')))
                else:
                    print('  {} depth not exist.'.format(cur_label_file_name))
    return 0
    
def GetCalibDepthFilesFromPlyFiles(PlyFolderName, ParamsFolderName, SaveColormapFolderName, SaveGrayFolderName, SavedepthFolderName, TransDepthRangeInput, DepthWidth, DepthHeight):
    """
    根据 ply文件和配准文件 获取对应的深度数据，并生成对应的图片数据: [Colormap，Gray，depth]
        如：在NJ-3000 ply 和 对应的配准 文件中重新生成 colormap 图片文件
    输入：
        PlyFolderName：ply文件夹
        ParamsFolderName：配准参数文件夹
        SaveColormapFolderName：colormap文件保存地址
        SaveGrayFolderName：gray文件保存地址
        SavedepthFolderName：配准后的depth文件保存地址
    """
    # loop label file
    for root, dirs, files in os.walk(PlyFolderName):
        for image_file in files:
            if image_file.endswith('.ply'):
                cur_ply_file_name = image_file
            else:
                continue
            print('  {}'.format(cur_ply_file_name))

            # sensor name, eg: 171
            cur_sensor_name = cur_ply_file_name.split('_')[0]
            # current sensor calibration info
            cur_calib_H = HSFFuns.ReadCalibParamsInfo(os.path.join(ParamsFolderName, 'H_'+cur_sensor_name+'.txt'))
            # read current ply file, [3 x N]
            cut_src_pts = CloudPointFuns.ReadKinect2FromPly(os.path.join(PlyFolderName, cur_ply_file_name)) # DepthWidth, DepthHeight
            NewPts = CloudPointFuns.Rot3D(cur_calib_H, cut_src_pts)# [3 x N], # calibrate point cloud 
            NewPts_Z = NewPts[-1,:]
            NewPts_Z = - NewPts_Z
            NewPts_Z = np.reshape(NewPts_Z, [DepthWidth, DepthHeight]) * 1000 # new pts height, [unit: mm]
            NewPts_Z = NewPts_Z.transpose()
            # TransDepthRange
            TransDepthRange = []
            TransDepthRange.append(-TransDepthRangeInput[1])
            TransDepthRange.append(-TransDepthRangeInput[0])
                                  
#            CloudPointFuns.SavePt3D2Ply('temp.ply', NewPts.transpose())
#            plt.figure('Test')
#            plt.imshow(NewPts_Z)
#            plt.show
#            print('DepthWidth = {}, DepthHeight = {}'.format(DepthWidth, DepthHeight))
#            print('TransDepthRange = {}'.format(TransDepthRange))
                                  
            # Colormap
            if len(SaveColormapFolderName) > 0:
                SaveColormapImageName = os.path.join(SaveColormapFolderName, cur_ply_file_name.replace('.ply','.png'))
                HSFFuns.GenImageFromDepth(NewPts_Z, TransDepthRange, SaveColormapImageName, TransFormat='Colormap') # [424,512,3]
            # Gray
            if len(SaveGrayFolderName) > 0:
                if SaveGrayFolderName.find('Gray24') == -1:
                    # gray-8
                    SaveGrayImageName = os.path.join(SaveGrayFolderName, cur_ply_file_name.replace('.ply','.png'))
                    if not os.path.exists(SaveGrayImageName):
                        HSFFuns.GenImageFromDepth(NewPts_Z, TransDepthRange, SaveGrayImageName, TransFormat='Gray') # [424,512]
                else:
                    # gray-24
                    SaveGrayImageName = os.path.join(SaveGrayFolderName, cur_ply_file_name.replace('.ply','.png'))
                    if not os.path.exists(SaveGrayImageName):
                        HSFFuns.GenImageFromDepth(NewPts_Z, TransDepthRange, SaveGrayImageName, TransFormat='Gray24') # [424,512,3]

            # depth
            if len(SavedepthFolderName) > 0:
                SavedepthImageName = os.path.join(SavedepthFolderName, cur_ply_file_name.replace('.ply','.depth'))
                # depth data
                DepthDataSave = NewPts_Z.flatten() # 深度数据
                fpDepth = open(SavedepthImageName, 'wb')
                fpDepth.write(DepthDataSave)
                fpDepth.close()
                
#            break

    return 0

def GenMultiFrameFromHSF(InputDataInfo, TransDepthRange):
    """
    根据 HSF 文件读取对应的帧数文件
    """
    print('InputDataInfo ColorHSFFileName = {}'.format(InputDataInfo.ColorHSFFileName))
    
    # HSF 文件地址
    OneColorHSFFileName = InputDataInfo.ColorHSFFileName # color data
    OneDepthHSFFileName = InputDataInfo.DepthHSFFileName # depth data
    # 保存数据类型
    SelectColorHSFFileFlag = InputDataInfo.SaveColor
    SelectDepthHSFFileFlag = InputDataInfo.SaveDepth
    SelectSrcDepthHSFFileFlag = InputDataInfo.SaveSrcDepth
    SelectSrcDepthEachFrameFlag = InputDataInfo.SaveDepthEachFrame
    # frame idx
    FrameRange = InputDataInfo.FrameRange
    # save folder
    SaveFolder = InputDataInfo.SaveFolder
    # prefix
    PrefixName = InputDataInfo.PrefixName
    
    # save image files
    # RGB
    if os.path.exists(OneColorHSFFileName):
        print('exists OneColorHSFFileName')
        if SelectColorHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)
            SaveFolderName = SaveFolder
            
            FileName = OneColorHSFFileName.split('.HSF')[0].split('\\')[-1]
            
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            HSFFuns.SaveHSFFile(OneColorHSFFileName, 'Color', FrameRange, SaveFileName, TransDepthRange, SelectColorHSFFileFlag, SelectDepthHSFFileFlag, SelectSrcDepthHSFFileFlag, SelectSrcDepthEachFrameFlag)
    # Colormap
    if os.path.exists(OneDepthHSFFileName):
        print('exists OneDepthHSFFileName')
        if SelectDepthHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)
            SaveFolderName = SaveFolder
            FileName = OneDepthHSFFileName.split('.HSF')[0].split('\\')[-1]
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            HSFFuns.SaveHSFFile(OneDepthHSFFileName, 'Depth', FrameRange, SaveFileName, TransDepthRange, SelectColorHSFFileFlag, SelectDepthHSFFileFlag, 0, 0)
    # Gray
    if os.path.exists(OneDepthHSFFileName):
        if SelectDepthHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(os.path.join(SaveFolder, 'Gray')):
                os.makedirs(os.path.join(SaveFolder, 'Gray'))
            SaveFolderName = os.path.join(SaveFolder, 'Gray')
            FileName = OneDepthHSFFileName.split('.HSF')[0].split('\\')[-1]
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            HSFFuns.SaveHSFFile(OneDepthHSFFileName, 'Depth', FrameRange, SaveFileName, TransDepthRange, SelectColorHSFFileFlag, 0, SelectSrcDepthHSFFileFlag, SelectSrcDepthEachFrameFlag) # gray+depth 文件存储在 Gray 文件夹中    
            
            
    return 0    
    
def GenMultiFrameFromUncompressHSF(InputDataInfo, TransDepthRange):
    """
    根据 解压缩 HSF 文件读取对应的帧数文件
    """
    print('InputDataInfo ColorHSFFileName = {}'.format(InputDataInfo.ColorHSFFileName))
    
    # HSF 文件地址
    OneColorHSFFileName = InputDataInfo.ColorHSFFileName # color data
    OneDepthHSFFileName = InputDataInfo.DepthHSFFileName # depth data
    # 保存数据类型
    SelectColorHSFFileFlag = InputDataInfo.SaveColor
    SelectDepthHSFFileFlag = InputDataInfo.SaveDepth
    SelectSrcDepthHSFFileFlag = InputDataInfo.SaveSrcDepth
    SelectSrcDepthEachFrameFlag = InputDataInfo.SaveDepthEachFrame
    # frame idx
    FrameRange = InputDataInfo.FrameRange
    # save folder
    SaveFolder = InputDataInfo.SaveFolder
    # prefix
    PrefixName = InputDataInfo.PrefixName
    
    # save image files
    # RGB
    if os.path.exists(OneColorHSFFileName):
        print('exists OneColorHSFFileName')
        if SelectColorHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)
            SaveFolderName = SaveFolder
            
            FileName = OneColorHSFFileName.split('.HSF')[0].split('\\')[-1]
            
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            CurParam = FileHsf.SaveParam()
            CurParam.SaveDepthGray = 0
            CurParam.SaveDepthPseudoColor = 0
            CurParam.SaveDepthEachFrame = 0
            FileHsf.SaveHSFFile(OneColorHSFFileName, 'Color', FrameRange, SaveFileName, TransDepthRange, CurParam)

    # Colormap
    if os.path.exists(OneDepthHSFFileName):
        print('exists OneDepthHSFFileName')
        if SelectDepthHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(SaveFolder):
                os.makedirs(SaveFolder)
            SaveFolderName = SaveFolder
            FileName = OneDepthHSFFileName.split('.HSF')[0].split('\\')[-1]
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            CurParam = FileHsf.SaveParam()
            CurParam.SaveDepthGray = 0
            CurParam.SaveDepthPseudoColor = SelectDepthHSFFileFlag
            CurParam.SaveDepthEachFrame = 0
            FileHsf.SaveHSFFile(OneDepthHSFFileName, 'Depth', FrameRange, SaveFileName, TransDepthRange, CurParam)
            
    # Gray
    if os.path.exists(OneDepthHSFFileName):
        if SelectDepthHSFFileFlag == 1:
            # SaveFileName
            if not os.path.exists(os.path.join(SaveFolder, 'Gray')):
                os.makedirs(os.path.join(SaveFolder, 'Gray'))
            SaveFolderName = os.path.join(SaveFolder, 'Gray')
            FileName = OneDepthHSFFileName.split('.HSF')[0].split('\\')[-1]
            if len(PrefixName) == 0:
                SaveFileName = SaveFolderName + '\\' + FileName
            else:
                SaveFileName = SaveFolderName + '\\' + PrefixName + '_' + FileName
            # save image
            CurParam = FileHsf.SaveParam()
            CurParam.SaveDepthGray = SelectSrcDepthHSFFileFlag
            CurParam.SaveDepthPseudoColor = 0
            CurParam.SaveDepthEachFrame = SelectSrcDepthEachFrameFlag
            FileHsf.SaveHSFFile(OneDepthHSFFileName, 'Depth', FrameRange, SaveFileName, TransDepthRange, CurParam)            
            
    return 0    

if __name__ == '__main__':
    print('Start.')
    
    TestGetDepthFilesFromColorFilesFlag = 0 # 根据标注的RGB文件获取对应的深度图片数据
    TestGetCalibDepthFilesFromPlyFilesFlag = 1 # 根据 ply文件和配准文件 获取对应的深度数据，并生成对应的图片数据
    
    TestGetRGBFilesFromMultiSensorFlag = 0 # 从多个传感器文件中复制RGB数据，此处针对 龙岗监仓 RGB数据
    
    TestRandSelectRGBFilesFromMultiRGBFilesFlag = 0 # 从多个数据中随机挑选部分，如：从调好的RGB数据中挑选部分RGB数据进行内部标注
    
    
    if TestGetDepthFilesFromColorFilesFlag == 1:
        # 在NJ-3000 RGB数据中找到对应的 Depth 文件数据
        ColorLabelFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Label_3000'
        DepthImagesFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image'
        # save folder name
        SaveColormapFolderName = ''
        SaveGrayFolderName = ''
        SavedepthFolderName = ''
        
#        SaveColormapFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Colormap'
#        SaveGrayFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Gray'
#        SavedepthFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\depth'
        SaveJSONFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\JSON'
        
        GetDepthFilesFromColorFiles(ColorLabelFolderName, DepthImagesFolderName, SaveColormapFolderName, SaveGrayFolderName, SavedepthFolderName, SaveJSONFolderName)

    if TestGetCalibDepthFilesFromPlyFilesFlag == 1: # 根据 ply文件和配准文件 获取对应的深度数据，并生成对应的图片数据
        DataSetCase = 2 # if DataSetCase == 1, NJ-3000 数据
                        # if DataSetCase == 2, LG 数据
    
        if DataSetCase == 1: # NJ-3000 数据
            # 1,先生成对应的 ply 文件并保存
            # 2，根据配准文件生成对应的图片结果并保存
            GenPlyFlag = 0 # 是否生成对应的 ply 文件并保存
            GenCalibDepthFilesFlag = 1 # 根据配准文件生成对应的图片结果并保存
            # 原始的 depth 文件地址，calib 配准文件地址
            SrcDepthFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\depth'
            SrcCalibParamFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\CalibParam'
            # 生成的ply文件地址
            SaveGenPlyFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\GenPly'
            # 配准后保存文件地址
    #        SaveColormapFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Calib\Colormap'
    #        SaveGrayFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Calib\Gray'
    #        SavedepthFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Calib\depth'
    
            SaveColormapFolderName = ''
            SaveGrayFolderName = r'V:\PoseEstimate\DepthData\NSDAT\AnnoSelectData\Image\SelectImage_20190909\Depth_3000\Calib\Gray24'
            SavedepthFolderName = ''
    
            # SensorInnerParam
            ycenterF= 202.546464
            xcenterF= 256.685317
            VfoclenInPixelsF= 368.901024
            HfoclenInPixelsF= 368.874187
            # param
            Param = Depth2Pts.SensorInnerParam()
            Param.cx = xcenterF
            Param.cy = ycenterF
            Param.fx = HfoclenInPixelsF
            Param.fy = VfoclenInPixelsF
            SensorInnerParam = Param
            # sensor size
            DepthWidth = 512
            DepthHeight = 424
            # TransDepthRange
            TransDepthRange = [-250, 2500] # [unit: mm]
            
            # 1,先生成对应的 ply 文件并保存
            if GenPlyFlag == 1:
                # depth to pts
                Depth2Pts.TransMultiDepth2Pts(SrcDepthFolderName, SaveGenPlyFolderName, SensorInnerParam, DepthWidth, DepthHeight)
    
            # 2,根据配准文件生成对应的图片结果并保存
            if GenCalibDepthFilesFlag == 1:
                GetCalibDepthFilesFromPlyFiles(SaveGenPlyFolderName, SrcCalibParamFolderName, SaveColormapFolderName, SaveGrayFolderName, SavedepthFolderName, TransDepthRange, DepthWidth, DepthHeight)
            
        elif DataSetCase == 2: #  LG 数据
            # 1,复制对应的depth 到指定文件夹
            # 2,先生成对应的 ply 文件并保存
            # 3，根据配准文件生成对应的图片结果并保存
            CopyDepthFilesFlag = 0 # 是否复制对应的depth 文件
            GenPlyFlag = 0 # 是否生成对应的 ply 文件并保存
            GenCalibDepthFilesFlag = 1 # 根据配准文件生成对应的图片结果并保存
            # 选择获取的数据
            SelectImageCase = 2 #
            if SelectImageCase == 1: # SelectImage
                # 原始的 depth 文件地址，calib 配准文件地址
                SrcDiffDepthFolderName = r'X:\ImageData\LGDAT'
                SrcDepthFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\depth'
                SrcCalibParamFolderName = r'X:\DepthData\LGDAT_2019\SelectData\CalibParam'
                # 生成的ply文件地址
                SaveGenPlyFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\GenPly'
                # 配准后保存文件地址
    #            SaveColormapFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\Calib\Colormap'
    #            SaveGrayFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\Calib\Gray'
    #            SavedepthFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\Calib\depth'
    
                SaveColormapFolderName = r''
                SaveGrayFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\Calib\Gray24'
                SavedepthFolderName = r''
            elif SelectImageCase == 2: # SelectImage_Anno
                SrcDiffDepthFolderName = r''
                SrcDepthFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\depth'
                SrcCalibParamFolderName = r'X:\DepthData\LGDAT_2019\SelectData\CalibParam'
                # 生成的ply文件地址
                SaveGenPlyFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\GenPly'
                # 配准后保存文件地址
                SaveColormapFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\Calib\Colormap'
                SaveGrayFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\Calib\Gray'
                SavedepthFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\Calib\depth'
    
    
            # SensorInnerParam
            ycenterF= 202.546464
            xcenterF= 256.685317
            VfoclenInPixelsF= 368.901024
            HfoclenInPixelsF= 368.874187
            # param
            Param = Depth2Pts.SensorInnerParam()
            Param.cx = xcenterF
            Param.cy = ycenterF
            Param.fx = HfoclenInPixelsF
            Param.fy = VfoclenInPixelsF
            SensorInnerParam = Param
            # sensor size
            DepthWidth = 512
            DepthHeight = 424
            # TransDepthRange
            TransDepthRange = [-250, 2500] # [unit: mm]
            
            # 1,复制对应的depth 到指定文件夹
            if CopyDepthFilesFlag == 1:
                # loop label file
                for root, dirs, files in os.walk(SrcDiffDepthFolderName):
                    for image_file in files:
                        if image_file.endswith('.depth'):
                            cur_depth_file_name = image_file
                        else:
                            continue
                        print('  {}'.format(cur_depth_file_name))
                        # copyfile
                        DestFileName = os.path.join(SrcDepthFolderName, cur_depth_file_name)
                        if not os.path.exists(DestFileName):
                            copyfile(os.path.join(root, cur_depth_file_name), DestFileName)
            
            # 2,先生成对应的 ply 文件并保存
            if GenPlyFlag == 1:
                # depth to pts
                Depth2Pts.TransMultiDepth2Pts(SrcDepthFolderName, SaveGenPlyFolderName, SensorInnerParam, DepthWidth, DepthHeight)
    
            # 3,根据配准文件生成对应的图片结果并保存
            if GenCalibDepthFilesFlag == 1:
                GetCalibDepthFilesFromPlyFiles(SaveGenPlyFolderName, SrcCalibParamFolderName, SaveColormapFolderName, SaveGrayFolderName, SavedepthFolderName, TransDepthRange, DepthWidth, DepthHeight)

    if TestGetRGBFilesFromMultiSensorFlag == 1: # 从多个传感器文件中复制RGB数据，此处针对 龙岗监仓 RGB数据
        # 原始的 RGB 文件地址
#        SrcDiffRGBFolderName = r'X:\ImageData\LGDAT'
#        SrcRGBFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage\RGB\SrcRGB'

        SrcDiffRGBFolderName = r'X:\ImageData\LGDAT'
        SrcRGBFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB'
        SrcRGB2DepthColormapFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\Colormap'
        SrcRGB2DepthRangeFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\depth'
        
        CopyRGBFilesFlag = 1 # 是否复制RGB数据
        CopyRGB2DepthFilesFlag = 1 # 是否复制对应的Depth 数据
            
        # 复制对应的 RGB 到指定文件夹
        if CopyRGBFilesFlag == 1:
            # loop label file
            for root, dirs, files in os.walk(SrcDiffRGBFolderName):
                for image_file in files:
                    if image_file.endswith('.png') and image_file.find('Color')>-1:
                        cur_rgb_file_name = image_file
                    else:
                        continue
                    # sensor name
                    cur_sensor_name = cur_rgb_file_name.split('_')[0]
                    # 选择部分需要转换的传感器数据
#                    if cur_sensor_name == '5' or cur_sensor_name == '6' or cur_sensor_name == '7' or cur_sensor_name == '8':
#                        continue

                    if cur_sensor_name == '1' or cur_sensor_name == '2' or cur_sensor_name == '3' or cur_sensor_name == '4':
                        continue
                    
                    # copyfile
                    if CopyRGB2DepthFilesFlag == 1: # 是否复制对应的Depth 数据
                        CurRGBFileName = os.path.join(root, cur_rgb_file_name)
                        CurDestRGBFileName = os.path.join(SrcRGBFolderName, cur_rgb_file_name)
                        if os.path.exists(CurDestRGBFileName): # 如果数据已经复制到目标地址了，则跳过
                            continue
                        # CopyRGB2DepthFiles
                        CurDepthFileName = os.path.join(root, cur_rgb_file_name.replace('Color', 'Depth'))
                        CurDestDepthFileName = os.path.join(SrcRGB2DepthColormapFolderName, cur_rgb_file_name.replace('Color', 'Depth'))
                        CurDepthRangeFileName = os.path.join(root, 'Gray', cur_rgb_file_name.replace('Color', 'Depth').replace('.png', '.depth'))
                        CurDestDepthRangeFileName = os.path.join(SrcRGB2DepthRangeFolderName, cur_rgb_file_name.replace('Color', 'Depth').replace('.png', '.depth'))
                        if os.path.exists(CurDepthFileName): # 是否存在对应的深度数据
                            print('  {}'.format(cur_rgb_file_name))
                            copyfile(CurDepthFileName, CurDestDepthFileName)
                            copyfile(CurDepthRangeFileName, CurDestDepthRangeFileName)
                            copyfile(CurRGBFileName, CurDestRGBFileName)
                        else:
                            continue
                    else:
                        DestFileName = os.path.join(SrcRGBFolderName, cur_rgb_file_name)
                        if not os.path.exists(DestFileName):
                            print('  {}'.format(cur_rgb_file_name))
                            copyfile(os.path.join(root, cur_rgb_file_name), DestFileName)
                    
                    
    if TestRandSelectRGBFilesFromMultiRGBFilesFlag == 1: # 从多个数据中随机挑选部分，如：从调好的RGB数据中挑选部分RGB数据进行内部标注           
        SrcImageFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select'
        DestSeletcImageFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Bbox'
        DestRemainSeletcImageFolderName = r'X:\DepthData\LGDAT_2019\SelectData\SelectImage_Anno\RGB\SrcRGB _Select_Keypoints'
        SelectNum = 100
        # select image
        ImageList = os.listdir(SrcImageFolderName)
        ImageNumber = len(ImageList)
        SelectIndex = np.random.permutation(ImageNumber)[:SelectNum]
        # copyfile
        for CurIdx, CurImageName in enumerate(ImageList):
            CurImageFileName = os.path.join(SrcImageFolderName, CurImageName)
            DestSeletcImageFileName = os.path.join(DestSeletcImageFolderName, CurImageName)
            DestRemainSeletcImageFileName = os.path.join(DestRemainSeletcImageFolderName, CurImageName)
            if CurIdx in SelectIndex:
                copyfile(CurImageFileName, DestSeletcImageFileName)
            else:
                copyfile(CurImageFileName, DestRemainSeletcImageFileName)
        
    print('End.')