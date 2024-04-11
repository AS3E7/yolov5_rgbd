# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:12:13 2020

@author: HYD
"""

cfg_case = 4 # 选择测试的模式
             # if cfg_case == 1, 选择 ZT-场景
             # if cfg_case == 2, 选择 ZT-认证场景
             # if cfg_case == 3, 选择 LG-场景
             # if cfg_case == 4, 选择 NewLG-场景
   
if cfg_case == 1:
    # 在线/离线,[ OffLine / OnLine ]
    online_offline_type = 'OnLine'
    # 保存结果
    log_info = {
                    'debug_info': '1111100000', # 打印信息: [0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4=打印各类告警检测信息]
                    
                    'save_detect_bbox_image_flag': 0, # 保存bbox目标检测图像结果
                    
                    'save_data_info': '0000100000', # 根据告警类型结果保存数据: [1=未在指定时间休息, 2=未在制定区域监督, 3=厕所区域异常, 4=窗户区域异常, 5=高度异常]
                    'save_data_size_depth': 1000, # 单位：MB， 保存depth文件大小
                    'save_data_size_ply': 1000, # 单位：MB， 保存ply文件大小
                        
                    'save_input_pts_flag': 0, # 保存点云数据
                    
                }
    # 单个传感器参数
    one_sensor_params = {
                         'detect_bbox_pose_type': 1, # 检测目标类别
                         'detect_bbox_pose_index': None, # 检测目标类别序号
                         
                         # 'ssd'
                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
                         'detect_bbox_method_name': 'ssd', # yolov3,ssd
                         'detect_bbox_model_name': 'ZT_OneClass_20200612', # 选择目标检测模型名称
                         'detect_bbox_input_image_size': 300, # 模型输入图像大小
                         'detect_bbox_score_thod': 0.5, # 目标检测目标框分数阈值
                         'detect_bbox_nms_thod': 0.45, # 目标检测目标框nms阈值
                         
                         'alarm_valid_near_time': 0.2, # 告警时间边界的前后一段时间无效
                         'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
                         
                         'alarm_bed_height': 0.1, # 床位高度设置
                         
                         'detect_bbox_use_bg_info': True, # 目标检测是否读取使用信息
                         'use_bg_info': True, # 是否读取背景信息
                         'use_bg_info_height_limit': False, # 高度信息更细背景,eg: [Flase/0.5]
                         'use_bg_info_outlier_dist': 0.05, # 获取边界离群点距离阈值
                         'pt_cluster_num_thod': 200, # 超高聚类最少点云点数据
                         
                         'depth_width': 512, # 传感器深度数据信息
                         'depth_height':424,
                         }
    # 多个传感器参数
    multi_sensor_params = {
              'alarm_valid_near_time': 0.2, # 告警时间边界的前后一段时间无效
              'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
              
              'alone_stay_person_num': 1, # 单人留仓人数
              'alone_stay_sensor_group': None, # 单人留仓内外仓传感器分组序号； [默认值：None]
              
              'internal_supervisor_continuous_time': 0.5, # 内部监管持续多长时间，单位：min
              
              'conflict_person_num': 3, # 多人聚集告警最低人数
              'conflict_near_radius': 0.5, # 多人聚集半径

              }
    
elif  cfg_case == 2:
    # 在线/离线,[ OffLine / OnLine ]
    online_offline_type = 'OnLine'
    # 保存结果
    log_info = {
                    'debug_info': '1111100000', # 打印信息: [0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4=打印各类告警检测信息]
                    
                    'save_detect_bbox_image_flag': 0, # 保存bbox目标检测图像结果
                    
                    'save_data_info': '0000100000', # 根据告警类型结果保存数据: [1=未在指定时间休息, 2=未在制定区域监督, 3=厕所区域异常, 4=窗户区域异常, 5=高度异常]
                    'save_data_size_depth': 1000, # 单位：MB， 保存depth文件大小
                    'save_data_size_ply': 1000, # 单位：MB， 保存ply文件大小
                        
                    'save_input_pts_flag': 0, # 保存点云数据
                    
                }
    # 单个传感器参数
    one_sensor_params = {
                         'detect_bbox_pose_type': 2, # 检测目标类别
                         'detect_bbox_pose_index': None, # 检测目标类别序号
                         
                         # 'ssd'
                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
                         'detect_bbox_method_name': 'ssd', # yolov3,ssd
                         'detect_bbox_model_name': 'ZTRZ_TwoClass_20200612', # 选择目标检测模型名称
                         'detect_bbox_input_image_size': 300, # 模型输入图像大小
                         'detect_bbox_score_thod': 0.42, # 目标检测目标框分数阈值, init:0.5
                         'detect_bbox_nms_thod': 0.4, # 目标检测目标框nms阈值, init:0.4
                         
                         'alarm_valid_near_time': 0.2, # 告警时间边界的前后一段时间无效
                         'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
                         
                         'alarm_bed_height': 0.1, # 床位高度设置
                         
                         'detect_bbox_use_bg_info': True, # 目标检测是否读取使用信息
                         'use_bg_info': True, # 是否读取背景信息
                         'use_bg_info_height_limit': False, # 高度信息更细背景,eg: [Flase/0.5]
                         'use_bg_info_outlier_dist': 0.05, # 获取边界离群点距离阈值
                         'pt_cluster_num_thod': 200, # 超高聚类最少点云点数据
                         
                         'depth_width': 512, # 传感器深度数据信息
                         'depth_height':424,
                         }
    # 多个传感器参数
    multi_sensor_params = {
              'alarm_valid_near_time': 0.2, # 告警时间边界的前后一段时间无效
              'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
              
              'alone_stay_person_num': 1, # 单人留仓人数
              'alone_stay_sensor_group': None, # 单人留仓内外仓传感器分组序号； [默认值：None]
              
              'internal_supervisor_continuous_time': 0.5, # 内部监管持续多长时间，单位：min
              
              'conflict_person_num': 3, # 多人聚集告警最低人数
              'conflict_near_radius': 0.5, # 多人聚集半径

              }   
              
elif  cfg_case == 3:
    # 在线/离线,[ OffLine / OnLine ]
    online_offline_type = 'OnLine'
    # 保存结果
    log_info = {
                    'debug_info': '1111100000', # 打印信息: [0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4=打印各类告警检测信息]
                    
                    'save_detect_bbox_image_flag': 0, # 保存bbox目标检测图像结果
                    
                    'save_data_info': '0000100000', # 根据告警类型结果保存数据: [1=未在指定时间休息, 2=未在制定区域监督, 3=厕所区域异常, 4=窗户区域异常, 5=高度异常]
                    'save_data_size_depth': 1000, # 单位：MB， 保存depth文件大小
                    'save_data_size_ply': 1000, # 单位：MB， 保存ply文件大小
                        
                    'save_input_pts_flag': 0, # 保存点云数据
                    
                }
    # 单个传感器参数
    one_sensor_params = {
                         'detect_bbox_pose_type': 1, # 检测目标类别
                         'detect_bbox_pose_index': None, # 检测目标类别序号
                         
                         # 'ssd'
                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
                         'detect_bbox_method_name': 'ssd', # yolov3,ssd
                         'detect_bbox_model_name': 'ssd300_pose_9999_20200326', # 选择目标检测模型名称
                         'detect_bbox_input_image_size': 300, # 模型输入图像大小
                         'detect_bbox_score_thod': 0.5, # 目标检测目标框分数阈值
                         'detect_bbox_nms_thod': 0.45, # 目标检测目标框nms阈值
                         
#                         # 'yolov3'
#                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
#                         'detect_bbox_method_name': 'yolov3', # yolov3,ssd
#                         'detect_bbox_model_name': 'last', # 选择目标检测模型名称，【NewLG_OneClass_20200612】
#                         'detect_bbox_input_image_size': 416, # 模型输入图像大小
#                         'detect_bbox_score_thod': 0.3, # 目标检测目标框分数阈值
#                         'detect_bbox_nms_thod': 0.5, # 目标检测目标框nms阈值
                         
                         'alarm_valid_near_time': 5, # 告警时间边界的前后一段时间无效
                         'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
                         
                         'alarm_bed_height': 0.6, # 床位高度设置
                         
                         'detect_bbox_use_bg_info': True, # 目标检测是否读取使用信息
                         'use_bg_info': True, # 是否读取背景信息
                         'use_bg_info_height_limit': 1.8, # 高度信息更细背景,eg: [Flase/0.5]
                         'use_bg_info_outlier_dist': 0.13, # 获取边界离群点距离阈值
                         'pt_cluster_num_thod': 50, # 超高聚类最少点云点数据
                         
                         'depth_width': 512, # 传感器深度数据信息
                         'depth_height':424,
                         }
    # 多个传感器参数
    multi_sensor_params = {
              'alarm_valid_near_time': 5, # 告警时间边界的前后一段时间无效
              'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
              
              'alone_stay_person_num': 2, # 单人留仓人数
              'alone_stay_sensor_group': [[5,6,7,8],[1,2,3,4]], # 单人留仓内外仓传感器分组序号； [默认值：None]
              
              'internal_supervisor_continuous_time': 1.0, # 内部监管持续多长时间，单位：min
              
              'conflict_person_num': 3, # 多人聚集告警最低人数
              'conflict_near_radius': 0.5, # 多人聚集半径

              }  

elif  cfg_case == 4:
    # 在线/离线,[ OffLine / OnLine ]
    online_offline_type = 'OnLine'
    # 保存结果
    log_info = {
                    'debug_info': '1111100000', # 打印信息: [0 = 打印基本流程信息， 1 = 打印接口输入输出信息， 2 = 打印目标检测信息， 3 = 打印聚类检测信息， 4=打印各类告警检测信息]
                    
                    'save_detect_bbox_image_flag': 0, # 保存bbox目标检测图像结果
                    
                    'save_data_info': '0000000000', # 根据告警类型结果保存数据: [1=未在指定时间休息, 2=未在制定区域监督, 3=厕所区域异常, 4=窗户区域异常, 5=高度异常]
                    'save_data_size_depth': 500, # 单位：MB， 保存depth文件大小
                    'save_data_size_ply': 1000, # 单位：MB， 保存ply文件大小
                        
                    'save_input_pts_flag': 0, # 保存点云数据
                    
                }
    # 单个传感器参数
    one_sensor_params = {
#                         'detect_bbox_pose_type': 1, # 检测目标类别
#                         'detect_bbox_pose_index': None, # 检测目标类别序号

                         'detect_bbox_pose_type': 3, # 检测目标类别
                         'detect_bbox_pose_index': [2, 0, 1], # 检测目标类别序号，新标注信息类别序号转换
                         
#                         # 'ssd'
#                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
#                         'detect_bbox_method_name': 'ssd', # yolov3,ssd
#                         'detect_bbox_model_name': 'NewLG_OneClass_20200623', # 选择目标检测模型名称，【NewLG_OneClass_20200612】
#                         'detect_bbox_input_image_size': 300, # 模型输入图像大小
#                         'detect_bbox_score_thod': 0.5, # 目标检测目标框分数阈值
#                         'detect_bbox_nms_thod': 0.45, # 目标检测目标框nms阈值
                         
#                         # 'yolov3'
#                         'detect_bbox_image_type': 'colormap', # 'rgb'/'colormap'
#                         'detect_bbox_method_name': 'yolov3', # yolov3,ssd
#                         'detect_bbox_model_name': 'LG_Depth_20200922', # 选择目标检测模型名称，【LG_Depth_20200819】
#                         'detect_bbox_input_image_size': 416, # 模型输入图像大小
#                         'detect_bbox_score_thod': 0.3, # 目标检测目标框分数阈值
#                         'detect_bbox_nms_thod': 0.5, # 目标检测目标框nms阈值

#                         # 'yolov3'
#                         'detect_bbox_image_type': 'colormap_rgb', # 'colormap'/'colormap_rgb'
#                         'detect_bbox_method_name': 'yolov3', # yolov3,ssd
#                         'detect_bbox_model_name': ['LG_Depth_20200922', 'LG_Color_20200909'], 
#                         'detect_bbox_input_image_size': 416, # 模型输入图像大小
#                         'detect_bbox_score_thod': 0.3, # 目标检测目标框分数阈值
#                         'detect_bbox_nms_thod': 0.5, # 目标检测目标框nms阈值

#                         # 'yolov3', 20201117
#                         'detect_bbox_image_type': 'colormap_rgb', # 'colormap'/'colormap_rgb'
#                         'detect_bbox_method_name': 'yolov3', # yolov3,ssd
#                         'detect_bbox_model_name': ['LG_Depth_20201117', 'LG_Color_20201111'], 
#                         'detect_bbox_input_image_size': 416, # 模型输入图像大小
#                         'detect_bbox_score_thod': 0.5, # 目标检测目标框分数阈值
#                         'detect_bbox_nms_thod': 0.5, # 目标检测目标框nms阈值
                         
                         # 'yolov3', 20201117
                         'detect_bbox_image_type': 'colormap', # 'colormap'/'colormap_rgb'
                         'detect_bbox_method_name': 'yolov3', # yolov3,ssd
                         'detect_bbox_model_name': 'LG_Depth_20210331', # LG_Depth_20200922 / LG_Depth_20201214 / LG_Depth_20210331
                         'detect_bbox_input_image_size': 416, # 模型输入图像大小
                         'detect_bbox_score_thod': 0.6, # 目标检测目标框分数阈值
                         'detect_bbox_nms_thod': 0.5, # 目标检测目标框nms阈值
                         
                         'alarm_valid_near_time': 5, # 告警时间边界的前后一段时间无效
                         'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
                         
                         'alarm_bed_height': 0.5, # 床位高度设置
                         
                         'detect_bbox_use_bg_info': False, # 目标检测是否读取使用信息
                         'use_bg_info': True, # 是否读取背景信息
                         'use_bg_info_height_limit': 1.7, # 高度信息更细背景,eg: [Flase/0.5]，init：1.8
                         'use_bg_info_outlier_dist': 0.13, # 获取边界离群点距离阈值
                         'pt_cluster_num_thod': 50, # 超高聚类最少点云点数据
                         
                         'depth_width': 512, # 传感器深度数据信息
                         'depth_height':424,
                         }
    # 多个传感器参数
    multi_sensor_params = {
              'alarm_valid_near_time': 5, # 告警时间边界的前后一段时间无效
              'alarm_valid_frame_num': 5, # 连续多少帧开始触发告警状态
              
              'alone_stay_person_num': 1, # 单人留仓人数
              'alone_stay_sensor_group': [[5,6,7,8],[1,2,3,4]], # 单人留仓内外仓传感器分组序号； [默认值：None]
              
              'internal_supervisor_continuous_time': 0.3, # 内部监管持续多长时间，单位：min
              
              'conflict_person_num': 3, # 多人聚集告警最低人数
              'conflict_near_radius': 0.5, # 多人聚集半径

              }              
              

# 不同告警等级参数设置
#   等级1：严格
#   等级2：正常
#   等级3：不严格
alarm_level = dict()
alarm_level_first = {
                    # 多个功能通用参数设置
                    'use_bg_info_outlier_dist': 0.13, # 获取边界离群点距离阈值，单位：米
                    
                    # 单人留仓
                    'alone_continuous_frame_num': 3, # 单人留仓 连续多帧有效
                    'alone_continuous_time': 0, # 单人留仓 超时时长，单位：分钟
                    # 超高
                    'above_height_continuous_frame_num': 2, # 超高 连续多帧有效
                    'above_height_edge_inner_dist': 0.15, # 超高 边界往内部一定距离，单位：米
                    'above_height_cluster_pts_num': 40, # 超高 目标最少点数
                    'above_height_cluster_distribute_select': 1, # 超高 目标点云分布参数设置
                    'above_height_using_human_dete': False, # 结合人员检测信息判断超高有效性
                    # 内部监管
                    'internal_supervisor_continuous_time': 0.3, # 内部监管 超时时长，单位：分钟
                    # 厕所区域异常
                    'toilet_continuous_time': 0.3, # 厕所区域 超时时长，单位：分钟
                    }
alarm_level_second = {
                    # 多个功能通用参数设置
                    'use_bg_info_outlier_dist': 0.16, # 获取边界离群点距离阈值，单位：米
                    
                    # 单人留仓
                    'alone_continuous_frame_num': 6, # 单人留仓 连续多帧有效
                    'alone_continuous_time': 0.25, # 单人留仓 超时时长，单位：分钟
                    # 超高
                    'above_height_continuous_frame_num': 6, # 超高 连续多帧有效
                    'above_height_edge_inner_dist': 0.2, # 超高 边界往内部一定距离，单位：米
                    'above_height_cluster_pts_num': 150, # 超高 目标最少点数
                    'above_height_cluster_distribute_select': 2, # 超高 目标点云分布参数设置
                    'above_height_using_human_dete': True, # 结合人员检测信息判断超高有效性
                    # 内部监管
                    'internal_supervisor_continuous_time': 1, # 内部监管 超时时长，单位：分钟
                    # 厕所区域异常
                    'toilet_continuous_time': 5, # 厕所区域 超时时长，单位：分钟
                    }
alarm_level_third = {
                    # 多个功能通用参数设置
                    'use_bg_info_outlier_dist': 0.2, # 获取边界离群点距离阈值，单位：米
                    
                    # 单人留仓
                    'alone_continuous_frame_num': 10, # 单人留仓 连续多帧有效
                    'alone_continuous_time': 0.5, # 单人留仓 超时时长，单位：分钟
                    # 超高
                    'above_height_continuous_frame_num': 15, # 超高 连续多帧有效
                    'above_height_edge_inner_dist': 0.25, # 超高 边界往内部一定距离，单位：米
                    'above_height_cluster_pts_num': 250, # 超高 目标最少点数
                    'above_height_cluster_distribute_select': 3, # 超高 目标点云分布参数设置
                    'above_height_using_human_dete': True, # 结合人员检测信息判断超高有效性
                    # 内部监管
                    'internal_supervisor_continuous_time': 2, # 内部监管 超时时长，单位：分钟
                    # 厕所区域异常
                    'toilet_continuous_time': 10, # 厕所区域 超时时长，单位：分钟
                    }
alarm_level = { # 告警等级总和
               '1': alarm_level_first, 
               '2': alarm_level_second, 
               '3': alarm_level_third, 
               }

            