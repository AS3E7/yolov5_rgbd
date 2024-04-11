# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 09:19:13 2019

@author: Xiaowei SHAO

Common Algorithm Lib - Point Cloud(cal_pc)
"""

import numpy as np

def ind2sub(size, index):
    """
      size: [M, N], dimension
      index: 整型list或np.array，值域[0, M * N - 1]
    """ 
    
    N = size[1]
    
    i = np.array(index) // N
    j = np.array(index) %  N
    
    return i, j

def match_by_distmat(dist_mat, dist_thresh):
    '''
    dist_mat: ndarray, M(truth) x N(pred)
    匹配准则：
    1） 单一匹配，可以无匹配。不存在一对多或多对一映射
    2） dist_mat矩阵元素<=dist_thresh时则可存在映射的可能性
    3） 按最小距离开始建立匹配关系，依次迭代
    '''
    
    if(len(dist_mat.shape) != 2):
        print('Warning in match_by_distmat: dist_mat must be a 2D numpy array')
        
    M = dist_mat.shape[0]
    N = dist_mat.shape[1]

    if( (M == 0) | (N == 0)):
        return [0, 0, 0]
    
    # 排序 
    dist_vec = dist_mat.reshape([-1])
    sort_idx = np.argsort(dist_vec) 
    
    pos = 0
    match_tbl  = np.ones([M]) * -1
    match_tbl2 = np.ones([N]) * -1
    while(True):        
        
        if(dist_vec[sort_idx[pos]] > dist_thresh):
            break
        
        i, j = ind2sub([M, N], sort_idx[pos])
        if((match_tbl[i] < 0) & (match_tbl2[j] < 0)):
            # match found            
            match_tbl[i] = j 
            match_tbl2[j] = i
        
        pos += 1
        if(pos >= (M * N)):
            break
        
    #print(match_tbl)    
    #print(match_tbl2)   
    
    n1 = len(match_tbl[match_tbl >= 0])     # 正
    n2 = len(match_tbl[match_tbl  < 0])     # 漏
    n3 = len(match_tbl2[match_tbl2 < 0])    # 误
    
    return [n1, n2, n3]    
    
def match_point_result(pt_truth, pt_pred, match_dist):
    """
    input:
      pt_truth: M x D, numpy array
      pt_pred:  N x D, numpy array
      match_dist:欧式距离下的邻域范围
      D值域：{1, 2, 3}
    output:  
      匹配结果。输出三元组[正确匹配数，漏报数，误报数]
      其中 M = 正确匹配数 + 漏报数, N = 正确匹配数 + 误报数            
    匹配准则：
      1）最多在对方集合中匹配一个目标；可以无匹配
      2）距离 <= match_dist则存在匹配可能性
      3）按距离最小值开始逐一确定匹配关系。
    """
    
    M = pt_truth.shape[0]
    D = pt_truth.shape[1]    
    N = pt_pred.shape[0]
    if(D != pt_pred.shape[1]):
        print('Error in match_point_result: dimension of pt_truth and pt_pred must be equal.')
        return None
    
    if(M == 0):
        return [0, 0, N]         # 正/漏/误
    
    if(N == 0):
        return [0, M, 0]         # 正/漏/误
    
    # step 1: 计算距离矩阵（欧式距离）
    tmp1 = pt_truth.reshape(M, 1, D)
    tmp2 =  pt_pred.reshape(1, N, D)
    
    mat_m = np.ones([M, 1])
    mat_n = np.ones([1, N])    
    dist_mat = np.zeros([M, N])
    for d in range(D):        
        dist_mat += (tmp1[:, :, d] * mat_n - mat_m * tmp2[:, :, d]) ** 2
    dist_mat = dist_mat ** 0.5
    
    return match_by_distmat(dist_mat, match_dist)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
           box: ndarray, 1 x 4 (x1, y1, x2, y2)
         boxes: ndarray, N x 4 (x1, y1, x2, y2)
      box_area: area of 'box'
    boxes_area: area of 'boxes'

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[2], boxes[:, 2])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_ious(boxes1, boxes2):
    """Computes IoU between two sets of boxes.
    boxes1: ndarray, M x 4 (x1, y1, x2, y2)
    boxes2: ndarray, N x 4 (x1, y1, x2, y2)
      ious: ndarray, M x N
    """
    # Compute areas of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    ious = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(ious.shape[1]):
        box2 = boxes2[i, :]
        ious[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return ious

def match_ious_result(ious, iou_thresh):
    # 因为dist是距离越小匹配度越高，因此做相应转换
    dist_mat = 1.0 - ious
    dist_thresh = 1.0 - iou_thresh
    return match_by_distmat(dist_mat, dist_thresh)

if __name__ == '__main__':
    
    TestCase = 2

    if TestCase == 1:
        pt_truth = np.array([[0, 0], 
                             [1, 1], 
                             [2, 2]])
        pt_pred  = np.array([[3.0, 1.0],
                             [0.0, 0.0], 
                             [0.0, 0.0], 
                             [0.6, 1.5], 
                             [1.8, 1.3]])     
        match_dist = 5.0
        match_res = match_point_result(pt_truth, pt_pred, match_dist)  
        print(match_res)
        
    elif TestCase == 2: # LG-bbox-predict
        import os
        from shutil import copyfile
        from CalcAP import ReadBboxFileInfo
        
        ########### Depth ###########
#        # model name
#        TestModelName = '20200612_NewLG_Depth_resume-2.3' # LG- MODEL
#        TestModelTrain = 'train' # 'train'/'val'
#        TestModelTime = '1592530304'
#        TestModelTrain = 'val' # 'train'/'val'
#        TestModelTime = '1592529680'

#        # model name
#        TestModelName = '20200623_NewLG_Depth_resume-2.1' # LG- MODEL
##        TestModelTrain = 'train' # 'train'/'val'
##        TestModelTime = '1593482322'
#        TestModelTrain = 'val' # 'train'/'val'
#        TestModelTime = '1593485350'

#        # model name
#        TestModelName = '20200612_NewLG_Depth_resume-2.3' # LG- MODEL
#        TestModelTrain = 'train' # 'train'/'val'
#        TestModelTime = '1593487301'
##        TestModelTrain = 'val' # 'train'/'val'
##        TestModelTime = '1593485687'

#        # model name
#        TestModelName = '20200621_NewLG_Depth_resume-2.0' # LG- MODEL
##        TestModelTrain = 'train' # 'train'/'val'
##        TestModelTime = '1593482322'
#        TestModelTrain = 'val' # 'train'/'val'
#        TestModelTime = '1593486479'

        ########### RGB ###########
        # model name
#        TestModelName = '20200612_NewLG_RGB_init-2.1' # LG- MODEL
#        TestModelTrain = 'train' # 'train'/'val'
#        TestModelTime = '1592547524'
#        TestModelTrain = 'val' # 'train'/'val'
#        TestModelTime = '1592548817'

        TestModelName = '20200612_NewLG_RGB_init-2.0' # LG- MODEL
        TestModelTrain = 'train' # 'train'/'val'
        TestModelTime = '1593513543'
#        TestModelTrain = 'val' # 'train'/'val'
#        TestModelTime = '1593514389'
        
        SaveFolderSrc = r'D:\xiongbiao\HYD\Code\SolitaryCellDetect\Code\DetectPyCode\LGPoseDete\Code\NASPoseDetect_OneClass\result'
        SaveFolderName = os.path.join(SaveFolderSrc, TestModelName + '_' + TestModelTrain)
        
        # bbox file full name
        predict_file = os.path.join(SaveFolderName, 'eval_result_colormap_' + TestModelName + '_' + TestModelTrain + '_' + TestModelTime + '.csv')
        ground_truth_file = os.path.join(SaveFolderName, 'ImageLabel_colormap_' + TestModelName + '_' + TestModelTrain + '_' + TestModelTime + '.csv')
        
        # read file name
        ground_truth_info = ReadBboxFileInfo(ground_truth_file)
        pred_info = ReadBboxFileInfo(predict_file)
#        print('ground_truth_info = ', ground_truth_info)
#        print('pred_info = ', pred_info)

        # save result
        SavePredErrorResultFolderName = os.path.join(SaveFolderName, 'ErrorPred')
        if not os.path.exists(SavePredErrorResultFolderName):
            os.mkdir(SavePredErrorResultFolderName)
        
        # calculate bbox ious
        iou_match_res = np.zeros([3]) # objects ious
        image_iou_match_res = np.zeros([3]) # image ious
        for cur_file_name in ground_truth_info:
            print(cur_file_name)
            # bboxes
            boxes_tr = ground_truth_info[cur_file_name] # (x1, y1, x2, y2)
            try:
                boxes_pred = pred_info[cur_file_name] # (x1, y1, x2, y2)
            except:
                boxes_pred = np.zeros([1,4])

            boxes_tr = np.array(boxes_tr)
            boxes_pred = np.array(boxes_pred)
#            print(boxes_tr, boxes_pred)
            # 去除无效的目标框
            boxes_tr_new = []
            for i_boxes_tr in range(boxes_tr.shape[0]):
                if (boxes_tr[i_boxes_tr][0]==0 and boxes_tr[i_boxes_tr][2]==0) or (boxes_tr[i_boxes_tr][0]==512 and boxes_tr[i_boxes_tr][2]==512):
                    print('bbox error')
                else:
                    boxes_tr_new.append(boxes_tr[i_boxes_tr])
            boxes_tr = np.array(boxes_tr_new)
            
            
            # objects ious
            ious = compute_ious(boxes_tr, boxes_pred)
            iou_thresh = 0.35 # 0.35/0.5/0.45
            curr_match_res = match_ious_result(ious, iou_thresh)
            iou_match_res += curr_match_res
            # image ious
            image_curr_match_res = match_ious_result(ious, iou_thresh)
            if image_curr_match_res[1] == 0 and image_curr_match_res[2] == 0:
                image_curr_match_res = [1,0,0]
            else:
                if image_curr_match_res[1] > image_curr_match_res[2]:
                    image_curr_match_res = [0,1,0]
                else:
                    image_curr_match_res = [0,0,1]
                # save error pred result
                copyfile(os.path.join(SaveFolderName, 'DeteResult', cur_file_name+'.png'), os.path.join(SavePredErrorResultFolderName, cur_file_name+'.png'))
            # add
            image_iou_match_res += image_curr_match_res
        # print result
        if len(iou_match_res) > 0: # objects ious
            print('iou_match_res = ', iou_match_res)
            print('  overall precision = %.6f'%(iou_match_res[0] / (np.sum(iou_match_res[[0, 1]]))))
            print('             recall = %.6f'%(iou_match_res[0] / (np.sum(iou_match_res[[0, 2]]))))
            print('           accuracy = %.6f'%(iou_match_res[0] / (np.sum(iou_match_res))))
        if len(image_iou_match_res) > 0: # image ious
            print('image_iou_match_res = ', image_iou_match_res)
            print('  overall precision = %.6f'%(image_iou_match_res[0] / (np.sum(image_iou_match_res[[0, 1]]))))
            print('             recall = %.6f'%(image_iou_match_res[0] / (np.sum(image_iou_match_res[[0, 2]]))))
            print('           accuracy = %.6f'%(image_iou_match_res[0] / (np.sum(image_iou_match_res))))        

            
        
        
        
        
        
