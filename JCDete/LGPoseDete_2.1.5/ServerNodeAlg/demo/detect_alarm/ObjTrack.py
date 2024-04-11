# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:49:53 2020

@author: HYD
"""

import numpy as np

def TwoFrameObjMatch(PreFrameInfo, CurFrameInfo, MatchDistThod):
    """
    功能：前后两帧匹配关系
    输入：
        PreFrameInfo：前一帧信息,[N1 x 3]
        CurFrameInfo：当前帧信息,[N2 x 3]
        MatchDistThod：前后帧目标匹配距离
    """
    PreObjNum = PreFrameInfo.shape[0]
    CurObjNum = CurFrameInfo.shape[0]
    # 前后帧目标点相互距离
    DistMatrix = np.ones([PreObjNum, CurObjNum])
    for i_pred in range(PreObjNum):
        for i_cur in range(CurObjNum):
            DistMatrix[i_pred, i_cur] = np.sqrt((PreFrameInfo[i_pred,0] - CurFrameInfo[i_cur,0])**2 \
                                                + (PreFrameInfo[i_pred,1] - CurFrameInfo[i_cur,1])**2)
#    print('DistMatrix = ', DistMatrix)
    # 初始化当前帧数匹配关系
    PreMatchFlag = PreObjNum*[-1]
    CurMatchFlag = CurObjNum*[-1]
    for i_obj in range(CurObjNum):
        PreObjDistThod = DistMatrix[:,i_obj][DistMatrix[:,i_obj]<MatchDistThod]
        if PreObjDistThod.shape[0]>0: # 存在小于阈值目标
            PreObjValueMin = min(PreObjDistThod)
            PreObjIndex = np.where(DistMatrix[:,i_obj] == PreObjValueMin)[0][0]
            if PreMatchFlag[PreObjIndex] == -1: # 当前目标未被匹配
                PreMatchFlag[PreObjIndex] = i_obj
                CurMatchFlag[i_obj] = PreObjIndex
    #            print('PreMatchFlag = ', PreMatchFlag)
    #            print('CurMatchFlag = ', CurMatchFlag)
    return PreMatchFlag, CurMatchFlag

if __name__ == '__main__':
    print('Start.')
    TestTwoFrameObjMatchFlag = 1 # 测试前后两帧匹配关系
    
    if TestTwoFrameObjMatchFlag == 1:
        PreFrameInfo = np.random.random([5,3])
        CurFrameInfo = PreFrameInfo[1:,:]
#        PreFrameInfo = PreFrameInfo[2:,:]
        
        MatchDistThod = 0.25
        PreMatchFlag, CurMatchFlag = TwoFrameObjMatch(PreFrameInfo, CurFrameInfo, MatchDistThod)
        print('MatchFlag = ', PreMatchFlag, CurMatchFlag)
    
    print('End.')