# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 17:36:38 2020

@author: HYD
"""

def TransBboxToPoints(Bbox):
    """
    功能：将 bbox 信息转换为多点信息
    输入：
        Bbox: [xmin, xmax, ymin, ymax]
    输出：
        CurPts：[x1,y1,x2,y2,x3,y3,x4,y4,...]
    """
    CurPts = []
    # xmin, xmax, ymin, ymax
    xmin = Bbox[0]
    xmax = Bbox[1] 
    ymin = Bbox[2]
    ymax = Bbox[3]
    # 4-points, 顺时针
    CurPts.append(xmin) # 1-pt
    CurPts.append(ymin)
    CurPts.append(xmin) # 2-pt
    CurPts.append(ymax)
    CurPts.append(xmax) # 3-pt
    CurPts.append(ymax)
    CurPts.append(xmax) # 4-pt
    CurPts.append(ymin) 
    
    return CurPts
    
    
def TransPointsToBbox(BboxPts):
    """
    功能：将 多点信息 边界信息转换为 bbox
    输入：
        CurPts：[x1,y1,x2,y2,x3,y3,x4,y4,...]
    输出：
        Bbox: [xmin, xmax, ymin, ymax]
    """
    Bbox = []
    # multi-points
    CurOneArea = BboxPts.reshape([int((BboxPts.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
    CurOneArea_xmin = min(CurOneArea[:,0])
    CurOneArea_xmax = max(CurOneArea[:,0])
    CurOneArea_ymin = min(CurOneArea[:,1])
    CurOneArea_ymax = max(CurOneArea[:,1])
    # xmin, xmax, ymin, ymax
    Bbox = [CurOneArea_xmin, CurOneArea_xmax, CurOneArea_ymin, CurOneArea_ymax]
    
    return Bbox
    
if __name__ == '__main__':
    print('Start.')
    TestTransBboxToPointsFlag = 1 # 测试 TransBboxToPoints
    
    if TestTransBboxToPointsFlag == 1:
        CurBbox = [100,200,300,400] # [xmin, xmax, ymin, ymax]
        CurPts = TransBboxToPoints(CurBbox)
        print('CurPts = {}'.format(CurPts))
    
    print('End.')