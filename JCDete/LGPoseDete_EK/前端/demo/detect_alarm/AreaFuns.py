# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:20:00 2020

@author: HYD
"""

import numpy as np

def CalIntersectAreaFromTwoArea(Area1, Area2):
    """
    功能：计算两个区域的相交区域
        输入：[xmin, xmax, ymin, ymax]
        输出：[xmin, xmax, ymin, ymax]
    """
    AreaShapeType = 1 # AreaShapeType == 1,两个矩形区域，输入数据为 [xmin, xmax, ymin, ymax]
                      # AreaShapeType == 2,两个任意多边形区域。输入数据为点数据形式
    OverlapArea = []
    
    if AreaShapeType == 1: # 两个矩形区域，输入数据为 [xmin, xmax, ymin, ymax]
        Rect1 = Area1
        Rect2 = Area2
        (x11,x12,y11,y12)=Rect1 #矩形1左上角(x11,y11)和右下角(x12,y12) 
        (x21,x22,y21,y22)=Rect2 #矩形2左上角(x21,y21)和右下角(x22,y22)

        StartX=min(x11,x21) #外包框在x轴上左边界
        EndX=max(x12,x22)  #外包框在x轴上右边界
        StartY=min(y11,y21)#外包框在y轴上上边界
        EndY=max(y12,y22)#外包框在y轴上下边界

        CurWidth=(x12-x11)+(x22-x21)-(EndX-StartX)#(EndX-StartX)表示外包框的宽度
        CurHeight=(y12-y11)+(y22-y21)-(EndY-StartY)#(Endy-Starty)表示外包框的高度
        
        if CurWidth<=0 or CurHeight<=0:#不相交		
            OverlapArea = []	
        else:#相交		
            X1=max(x11,x21)#有相交则相交区域位置为：小中取大为左上角，大中取小为右下角		
            Y1=max(y11,y21)		
            X2=min(x12,x22)		
            Y2=min(y12,y22)
            OverlapArea = [X1, X2, Y1, Y2]

    return OverlapArea

def CalMultiAreaEdge(MultiArea):
    """
    功能：计算多区域边界区域
        输入：[x1, y1, x2, y2, x3, y3, x4, y4, ...]
        输出：[xmin, xmax, ymin, ymax]
    """

    # 计算方法
    CalAreaEdgeMethodFlag = 1 # 将多边形区域转换成矩形进行计算
    
    # 边界区域
    DestMultiAreaEdge = []  # [xmin, xmax, ymin, ymax]
            
    if CalAreaEdgeMethodFlag == 1:
        # 1, 先将多边形区域转换为矩形区域标识，[xmin, xmax, ymin, ymax]
        SrcMultiArea = []
        for i in range(len(MultiArea)):
            CurOneArea = np.array(MultiArea[i])
            CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
            CurOneArea_xmin = min(CurOneArea[:,0])
            CurOneArea_xmax = max(CurOneArea[:,0])
            CurOneArea_ymin = min(CurOneArea[:,1])
            CurOneArea_ymax = max(CurOneArea[:,1])
            SrcMultiArea.append([CurOneArea_xmin, CurOneArea_xmax, CurOneArea_ymin, CurOneArea_ymax])
    
        # 2,再计算矩形之间的相互边界
        TwoAreaEdgeDisMax = 0.1 # 两个区域边界的最远距离
        for i in range(len(SrcMultiArea)):
            CurOneArea_i = np.array(SrcMultiArea[i]) # [xmin, xmax, ymin, ymax]
            for j in range(i+1, len(SrcMultiArea), 1):
#                print('i, j', i, j)
                CurOneArea_j = np.array(SrcMultiArea[j]) # # [xmin, xmax, ymin, ymax]
                # overlap
                X_Overlap = (min(CurOneArea_i[1], CurOneArea_j[1]) - max(CurOneArea_i[0], CurOneArea_j[0]))/(max(CurOneArea_i[1], CurOneArea_j[1]) - min(CurOneArea_i[0], CurOneArea_j[0]))
                Y_Overlap = (min(CurOneArea_i[3], CurOneArea_j[3]) - max(CurOneArea_i[2], CurOneArea_j[2]))/(max(CurOneArea_i[3], CurOneArea_j[3]) - min(CurOneArea_i[2], CurOneArea_j[2]))
                # left
                if (abs(CurOneArea_i[0] - CurOneArea_j[1]) < TwoAreaEdgeDisMax) and (Y_Overlap>0.5):
                    DestMultiAreaEdge.append([(CurOneArea_i[0]+CurOneArea_j[1])/2, (CurOneArea_i[0]+CurOneArea_j[1])/2, max(CurOneArea_i[2], CurOneArea_j[2]), min(CurOneArea_i[3], CurOneArea_j[3])])
                # right
                if (abs(CurOneArea_i[1] - CurOneArea_j[0]) < TwoAreaEdgeDisMax) and (Y_Overlap>0.5):
                    DestMultiAreaEdge.append([(CurOneArea_i[1]+CurOneArea_j[0])/2, (CurOneArea_i[1]+CurOneArea_j[0])/2, max(CurOneArea_i[2], CurOneArea_j[2]), min(CurOneArea_i[3], CurOneArea_j[3])])
                # down
                if (abs(CurOneArea_i[2] - CurOneArea_j[3]) < TwoAreaEdgeDisMax) and (X_Overlap>0.5):
                    DestMultiAreaEdge.append([max(CurOneArea_i[0], CurOneArea_j[0]), min(CurOneArea_i[1], CurOneArea_j[1]), (CurOneArea_i[2]+CurOneArea_j[3])/2, (CurOneArea_i[2]+CurOneArea_j[3])/2])
                # up
                if (abs(CurOneArea_i[3] - CurOneArea_j[2]) < TwoAreaEdgeDisMax) and (X_Overlap>0.5):
                    DestMultiAreaEdge.append([max(CurOneArea_i[0], CurOneArea_j[0]), min(CurOneArea_i[1], CurOneArea_j[1]), (CurOneArea_i[3]+CurOneArea_j[2])/2, (CurOneArea_i[3]+CurOneArea_j[2])/2])

    
    return DestMultiAreaEdge
    

def CalAboveHeightValidArea(SensorDeteArea, MultiAreaEdge, DeteAreaInnerDis = 0.1):
    """
    功能：计算超高行为的有效区域
    
    """
    AboveHeightValidArea = []
    # 扩展边界区域
    MultiAreaEdgeValidArea = []
    for i in range(len(MultiAreaEdge)):
        CurMultiAreaEdge = MultiAreaEdge[i]
        # 边界区域
        CurMultiAreaEdgeRect = CurMultiAreaEdge
        if abs(CurMultiAreaEdge[0] - CurMultiAreaEdge[1]) < 0.0001:
            CurMultiAreaEdgeRect = [CurMultiAreaEdge[0]-DeteAreaInnerDis, CurMultiAreaEdge[1]+DeteAreaInnerDis, CurMultiAreaEdge[2], CurMultiAreaEdge[3]]
        elif abs(CurMultiAreaEdge[2] - CurMultiAreaEdge[3]) < 0.0001:
            CurMultiAreaEdgeRect = [CurMultiAreaEdge[0], CurMultiAreaEdge[1], CurMultiAreaEdge[2]-DeteAreaInnerDis, CurMultiAreaEdge[3]+DeteAreaInnerDis]
        # 边界区域与原始检测区域并集
        Rect1 = SensorDeteArea
        Rect2 = CurMultiAreaEdgeRect
        (x11,x12,y11,y12)=Rect1 #矩形1左上角(x11,y11)和右下角(x12,y12) 
        (x21,x22,y21,y22)=Rect2 #矩形2左上角(x21,y21)和右下角(x22,y22)

        StartX=min(x11,x21) #外包框在x轴上左边界
        EndX=max(x12,x22)  #外包框在x轴上右边界
        StartY=min(y11,y21)#外包框在y轴上上边界
        EndY=max(y12,y22)#外包框在y轴上下边界

        CurWidth=(x12-x11)+(x22-x21)-(EndX-StartX)#(EndX-StartX)表示外包框的宽度
        CurHeight=(y12-y11)+(y22-y21)-(EndY-StartY)#(Endy-Starty)表示外包框的高度
        if CurWidth<=0.0001 or CurHeight<=0.0001:#不相交		
            OverlapArea = []	
        else:#相交	
            if (CurWidth/(x12-x11)<0.5) and (CurHeight/(y12-y11)<0.5): # 相交区域某一边界需要大于一定比例
                continue
            X1=max(x11,x21)#有相交则相交区域位置为：小中取大为左上角，大中取小为右下角
            Y1=max(y11,y21)
            X2=min(x12,x22)
            Y2=min(y12,y22)
            if (X2-X1) > (Y2-Y1):
                CurMultiAreaEdgeRectNew = [X1, X2, CurMultiAreaEdgeRect[2], CurMultiAreaEdgeRect[3]]
            else:
                CurMultiAreaEdgeRectNew = [CurMultiAreaEdgeRect[0], CurMultiAreaEdgeRect[1], Y1, Y2]
            MultiAreaEdgeValidArea.append(CurMultiAreaEdgeRectNew)
    # 合并多个边界区域
    MultiAreaEdgeValidArea.append(SensorDeteArea)
    MultiAreaEdgeValidArea = np.array(MultiAreaEdgeValidArea)
    AboveHeightValidArea = [np.min(MultiAreaEdgeValidArea[:,0]), np.max(MultiAreaEdgeValidArea[:,1]), np.min(MultiAreaEdgeValidArea[:,2]), np.max(MultiAreaEdgeValidArea[:,3])] # [xmin, xmax, ymin, ymax]
    AboveHeightValidArea = AboveHeightValidArea  
    
    return AboveHeightValidArea


if __name__ == '__main__':
    print('Start.')
    TestCalIntersectAreaFromTwoAreaFlag = 1 # 计算两个区域的相交区域
    TestCalMultiAreaEdgeFlag = 0 # 计算多区域边界区域
    TestCalAboveHeightAreaFlag = 0 # 计算超高区域的检测范围
    
    TestPlotAllAlarmAreaFlag = 0 # 显示所有告警区域范围
    
    import matplotlib.pyplot as plt
    
    if TestCalIntersectAreaFromTwoAreaFlag == 1:
#        Area1 = np.array([0,0,2.5,2.5])
#        Area2 = np.array([2,2,20,20])

#        Area1 = np.array([0,2.5,0,2.5])
#        Area2 = np.array([2,20,2,20])
        
        Area1 = np.array([0,2.5,0,2.5]) # [xmin, xmax, ymin, ymax]
        Area2 = np.array([0.5,20,0.5,20])
        
        OverlapArea = CalIntersectAreaFromTwoArea(Area1, Area2)
        print('OverlapArea = {}'.format(OverlapArea))
        
        
    if TestCalMultiAreaEdgeFlag == 1:
        PlotResultFlag = 1 # 是否显示结果
        SelectAreaFlag = 2 # NJ/ZT
        
        if SelectAreaFlag == 1: # NJ
            Area1 = [5.3315,  -2.6000,  8.0000,  -2.6000,  8.0000,  -5.8000,  5.3315,  -5.8000]
            Area2 = [2.7330,  -2.6000,  5.3315,  -2.6000,  5.3315,  -5.8000,  2.7330,  -2.6000]
            Area3 = [0.0000,  -3.3000,  1.8000,  -2.6000,  2.7330,  -2.6000,  2.7330,  -5.8000,  0.0000, -5.8000]
            Area4 = [5.3315,  0.0000,  8.0000,   0.0000,   8.0000,  -2.3000,  5.3315,  -2.3000]
            Area5 = [0.0000,  0.0000,  2.7330,   0.0000,   2.7330,  -2.3000,  1.8000,  -2.3000,  0.0000,  -1.8000]
            Area6 = [2.7330,  0.0000,  5.3315,   0.0000,   5.3315,  -2.3000,  2.7330,  -2.3000]
            MultiArea = [Area1, Area2, Area3, Area4, Area5, Area6]
        elif SelectAreaFlag == 2: # ZT
            import sys
            sys.path.append('..')
            from util.PolygonFuns import TransBboxToPoints
        
#            Area1 = TransBboxToPoints([2.456, 4.348, 0.078, 3.562]) # [xmin, xmax, ymin, ymax]
#            Area2 = TransBboxToPoints([-0.541, 0.147, 0.666, 1.67])
#            Area3 = TransBboxToPoints([0.635, 2.456, 1.569, 3.032])
#            Area4 = TransBboxToPoints([2.463, 4.341, 1.562, 3.047])
#            Area5 = TransBboxToPoints([0.628, 2.456, 0.078, 1.569])
#            MultiArea = [Area1, Area2, Area3, Area4, Area5]

#            Area1 = TransBboxToPoints([2.396, 4.28, 0.007, 1.497]) # [xmin, xmax, ymin, ymax]
#            Area2 = TransBboxToPoints([-0.604, 0.07, 0.604, 1.596])
#            Area3 = TransBboxToPoints([0.569, 2.396, 1.497, 2.959])
#            Area4 = TransBboxToPoints([2.403, 4.28, 1.497, 2.973])
#            Area5 = TransBboxToPoints([0.555, 2.396, 0.007, 1.497])
#            MultiArea = [Area1, Area2, Area3, Area4, Area5]

            Area1 = TransBboxToPoints([2.396, 4.28, 0.007, 1.497]) # [xmin, xmax, ymin, ymax]
            Area2 = TransBboxToPoints([-0.604, 0.07, 0.604, 1.596])
            Area3 = TransBboxToPoints([0.569, 2.396, 1.497, 2.959])
            Area4 = TransBboxToPoints([2.353, 4.28, 1.537, 3.973])
            Area5 = TransBboxToPoints([0.555, 2.396, 0.007, 1.497])
            MultiArea = [Area1, Area2, Area3, Area4, Area5]
        
        
        DestMultiAreaEdge = CalMultiAreaEdge(MultiArea)
        print('DestMultiAreaEdge = ', DestMultiAreaEdge)
        
        # plot result
        if PlotResultFlag == 1:
            fig = plt.figure()
            plt.clf()
            plt.subplot(111)
            ax = plt.gca()
            # 画原始框
            for i in range(len(MultiArea)):
                CurOneArea = np.array(MultiArea[i])
                CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                CurOneArea_xmin = min(CurOneArea[:,0])
                CurOneArea_xmax = max(CurOneArea[:,0])
                CurOneArea_ymin = min(CurOneArea[:,1])
                CurOneArea_ymax = max(CurOneArea[:,1])
                print(i, ' = ', CurOneArea_xmin, CurOneArea_xmax, CurOneArea_ymin, CurOneArea_ymax)
                rect = plt.Rectangle((CurOneArea_xmin,CurOneArea_ymin),CurOneArea_xmax-CurOneArea_xmin,CurOneArea_ymax-CurOneArea_ymin, fill=False, edgecolor='r', linewidth=3)
                ax.add_patch(rect)
                
            # 画边界框
            for i in range(len(DestMultiAreaEdge)):
                CurOneArea = np.array(DestMultiAreaEdge[i])
                CurOneArea_xmin = CurOneArea[0]
                CurOneArea_xmax = CurOneArea[1]
                CurOneArea_ymin = CurOneArea[2]
                CurOneArea_ymax = CurOneArea[3]
                print(i, ' = ', CurOneArea_xmin, CurOneArea_xmax, CurOneArea_ymin, CurOneArea_ymax)
                rect = plt.Rectangle((CurOneArea_xmin,CurOneArea_ymin),CurOneArea_xmax-CurOneArea_xmin,CurOneArea_ymax-CurOneArea_ymin, fill=False, edgecolor='g', linewidth=3)
                ax.add_patch(rect)
            
            plt.axis('equal')
            plt.show()
            plt.savefig('area.png')
            
            
    if TestCalAboveHeightAreaFlag == 1: # 计算超高区域的检测范围
        PlotResultFlag = 1 # 显示结果
        SelectAboveHeightAreaFlag = 1 # ZT 超高区域
        SelectAreaFlag = 2 # ZT 多传感器检测区域
        
        if SelectAboveHeightAreaFlag == 1: # ZT 超高区域
            DeteAlarmArea = [] # [xmin, xmax, ymin, ymax]
            DeteAlarmArea = [0.576, 4.182, 0.175, 2.966]
        
        if SelectAreaFlag == 2: # ZT 多传感器检测区域
            # 服务器中配置文件
            Area1 = [2.3960, 0.0070, 4.2800, 0.0070, 4.2800, 1.4970, 2.3960, 1.4970]
            Area2 = [-0.6040, 0.6040, 0.0700, 0.6040, 0.0700, 1.5960, -0.6040, 1.5960]
            Area3 = [0.5690, 1.4970, 2.3960, 1.4970, 2.3960, 2.9590, 0.5690, 2.9590]
            Area4 = [2.4030, 1.4970, 4.2800, 1.4970, 4.2800, 2.9730, 2.4030, 2.9730]
            Area5 = [0.5550, 0.0070, 2.3960, 0.0070, 2.3960, 1.4970, 0.5550, 1.4970]
            MultiArea = [Area1, Area2, Area3, Area4, Area5]
        
        # 计算多传感器边界区域
        DestMultiAreaEdge = CalMultiAreaEdge(MultiArea)
        print('DestMultiAreaEdge = ', DestMultiAreaEdge)
        # 计算告警区域和对应传感器交叉区域
        CurSensorDeteArea = [[2.396, 4.28,  0.007, 1.497]] # [xmin, xmax, ymin, ymax], ZT-15
#        CurSensorDeteArea = [[0.569, 2.396, 1.497, 2.950]] # [xmin, xmax, ymin, ymax], ZT-17
        
        ValidDeteArea = CalIntersectAreaFromTwoArea(DeteAlarmArea, CurSensorDeteArea[0]) # 告警区域和传感器检测区域交叉区域，,【暂定单个传感器只有一个检测区域】
        print('ValidDeteArea = {}'.format(ValidDeteArea))
        # 有效告警区域和边界区域合并
        AboveHeightValidArea = CalAboveHeightValidArea(ValidDeteArea, DestMultiAreaEdge, DeteAreaInnerDis = 0.1)
        print('AboveHeightValidArea = {}'.format(AboveHeightValidArea))
        
        # 显示结果
        if PlotResultFlag == 1:
            ColorGroup = ['b','g','r','c','m','y','k']
            fig = plt.figure()
            plt.clf()
            plt.subplot(111)
            ax = plt.gca()
            # 画原始框
            for i in range(len(MultiArea)):
                CurOneArea = np.array(MultiArea[i])
                CurOneArea = CurOneArea.reshape([int((CurOneArea.shape[0])/2),2]) # [x1,y1;x2,y2;x3,y3;x4,y4]
                plt.plot(CurOneArea[:,0], CurOneArea[:,1], color=ColorGroup[i])
                plt.plot([CurOneArea[-1,0], CurOneArea[0,0]], [CurOneArea[-1,1], CurOneArea[0,1]], color=ColorGroup[i])
                            
            # 画边界框
            for i in range(len(DestMultiAreaEdge)):
                CurOneArea = np.array(DestMultiAreaEdge[i])
                CurOneArea_xmin = CurOneArea[0]
                CurOneArea_xmax = CurOneArea[1]
                CurOneArea_ymin = CurOneArea[2]
                CurOneArea_ymax = CurOneArea[3]
                print(i, ' = ', CurOneArea_xmin, CurOneArea_xmax, CurOneArea_ymin, CurOneArea_ymax)
                rect = plt.Rectangle((CurOneArea_xmin,CurOneArea_ymin),CurOneArea_xmax-CurOneArea_xmin,CurOneArea_ymax-CurOneArea_ymin, fill=False, edgecolor='g', linewidth=3)
                ax.add_patch(rect)
                
            # 画目标检测框
            if True:
                CurDeteLoc = ValidDeteArea
                rect = plt.Rectangle((CurDeteLoc[0], CurDeteLoc[2]), CurDeteLoc[1]-CurDeteLoc[0], CurDeteLoc[3]-CurDeteLoc[2], fill=False, edgecolor='y', linewidth=3)
                ax.add_patch(rect)
                
            # 画目标检测框最终区域
            if True:
                CurDeteLoc = AboveHeightValidArea
                rect = plt.Rectangle((CurDeteLoc[0], CurDeteLoc[2]), CurDeteLoc[1]-CurDeteLoc[0], CurDeteLoc[3]-CurDeteLoc[2], fill=False, edgecolor='k', linewidth=3)
                ax.add_patch(rect)
            
            plt.axis('equal')
            plt.show()
            plt.savefig('AboveHeightValidArea.png')
            
    if TestPlotAllAlarmAreaFlag == 1: # 显示所有告警区域范围
        # read config
        ConfigInfo = []
    
    print('End.')

