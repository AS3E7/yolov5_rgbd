import os
import time
import cv2
import numpy as np 

def GetSampData(SampPts, Samp_center, SampWinSize_x, SampWinSize_z, GridSize2d):

    SampList = Samp_center
    GS2_NUM_x = np.round(SampWinSize_x/ GridSize2d)
    GS2_NUM_z = np.round(SampWinSize_z/ GridSize2d)
    gbx1 = SampList[0] - SampWinSize_x / 2
    gbz1 = SampList[2] - SampWinSize_z
    gix = np.ceil((SampPts[0, :] - gbx1) / GridSize2d)
    giz = np.ceil((SampPts[2, :] - gbz1) / GridSize2d)
    gix[gix < 1] = 1
    gix[gix > GS2_NUM_x] = GS2_NUM_x
    giz[giz < 1] = 1
    giz[giz > GS2_NUM_z] = GS2_NUM_z

    gidx = (giz - 1) * GS2_NUM_x + gix  # grid map index 
    grid_y = SampPts[1, :]
    SampGridData2 = np.zeros([1,int(GS2_NUM_x * GS2_NUM_z)])
    gidx2 = gidx.astype(int)
    gidx3 = gidx2.tolist()
    gidx_sort = sorted(gidx3)  #排序后结果
    gidx_idx = np.argsort(gidx3) #排序后数据对应原始索引
    
    n, m  = np.histogram(gidx_sort, bins=int(GS2_NUM_x * GS2_NUM_z) , range=(0,int(GS2_NUM_x * GS2_NUM_z)))
    index_cum = np.cumsum(n)
    Grid_t = np.where(n != 0)
    Grid_t = list(Grid_t)
    Grid_t = Grid_t[0]
    i = 0
    for ng in Grid_t:
        pt_num = index_cum[ng]
        pts_idx = gidx_idx[i:pt_num]
        SampGridData2[0,ng-1] = 1 if len(grid_y[pts_idx])>0 else 0
        i = pt_num        
    img = SampGridData2[0,:].reshape(int(GS2_NUM_z),-1)
    GridData = np.asarray(img)             
    GridData = cv2.flip(GridData,0)

    return GridData, GS2_NUM_x, GS2_NUM_z


def connected(SampPts,Samp_center,SampWinSize_x,SampWinSize_z,GridSize2d=0.1,connect_rate=0.8):
    '''
    input: SampPts:点云数据， 3 x n 格式， 数据类型：array
           Samp_center：超高点中心，[x, y, z]，数据类型：list
           SampWinSize_x：选取样本的水平方向宽度，单位（米），数据类型：float
           SampWinSize_z：选取样本的竖直方向宽度，单位（米），数据类型：float
           GridSize2d：网格大小，单位（米），数据类型：float
           connect_rate：连通比例，数据类型：float

    return: mark：样本是否连通的标志，数据类型：int
    '''
    #data grid 
    GridData,GS2_NUM_x,GS2_NUM_z = GetSampData(SampPts,Samp_center,SampWinSize_x,SampWinSize_z,GridSize2d)
    # GridData = GridData*255
    # samp_img_name = 'temp/' + str(parm) + '.png'
    # cv2.imwrite(samp_img_name, GridData)
    # print('save pic in ', samp_img_name)
    ##connected components
    con_img = GridData.astype(np.uint8)
    ret, labels = cv2.connectedComponents(con_img, connectivity=8)
    # print('num of connected components :', ret)
    # print(labels)
    mark = 0
    rate_max = 0 #最大连通比例
    if ret > 1:
        label_arr = labels.flatten()
        for i in range(ret-1):
            k = i + 1
            lab_min_h = np.round(np.min(np.where(label_arr==k)) / GS2_NUM_x)
            lab_max_h = np.round(np.max(np.where(label_arr==k)) / GS2_NUM_x)
            h_rate = (lab_max_h-lab_min_h + 1) / GS2_NUM_z
            if h_rate > rate_max:
                rate_max = h_rate
            if h_rate >= connect_rate:
                mark = 1
                break
    # print('mark: ', mark)
    # print('rate_max: ', rate_max)
    # labels_img = labels*(255/max(1,ret-1))
    # cv2.imwrite('temp/' + str(parm) + '_' + str(mark) +'_labels.png', labels)

    return mark


if __name__ == "__main__":
    import CloudPointFuns
    import LoadGateCounterInfo
    
    #get pts
    H, theAxis, GateArea, GateAreaName, PlyFmt, Frame1, Frame2, ImshowFlag = LoadGateCounterInfo.config()
    save_ply = 1
    save_samp_ply = 1
    for parent, dirnames, filenames in os.walk(PlyFmt):
        for filename in filenames:
            PlyName = PlyFmt + '/' + filename
            parm = os.path.basename(PlyName)[:-4]
            print('======== Frame Name:',parm ,'========')
            Pts = CloudPointFuns.ReadKinect2FromPly(PlyName)
            NewPts = CloudPointFuns.Rot3D(H, Pts)
            if save_ply:
                result_ply_dir='temp/' + str(parm)+'.ply'
                CloudPointFuns.SavePt3D2Ply(result_ply_dir,NewPts.transpose())
            #select area
            SampList = [1,1,2] 
            SampWinSize_x = 1.0
            SampWinSize_y = 1.0
            SampWinSize_z = 1.5   
            SampPos = (NewPts[0, :] > (SampList[0] - 0.50 * SampWinSize_x)) & (NewPts[0, :] < (SampList[0] + 0.50 * SampWinSize_x)) \
                    & (NewPts[1, :] > (SampList[1] - 0.50 * SampWinSize_y)) & (NewPts[1, :] < (SampList[1] + 0.50 * SampWinSize_y)) \
                    & (NewPts[2, :] > (SampList[2] - SampWinSize_z)) & (NewPts[2, :] < (SampList[2]))
            SampPts = NewPts[:, SampPos]
            if save_samp_ply:
                result_ply_dir='temp/' + str(parm)+'_samp.ply'
                CloudPointFuns.SavePt3D2Ply(result_ply_dir,SampPts.transpose())
            Samp_center = SampList
            #接口调用
            t1 = time.time()
            mark = connected(SampPts,Samp_center,SampWinSize_x,SampWinSize_z,GridSize2d=0.1,connect_rate=0.8)
            t2 = time.time()
            print('mark: ', mark)
            print('time = {}'.format(t2 - t1))
            print('finish !')



       