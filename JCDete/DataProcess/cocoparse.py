#!/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/30
# @Author : NOAH
# @File : cocoparse.py

from PIL import Image
import numpy as np
import imgviz
import copy
import os
import cv2
import tqdm
import shutil
from pointcloudutils import ReadKinect2FromPly
from pointcloudutils import ReadKinect26FromPly,SavePt3D26Ply
from pycocotools.coco import COCO
import json
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
from scipy.spatial import distance
import time

def getobjInfo(cocofile, destfile,destplypath,plydestpath, plydest,graypath,CalibHFolderName):
    """
    功能：生成三维目标框及关节点并保存为json并展示标注结果

    """

    coco = COCO(cocofile)
    coco_dataset = copy.deepcopy(coco.dataset)
    annotations = []

    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):

        # 获取coco信息
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)

        # 获取点云、配准及深度信息
        plyname = img['file_name'].replace('.png', '.ply')
        # if plyname == '175_Depth2018-12-06-153000_01401.ply':


        i_file = 'H_' + plyname.split('_')[0] + '.txt'
        CurParamsFile = os.path.join(CalibHFolderName, i_file)
        CurSensorH = ReadCalibParamsInfo(CurParamsFile)
        H = CurSensorH[2, 3]
        OneplyDir = os.path.join(destplypath, plyname)
        pts = ReadKinect26FromPly(OneplyDir)
        onegreyimgdir = os.path.join(graypath, img['file_name'])
        onegreyimg = cv2.imread(onegreyimgdir, 0).transpose()

        colorlist = []
        for i in range(len(anns)):  # 遍历目标

            # 生成目标框
            onemask = coco.annToMask(anns[i])
            nanzero = np.reshape(onemask.transpose(), [217088, ])
            lim = np.nonzero(nanzero)
            lims = list(lim)[0]
            objpts = pts[lims, :]
            validrate = (objpts[:, 0] != 0) | (objpts[:, 1] != 0) | (objpts[:, 2] != round(H,3))
            validpts = objpts[validrate, :]
            if validpts.shape[0] < 50:  # 空框及边界无效目标
                bbox = [0, 0, 0, 0, 0, 0]
            elif validpts.shape[0] < 200:  # 边界有框
                bbox, dellist = postpro(validpts, 10, 0.1, 0.8)  # 聚类后处理
            else:  # 正常目标
                bbox, dellist = postpro(validpts, 50, 0.15, 0.8)  # 头部点数约为50-80

            # 生成关节点
            onekeypoints = anns[i]['keypoints']
            jointlist = listsplit(onekeypoints, 3)
            keypoints = []
            for j in jointlist:
                if j[2] != 0 and onegreyimg[j[0] - 1, j[1] - 1] != 0:  # 正常情况
                    jointpoint = pts[424 * (j[0] - 1) + j[1] - 1, :]
                    newkeypoints = getpoints(jointpoint, j)
                elif j[2] == 0:  # 不可见
                        newkeypoints = [0, 0, round(H,3), 0]
                else:  # 替代点
                    maxlen = 6
                    newkeypoints = pointnearest(maxlen, j, pts, onegreyimg)
                keypoints += newkeypoints

            # 汇总信息
            colorlist = plotpoint(colorlist,keypoints,bbox)
            pts = delepro(pts, validrate, dellist, lims)
            anns[i]['bbox'] = bbox
            anns[i]['keypoints'] = keypoints
        annotations += anns

        # 保存点云
        colorpoints = np.array(colorlist)
        Pts = np.concatenate((colorpoints, pts), axis=0, out=None)
        plyDir = os.path.join(plydest, plyname)
        SavePt3D26Ply(plyDir, Pts)
        CurPlyFileName = os.path.join(plydestpath, plyname)
        SavePt3D26Ply(CurPlyFileName, pts)


    # 保存json
    coco_dataset['annotations'] = annotations
    saveobjtojson(destfile, coco_dataset)

    return 0


def plypro (srcplypath,srcimgpath,destplypath):
    """
    生成XYZRGB点云数据

    """
    for rt, dirs, files in os.walk(srcplypath):
        print("plyId lens:{}".format(len(files)))
        for filename in files:
            if filename.endswith('.ply'):  # 注意缺失mask
                # if filename == '175_Depth2018-12-06-153000_01401.ply':
                OnePlyDir = os.path.join(srcplypath, filename)
                pts = ReadKinect2FromPly(OnePlyDir)
                OneimgDir = os.path.join(srcimgpath, filename.replace('.ply', '.png'))
                img0 = cv2.imread(OneimgDir, 1)
                img = np.reshape(img0.transpose(1, 0, 2), [217088, 3])  # 按列存放
                Pts = np.concatenate((pts, img), axis=1, out=None)
                CurPlyFileName = os.path.join(destplypath, filename.replace('.png', '.ply'))
                print(CurPlyFileName)
                SavePt3D26Ply(CurPlyFileName, Pts)

    return 0


def getmask(cocofile,graypath,outgraypath,Segimgpath):
    """
    获取mask图像

    """

    coco = COCO(cocofile)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):

        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        # plyname = img['file_name'].replace('.png', '.ply')
        # if plyname == '175_Depth2018-12-06-153000_01401.ply':
        # 获取mask图像
        if len(annIds) > 0:
            mask = coco.annToMask(anns[0]) * anns[0]['category_id']
            if len(annIds) == 1:
                img_origin_path = os.path.join(graypath, img['file_name'])
                img_output_path = os.path.join(outgraypath, img['file_name'])
                seg_output_path = os.path.join(Segimgpath, img['file_name'].replace('.jpg', '.png'))
                shutil.copy(img_origin_path, img_output_path)
                save_colored_mask(mask, seg_output_path)
            else:
                for i in range(len(anns) - 1):
                    mask += coco.annToMask(anns[i + 1]) * (i + 2)  # 设置调色板参数
                    img_origin_path = os.path.join(graypath, img['file_name'])
                    img_output_path = os.path.join(outgraypath, img['file_name'])
                    seg_output_path = os.path.join(Segimgpath, img['file_name'].replace('.jpg', '.png'))
                    shutil.copy(img_origin_path, img_output_path)
                    save_colored_mask(mask, seg_output_path)

    return 0

def plotpoint(colorlist,keypoints,bbox):
    """
    获取展示信息

    """
    onetable = np.array(
        [[139, 139, 0], [105, 105, 105], [72, 61, 139], [70, 130, 180], [64, 224, 208], [85, 107, 47],
         [154, 205, 50], [188, 143, 143], [255, 140, 0], [255, 105, 180], [0, 104, 139], [139, 0, 0],
         [139, 105, 20]])
    table = []
    for j in list(onetable):
        one = [j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j, j]
        table += one

    jointlist = np.array(listsplit(keypoints, 4))
    keypoint = list(jointlist[:, (0, 1, 2)])
    num = 0.01
    newkeypoints = expand(keypoint, num)
    newcolorpoints = list(np.concatenate((np.array(newkeypoints), np.array(table)), axis=1, out=None))
    colorlist += newcolorpoints

    innum = 10
    threedbox = underline(bbox, innum)
    if threedbox.shape[0] == 0:
        colorlist.append(np.array([0, 0, 0, 0, 0, 0]))
    else:
        onepoint = [255, 255, 255] * (threedbox.shape[0])
        line = np.array(listsplit(onepoint, 3))
        colorbbox = list(np.concatenate((threedbox, line), axis=1, out=None))
        colorlist += colorbbox
    
    return colorlist


def delepro(pts, validrate, dellist,lims):
    """
    替换无效点、噪点及边缘点
    """

    deletlist = []
    deletelist = []
    for a, b in enumerate(validrate):
        if b == 0:
            deletlist += [a]
        else:
            if a - len(deletlist) in dellist:
                deletelist += [a]
    if len(deletlist) == 0:
        pts = pts
    else:
        delarray = lims[np.array(deletlist + deletelist)]
        Pts = pts[delarray, :]
        black = [255] * (delarray.shape[0])  # 替换点颜色
        point = np.array(black).reshape((delarray.shape[0], 1))
        for s in range(3):
            Pts[:, [3 + s]] = point
        pts[delarray, :] = Pts
    return pts

def postpro (validpts, pointnum, height, distance):
    """
    聚类后处理

    """

    valpts = validpts[:, [0, 1, 2]]
    bandwidth = estimate_bandwidth(valpts, quantile=0.1, n_samples=1000)
    clf = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit(valpts)
    labels = clf.labels_
    centers = clf.cluster_centers_
    eclist, nclen = nclist(labels, centers, valpts)
    ecarray = np.array(eclist, dtype=object)
    zlen = np.array(nclen)

    if centers.shape[0] > 3:
        onecenter = centerpoint(centers[[0, 1], :])
        distance = distance
    else:
        onecenter = centers[0, :]
        distance = distance + 0.4

    dellist = []
    sollist = []
    for m in range(centers.shape[0]):
        if centers.shape[0] < 2:
            neardist = 0
        else:
            neardist = nearestdist(m, eclist, ecarray)

        if np.sum(labels == m) < pointnum:  # 去除噪点
            for a, b in enumerate(labels):
                if b == m:
                    dellist += [a]
        else:
            if neardist < 0.035 and zlen[m] > height:  # 保留连通并具有一定厚度部分
                for a, b in enumerate(labels):
                    if b == m:
                        sollist += [a]
            elif zlen[m] > height + 0.05:  # 保留厚度较大部分
                for a, b in enumerate(labels):
                    if b == m:
                        sollist += [a]
            elif dist(centers[m, :], onecenter) < distance:  # 保留一定距离部分
                for a, b in enumerate(labels):
                    if b == m:
                        sollist += [a]
            else:
                for a, b in enumerate(labels):
                    if b == m:
                        dellist += [a]

    indexs = np.array(sollist)
    if indexs.shape[0] == 0:
        bbox = [0, 0, 0, 0, 0, 0]
    else:
        newpts = validpts[indexs, :]
        bbox = loc(newpts)
    return bbox, dellist

def nearestdist(m,eclist,ecarray):
    """
    计算最近距离

    """

    neardist = []
    for j in eclist:
        if j.shape[0] < 20 or ecarray[m].shape[0] < 20:
            near = []
        else:
            near = distance.cdist(ecarray[m], j).min(axis=1)
            if np.min(near) == 0:
                near = []
            else:
                near = [np.min(near)]
        neardist += near
    if len(neardist) == 0:
        return 0
    else:
        return np.min(np.array(neardist))

def nclist(labels,centers,valpts):
    """
    获取聚类点云信息

    """

    nclist=[]
    lenn=[]
    for i in range(centers.shape[0]):
        index = np.argwhere(labels == i)
        indexs = np.squeeze(index)
        ncpts = valpts[indexs, :]
        nclist += list([ncpts])
        if ncpts.shape[0] < 5:
            zlen = 0
            lenn += [zlen]
        else:
            zlen = np.max(ncpts[:, 2]) - np.min(ncpts[:, 2])
            lenn += [zlen]

    return nclist, lenn

def centerpoint(center):
    """
    计算中心点

    """

    n = center.shape[0]
    X = Y = Z= 0
    for i in range(n):
        X += center[i,0]
        Y += center[i,1]
        Z += center[i,2]
    return [X/n,Y/n,Z/n]

def dist(pointA, pointB):
    """
    计算欧式距离

    """

    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)


def ReadCalibParamsInfo(CalibFileName):
    """
    读取配置文件信息
    输入：
        CalibFileName：配置文件名
    输出：
        H：配准参数
    """
    fp = open(CalibFileName)
    H = []
    while 1:
        line = fp.readline().strip()
        #        print(line)
        if line == '[H]':
            continue
        elif len(line) < 2:
            break
        else:
            line = line.split(' ')
            for i_line in line:
                #                print(i_line)
                if len(i_line) > 0:
                    H.append(float(i_line))
    fp.close()
    H = np.array(H).reshape((4, 4))

    return H

def underline(onebbox,innum):
    """
    划线

    """

    onebbox = np.array(onebbox)*1000
    x=list(range(int(onebbox[0]), int(onebbox[1]), innum))
    y=list(range(int(onebbox[2]), int(onebbox[3]), innum))
    z=list(range(int(onebbox[4]), int(onebbox[5]), innum))
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    for j in x:
        l = [j, int(onebbox[2]), int(onebbox[4])]
        l1 += l

        l = [j, int(onebbox[3]), int(onebbox[4])]
        l2 += l
    
        l = [j, int(onebbox[2]), int(onebbox[5])]
        l3 += l
    
        l = [j, int(onebbox[3]), int(onebbox[5])]
        l4 += l
    l5 = []
    l6 = []
    l7 = []
    l8 = []
    for j in y:
        l = [int(onebbox[0]), j, int(onebbox[4])]
        l5 += l
    
        l = [int(onebbox[1]), j, int(onebbox[4])]
        l6 += l
    
        l = [int(onebbox[0]), j, int(onebbox[5])]
        l7 += l
    
        l = [int(onebbox[1]), j, int(onebbox[5])]
        l8 += l
     
    l9 = []
    l10 = []
    l11 = []
    l12 = []
    for j in z:
        l = [int(onebbox[0]), int(onebbox[2]), j]
        l9 += l
    
        l = [int(onebbox[0]), int(onebbox[3]), j]
        l10 += l
    
        l = [int(onebbox[1]), int(onebbox[2]), j]
        l11 += l
    
        l = [int(onebbox[1]), int(onebbox[3]), j]
        l12 += l
      
    box =l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12
    threedbox = np.array(listsplit(box, 3)) /1000

    return threedbox

def expand(keypoint,num):
    """
    扩展点数量

    """

    newkeypoints = []
    for j in keypoint:
        p1 = [j[0] + num, j[1] + num, j[2]+num]
        p2 = [j[0] + num, j[1] + num, j[2]]
        p3 = [j[0] + num, j[1] + num, j[2] - num]
        p4 = [j[0] + num, j[1], j[2]+num]
        p5 = [j[0] + num, j[1], j[2]]
        p6 = [j[0] + num, j[1], j[2]-num]
        p7 = [j[0] + num, j[1] - num, j[2] + num]
        p8 = [j[0] + num, j[1]-num, j[2]]
        p9 = [j[0] + num, j[1]-num + num, j[2] - num]
        p10 = [j[0], j[1]+num, j[2]-num]
        p11 = [j[0], j[1] + num, j[2] + num]
        p12 = [j[0], j[1]+num, j[2]]
        p13 = [j[0], j[1], j[2] - num]
        p14 = [j[0], j[1], j[2]]
        p15 = [j[0], j[1], j[2] + num]
        p16 = [j[0], j[1] - num, j[2]]
        p17 = [j[0], j[1] - num, j[2] + num]
        p18 = [j[0], j[1] - num, j[2] - num]
        p19 = [j[0]-num , j[1] + num, j[2] + num]
        p20 = [j[0] - num, j[1]+num, j[2]]
        p21 = [j[0] - num, j[1] + num, j[2] - num]
        p22 = [j[0] - num, j[1], j[2]-num]
        p23 = [j[0] - num, j[1], j[2] + num]
        p24 = [j[0] - num, j[1], j[2]]
        p25 = [j[0] - num, j[1] - num, j[2] + num]
        p26 = [j[0] - num, j[1] - num, j[2]]
        p27 = [j[0] - num, j[1] - num, j[2] - num]
        point = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27]
        newkeypoints += point
    return newkeypoints

def pointnearest(maxlen,j,pts,onegreyimg):
    """
    寻找近距离点

    """

    for i in range(maxlen):
        if j[0]+i >511 or j[0]-i <0 or j[1]-i> 423 or j[1]-i<0:
            keypoint = [0,0,0,2]
            return keypoint
        elif onegreyimg[j[0]+i, j[1]-i] != 0 :
            point = pts[424 * (j[0] + i ) + j[1]-i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0] + i, j[1]] != 0:
            point = pts[424 * (j[0] + i ) + j[1], :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0] + i, j[1]+i] != 0:
            point = pts[424 * (j[0] + i ) + j[1]+i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0] - i, j[1]] != 0:
            point = pts[424 * (j[0] - i ) + j[1], :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0] - i, j[1]+i] != 0:
            point = pts[424 * (j[0] - i ) + j[1]+i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0] - i, j[1]-i] != 0:
            point = pts[424 * (j[0] - i ) + j[1]-i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0], j[1]] != 0:
            point = pts[424 * (j[0]) + j[1], :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0], j[1]+i] != 0:
            point = pts[424 * (j[0]) + j[1] +i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif onegreyimg[j[0], j[1]-i] != 0:
            point = pts[424 * (j[0]) + j[1]-i, :]
            keypoint = getpoints(point, j)
            return keypoint
        elif i == maxlen - 1:
            keypoint = [0, 0, 0, 2]
            return keypoint

def listsplit(keypoints, list_len):
    """
    切分列表

    """

    groups = zip(*(iter(keypoints),) * list_len)
    jointlist = [list(i) for i in groups]
    count = len(keypoints) % list_len
    jointlist.append(keypoints[-count:]) if count != 0 else jointlist
    return jointlist

def getpoints(jointpoint, j):

    x1 = jointpoint[0]
    y1 = jointpoint[1]
    z1 = jointpoint[2]
    v1 = j[2]
    keypoints = [x1, y1, z1, v1]
    return keypoints

def loc(validpts):

    xmin = np.min(validpts[:, 0])
    xmax = np.max(validpts[:, 0])
    ymin = np.min(validpts[:, 1])
    ymax = np.max(validpts[:, 1])
    zmin = np.min(validpts[:, 2])
    zmax = np.max(validpts[:, 2])
    bbox = [xmin, xmax, ymin, ymax, zmin, zmax]
    return bbox

def saveobjtojson(destfile, data):

    with open(destfile, 'w') as f:
        json.dump(data, f)

def save_colored_mask(mask, save_path):

    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

if __name__ == '__main__':
    print('start.')

    # 1.获取mask图像
    graypath = r'E:\work\project\JCPointCloudProcess\Data\Anno\images\train2017'
    outgraypath = r'E:\work\project\JCPointCloudProcess\Data\coco\train2017'
    #outgraypath = r'E:\work\project\JCPointCloudProcess\Data\coco\grayimg\val2017'
    cocofile = r'E:\work\project\JCPointCloudProcess\Data\Anno/person_keypoints_train2017.json'
    # cocofile = r'F:\HYD\project\JCPointCloudProcess\Data\Anno/person_keypoints_val2017.json'
    Segimgpath = r'E:\work\project\JCPointCloudProcess\Data\coco\Segmentationimg'

    # 2.生成XYZRGB点云数据
    plypath = r'E:\work\project\JCPointCloudProcess\Data\proply'
    destplypath = r'E:\work\project\JCPointCloudProcess\Data\PlyDest'

    # 3.生成三维目标框及关节点并保存点云及json文件，并展示标注结果
    destfile = r'E:\work\project\JCPointCloudProcess\Data\coco/person_keypoints_train2017.json'
    # destfile = r'F:\HYD\project\JCPointCloudProcess\Data\coco/person_keypoints_val2017.json'
    plydestpath = r'E:\work\project\JCPointCloudProcess\Data\moduleply'
    CalibHFolderName = r'E:\work\project\JCPointCloudProcess\Data\para'
    plydest = r'E:\work\project\JCPointCloudProcess\Data\ply'

    #  注意缺失mask
    Step1_Flag = 1
    Step2_Flag = 0
    Step3_Flag = 0

    if Step1_Flag == 1:
        getmask(cocofile, graypath, outgraypath, Segimgpath)

    if Step2_Flag == 1:
        plypro(plypath, Segimgpath, destplypath)

    if Step3_Flag == 1:
        getobjInfo(cocofile, destfile, destplypath, plydestpath, plydest, graypath, CalibHFolderName)

    print('end.')
