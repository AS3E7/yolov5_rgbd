# -*- coding: utf-8 -*-
"""
Created on Sat Nov 04 12:56:36 2017

@author: xbiao
"""

import numpy as np
from ctypes import *

def LoadPointLabelIndex():
#    PTtype=np.dtype([('INVALID','i'),('OUTBUND','i'),('NOISE','i'),('UNDEFINED','i'),('BACKGROUND','i')])
#    PT_LABEL=np.array([(-255,-3,-2,-1,0)],dtype=PTtype)

#    class PTtype(Structure):
#        _fields_ = [("INVALID", c_int),
#                    ("OUTBUND", c_int),
#                    ("NOISE", c_int),
#                    ("UNDEFINED", c_int),
#                    ("BACKGROUND", c_int)]

    class PTtype():
        def __init__(self):
            self.INVALID = 0
            self.OUTBUND = 0
            self.NOISE = 0
            self.UNDEFINED = 0
            self.BACKGROUND = 0
                    
    PT_LABEL = PTtype()
    PT_LABEL.INVALID = -255
    PT_LABEL.OUTBUND = -3
    PT_LABEL.NOISE = -2
    PT_LABEL.UNDEFINED = -1
    PT_LABEL.BACKGROUND = 0
    return PT_LABEL

def ReadKinect2FromPly(plyname, KNT2_WIDTH  = 512, KNT2_HEIGHT = 424):
    #KNT2_WIDTH  = 512
    #KNT2_HEIGHT = 424
    
    # Judge the file eixt or not
    fp = open(plyname, 'r')
    if (fp==-1):
        print('Cannot open file') 
        return 0
    lines=fp.readlines()
    if ((len(lines)-9)!=(KNT2_WIDTH*KNT2_HEIGHT)):
        print('Unmatched ply file')
        return 0
    fp.close()
    
    # read data
    fp = open(plyname, 'r')
    for i in range(0,9):
        CurrLine=fp.readline()
    
    data=np.zeros((3,(len(lines)-9)))
    data_num=0
    while 1:
        line = fp.readline()
        line = line.strip().split(' ')
#        print(line)
        filetemp=np.zeros((len(line)))
        if len(line)>1:
            for j in range(len(line)):
                p=float(line[j])
                filetemp[j]=p
            data[:,data_num]=filetemp.transpose()
            data_num = data_num+1
        else:
            break
#            print(data_num)
#        data_new=np.vstack((data,filetemp.transpose()))
#        data=data_new
        
    fp.close()
    return data

def Rot3D(H, pts):
    tmp = np.row_stack((pts,np.ones([1, pts.shape[1]])))
    newpts = np.dot(H,tmp)
    newpts = newpts[0:3, :]
    return newpts
    
def SavePt3D2Ply(plyname,pts):
#    # XYZ_RGB
#    VertexNum = pts.shape[0]
#    VertexFormat = 'XYZ_RGB'
#    fp = open(plyname, 'wb+')
#    SavePlyHeader(fp, VertexNum,VertexFormat)
#    for idx in range(0,VertexNum):
#        tmp=np.str('%.3f'%pts[idx,0])+' '+np.str('%.3f'%pts[idx,1])+' '+np.str('%.3f'%pts[idx,2])+' '+np.str('%.d'%pts[idx,3])+' '+np.str('%.d'%pts[idx,4])+' '+np.str('%.d'%pts[idx,5])+'\n';
#        fp.write(tmp.encode())
#    fp.close()

    # XYZ
    VertexNum = pts.shape[0]
    VertexFormat = 'XYZ'
    fp = open(plyname, 'wb+')
    SavePlyHeader(fp, VertexNum,VertexFormat)
    for idx in range(0,VertexNum):
        tmp=np.str('%.3f'%pts[idx,0])+' '+np.str('%.3f'%pts[idx,1])+' '+np.str('%.3f'%pts[idx,2])+'\n';
        fp.write(tmp.encode())
    fp.close()
    return 0
    
def SavePlyHeader(fp, VertexNum,VertexFormat):
    status = 0;
    if(fp == -1):
        return 0

#    fp.write(('ply\n').encode())
#    fp.write(('format ascii 1.0\n').encode()) 
#    fp.write(('element vertex %d\n' % VertexNum).encode())
#    fp.write(('property float x\n').encode())
#    fp.write(('property float y\n').encode())
#    fp.write(('property float z\n').encode())
#    fp.write(('property uint8 red\n').encode())
#    fp.write(('property uint8 green\n').encode())
#    fp.write(('property uint8 blue\n').encode())
#    fp.write(('element face 0\n').encode())
#    fp.write(('property list uint8 int32 vertex_index\n').encode())
#    fp.write(('end_header\n').encode())

    fp.write(('ply\n').encode())
    fp.write(('format ascii 1.0\n').encode()) 
    fp.write(('element vertex %d\n' % VertexNum).encode())
    fp.write(('property float x\n').encode())
    fp.write(('property float y\n').encode())
    fp.write(('property float z\n').encode())
    fp.write(('element face 0\n').encode())
    fp.write(('property list uint8 int32 vertex_index\n').encode())
    fp.write(('end_header\n').encode())
    
    status = 1
    return status
    
    
    
    
    
    