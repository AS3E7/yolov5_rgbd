import numpy as np


def ReadKinect2FromPly(plyname, KNT2_WIDTH=512, KNT2_HEIGHT=424):
    # KNT2_WIDTH  = 512
    # KNT2_HEIGHT = 424
    # Judge the file eixt or not
    fp = open(plyname, 'r')
    if (fp == -1):
        print('Cannot open file')
        return 0
    lines = fp.readlines()
    if ((len(lines) - 9) != (KNT2_WIDTH * KNT2_HEIGHT)):
        print('Unmatched ply file')
        return 0
    fp.close()

    # read data
    fp = open(plyname, 'r')
    for i in range(0, 9):
        CurrLine = fp.readline()

    data = np.zeros(((len(lines) - 9), 3))
    data_num = 0
    while 1:
        line = fp.readline()
        line = line.strip().split(' ')
        #        print(line)
        filetemp = np.zeros((len(line)))
        if len(line) > 1:
            for j in range(len(line)):
                p = float(line[j])
                filetemp[j] = p
            data[data_num, :] = filetemp
            data_num = data_num + 1
        else:
            break
    #            print(data_num)
    #        data_new=np.vstack((data,filetemp.transpose()))
    #        data=data_new

    fp.close()
    return data

def ReadKinect26FromPly(plyname, KNT2_WIDTH=512, KNT2_HEIGHT=424):
    # KNT2_WIDTH  = 512
    # KNT2_HEIGHT = 424
    # Judge the file eixt or not
    fp = open(plyname, 'r')
    if (fp == -1):
        print('Cannot open file')
        return 0
    lines = fp.readlines()
    if ((len(lines) - 12) != (KNT2_WIDTH * KNT2_HEIGHT)):
        print('Unmatched ply file')
        return 0
    fp.close()

    # read data
    fp = open(plyname, 'r')
    for i in range(0, 12):
        CurrLine = fp.readline()

    data = np.zeros(((len(lines) - 12), 6))
    data_num = 0
    while 1:
        line = fp.readline()
        line = line.strip().split(' ')
        #        print(line)
        filetemp = np.zeros((len(line)))
        if len(line) > 1:
            for j in range(len(line)):
                p = float(line[j])
                filetemp[j] = p
            data[data_num,:] = filetemp
            data_num = data_num + 1
        else:
            break
    #            print(data_num)
    #        data_new=np.vstack((data,filetemp.transpose()))
    #        data=data_new

    fp.close()
    return data





def SavePt3D26Ply(plyname,pts):


    # XYZ_RGB
    VertexNum = pts.shape[0]
    VertexFormat = 'XYZ_RGB'
    fp = open(plyname, 'wb+')
    SavePlyHeader(fp, VertexNum)
    for idx in range(0,pts.shape[0]):
       tmp=np.str_('%.3f'%pts[idx,0])+' '+np.str_('%.3f'%pts[idx,1])+' '+np.str_('%.3f'%pts[idx,2])+' '+np.str_('%.d'%pts[idx,3])+' '+np.str_('%.d'%pts[idx,4])+' '+np.str_('%.d'%pts[idx,5])+'\n';
       fp.write(tmp.encode())
    fp.close()

        # # XYZ
        # VertexNum = pts.shape[0]
        # VertexFormat = 'XYZRGB'
        # fp = open(plyname, 'wb+')
        # SavePlyHeader(fp, VertexNum,pts)
        # for idx in range(0,VertexNum):
        #     tmp=np.str('%.3f'%pts[idx,0])+' '+np.str('%.3f'%pts[idx,1])+' '+np.str('%.3f'%pts[idx,2])+'\n';
        #     fp.write(tmp.encode())
        # fp.close()
    return 0


def SavePlyHeader(fp, VertexNum):
    status = 0;
    if (fp == -1):
        return 0

    fp.write(('ply\n').encode())
    fp.write(('format ascii 1.0\n').encode())
    fp.write(('element vertex %d\n' % VertexNum).encode())
    fp.write(('property float x\n').encode())
    fp.write(('property float y\n').encode())
    fp.write(('property float z\n').encode())
    fp.write(('property uint8 red\n').encode())
    fp.write(('property uint8 green\n').encode())
    fp.write(('property uint8 blue\n').encode())
    fp.write(('element face 0\n').encode())
    fp.write(('property list uint8 int32 vertex_index\n').encode())
    fp.write(('end_header\n').encode())

        # # XYZ
        # fp.write(('ply\n').encode())
        # fp.write(('format ascii 1.0\n').encode())
        # fp.write(('element vertex %d\n' % VertexNum).encode())
        # fp.write(('property float x\n').encode())
        # fp.write(('property float y\n').encode())
        # fp.write(('property float z\n').encode())
        # fp.write(('element face 0\n').encode())
        # fp.write(('property list uint8 int32 vertex_index\n').encode())
        # fp.write(('end_header\n').encode())

    status = 1
    return status