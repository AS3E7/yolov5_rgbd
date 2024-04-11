# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 19:27:50 2019

@author: JiangZhuang
"""

import numpy as np
import matplotlib.path as path

def Fighting_Detection(xyz, number, radius=2.0):
    '''

    :param xyz: 人员位置(list)
    :param number: 人数限制(int)
    :param radius: 范围限制(float)
    :return: thing_num：坐标(list)
    '''
    thing_num = []
    for i in range(len(xyz)):
        xyz_one = xyz[i]
        x1 = xyz_one[0] - radius
        x2 = xyz_one[0] + radius
        y1 = xyz_one[1] + radius
        y2 = xyz_one[1] - radius
        point0 = (x1, y1)
        point1 = (x1, y2)
        point2 = (x2, y2)
        point3 = (x2, y1)
        region = path.Path([point0, point1, point2, point3])
#        region = Remove_Other(xyz[i])
        person = 0
        bool_list = region.contains_points(xyz)
        for j in bool_list:
            if j == True:
                person += 1
        
        if person >= number:
            thing_num.append([round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)])
    
    return thing_num

def Fighting_Detection_2(xyz, distance=0.2):
    thing_num = []
    for i in range(len(xyz)):
        xyz_one = xyz[i]
        x1 = xyz_one[0]
        y1 = xyz_one[1]
        for j in range(len(xyz)):
            if i != j:
                xyz_two = xyz[j]
                x2 = xyz_two[0]
                y2 = xyz_two[1]
                z = ((x1-x2)**2 + (y1-y2)**2)**(1/2)
                if z <= distance:
                    thing_num.append([round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)])
                    
    return thing_num

if __name__ == "__main__":
    b = np.random.random([40,2])
    c = Fighting_Detection(b, 5, radius=0.1)
    print(c)
    print(type(c))
    print(len(c))