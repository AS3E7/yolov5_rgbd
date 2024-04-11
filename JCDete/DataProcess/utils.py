# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:34:19 2021

@author: Administrator
"""

import os
from shutil import copyfile

def DirMake(FolderName):    
    """
    功能：生成文件夹
    """
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)
    return 0

def DirFileNum(FolderName):
    """
    功能：获取文件夹中文件数据量
    """
    FileList = os.listdir(FolderName)
    FileNum = len(FileList)
    return FileNum

def CopyFolderFile(SrcFolderNamem, DestFolderName):
    """
    功能：复制文件夹文件
    """
    SrcFileList = os.listdir(SrcFolderNamem)
    for iFileName in SrcFileList:
        CurFileName = os.path.join(SrcFolderNamem, iFileName)
        DestFileName = os.path.join(DestFolderName, iFileName)
        copyfile(CurFileName, DestFileName)
    return 0
    
def CopyFileFolderToFolder(SrcFolderName, DestFolderName):
    """
    功能：从一个文件夹复制文件到另一个文件夹
    """
    FileList = os.listdir(SrcFolderName)
    for OneFile in FileList:
        OneSrcFileFullName = os.path.join(SrcFolderName, OneFile)
        OneDestFileFullName = os.path.join(DestFolderName, OneFile)
        copyfile(OneSrcFileFullName, OneDestFileFullName)
    return 0
