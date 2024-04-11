# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:39:40 2019

@author: Administrator
"""

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = os.path.dirname(__file__)

# Add lib to python path
lib_path = os.path.join(this_dir, '..')
add_path(lib_path)
