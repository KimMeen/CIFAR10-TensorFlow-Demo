# -*- coding: utf-8 -*-
"""
@author: Ming JIN
"""

def getFileName(path):
    return path.split('/')[-1]

def readLines(file_path):
    with open(file_path, 'r') as T:
        lines = T.readlines()
    return lines

def getLabel(src):
    lines = src
    label_record = {}
    for line in lines:
        name = line.split(' ')[0]
        label = line.split(' ')[1]
        label = label.split('\n')[0]
        label_record[name] = label
    return label_record