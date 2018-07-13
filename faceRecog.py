# -*- coding: utf-8 -*-
#实现PCA+KNN的人脸识别

from numpy import*
from matplotlib.pyplot import*
import matplotlib.pyplot as plt
import cv2
from os import listdir
import myKNN
#读取图片并将图片转化为矩阵
def img2vect(inputArr):
    returnVect = zeros((1, 10000))
    for i in range(100):
        for j in range(100):
            returnVect[0, 100 * i + j] = int(inputArr[i,j]) # 因为返回的矩阵是Str类型的所以要转换为int类型
    return returnVect








