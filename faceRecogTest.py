# -*- coding: utf-8 -*-
#实现PCA+KNN的人脸识别

from numpy import*
from matplotlib.pyplot import*
import matplotlib.pyplot as plt
import cv2
from os import listdir
import myKNN
import faceRecog
import pca
import time

start_time = time.time()
trainingFileList = listdir('trainingfaces')
numOfFaces = len(trainingFileList)
classLabels = []
trainingArr = zeros((numOfFaces, 10000))
for i in range(numOfFaces):
    fileNameStr = trainingFileList[i].split('.')[0]
    fileClassStr = fileNameStr.split('_')[0]
    classLabels.append(fileClassStr)
    imagArr = mat(imread('trainingfaces/%s'%trainingFileList[i]))
    trainingArr[i, :] = faceRecog.img2vect(imagArr)
trainingMat = mat(trainingArr)
loadDataMat,reconMat,demension ,compressRatio,redEigVects,meanVals = pca.zsypca(trainingMat)
testFileList = listdir('testfaces')
numOfTestFaces = len(testFileList)
errorCount = 0.0
fig = plt.figure()
for j in range(numOfTestFaces):
    testNameStr = testFileList[j].split('.')[0]
    #testNameStr = testNameStr.split('_')[0]
    imagArr2 = mat(imread('testfaces/%s' % testFileList[j]))
    testArr2 = faceRecog.img2vect(imagArr2)
    removeMean = testArr2 - meanVals
    testData = removeMean * redEigVects
    classifyResults = myKNN.classify0(testData, loadDataMat, classLabels, 3)
    testMat = testData * redEigVects.T + meanVals
    testDataMat = testMat.reshape(100,100)
    fig.add_subplot(3,5,j+1)
    plt.axis('off')
    plt.title('Person %s'%classifyResults)
    plt.imshow(testDataMat,cmap='gray')
    print 'the classifyResults is: %s,the real answer is: %s'%(classifyResults,testNameStr)
    if classifyResults != testNameStr:
        errorCount += 1
plt.show()
print 'the total accuracy  is: %f'%(1-errorCount/float(numOfTestFaces))
end_time = time.time()
print 'total time is: ',end_time - start_time
