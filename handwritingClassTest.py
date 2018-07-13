# -*- coding: utf-8 -*-

#  手写数字识别测试程序
import time
from numpy import*
from os import*
import myKNN
import pca

start_time = time.time()
hwLabel = []
trainingFileList = listdir('trainingDigits')
N = len(trainingFileList)
trainingMat = zeros((N,1024))
for i in range(N):
    trainingNameStr = trainingFileList[i].split('.')[0]
    trainingNameStr = trainingNameStr.split('_')[0]
    hwLabel.append(trainingNameStr)
    trainingMat[i,:] = myKNN.img2vector('trainingDigits/%s'%trainingFileList[i])
trainingMat = mat(trainingMat)
loadDataMat,reconMat,demension ,compressRatio,redEigVects,meanVals = pca.zsypca(trainingMat)
testFileList = listdir('testDigits')
M = len(testFileList)
errorCount = 0.0
for j in range(M):
    testNameStr = testFileList[j].split('.')[0]
    testNameStr = testNameStr.split('_')[0]
    testArr = myKNN.img2vector('testDigits/%s'%testFileList[j])
    removeMean = testArr - meanVals
    testData = removeMean * redEigVects
    classifyResults = myKNN.classify0(testData, loadDataMat, hwLabel, 3)
    #print 'the classifyResults is: %s,the real answer is: %s'%(classifyResults,testNameStr)
    if classifyResults != testNameStr:
        errorCount += 1
print 'the total error rate is: %f'%(errorCount/float(M))
end_time = time.time()
print 'total time is: ',end_time - start_time






