# -*-coding: utf-8 -*-
import myKNN
import pca
from numpy import *

hoRatio = 0.1  # 将训练样本的10%作为测试
returnMat, classLabelVector = myKNN.file2matrix('datingTestSet2.txt')
returnMat = mat(returnMat)
loadDataMat,reconMat,demension ,compressRatio,redEigVects,meanVals = pca.zsypca(returnMat)
m = loadDataMat.shape[0]
numberOfTestSet = int(m * hoRatio)  # 求出样本的个数
errorCount = 0
for i in range(numberOfTestSet):
    classifyResults = myKNN.classify0(loadDataMat[i, :], loadDataMat[numberOfTestSet:m, :],
                                classLabelVector[numberOfTestSet:m], 3)
   # print "the classifier came back with: %d, the real answer is: %d" % (classifyResults, classLabelVector[i])
    if (classifyResults != classLabelVector[i]):
        errorCount += 1.0
print "the total error rate is: %f" % (errorCount / float(numberOfTestSet))
