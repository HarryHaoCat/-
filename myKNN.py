# -*- coding: utf-8 -*-
from  numpy import*
import  operator
import pca

# k-近邻算法
def classify0(inx,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = array(tile(inx,(dataSetSize,1)) - dataSet)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]   #返回元组列表的第一个元组的第一个元素就是分类的标签

def file2matrix(filename):
    fr = open(filename)
    returnArr = [line.strip().split('\t') for line in fr.readlines()]
    returnMat = mat([map(float,line) for line in returnArr])[:,0:3]
    classLabelVector1 = mat([map(float,line) for line in returnArr])[:,-1]
    classLabelVector2 = classLabelVector1.T.tolist()[0]
    return returnMat,classLabelVector2


# 归一化特征值
def aotuNorm(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0) #返回每一列的最大值
    ranges = maxVal - minVal
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - tile(minVal,(m,1)))/tile(ranges,(m,1))
    return  normDataSet,ranges ,maxVal, minVal

#测试分类的准确率
def dataingClassTest():
    hoRatio = 0.1  # 将训练样本的10%作为测试
    returnMat,classLabelVector = file2matrix('datingTestSet2.txt')
    normDataSet, ranges, maxVal, minVal = aotuNorm(returnMat)
    print normDataSet
    m = normDataSet.shape[0]
    numberOfTestSet = int(m*hoRatio)   #求出样本的个数
    errorCount = 0
    for i in range(numberOfTestSet):
        classifyResults = classify0(normDataSet[i,:],normDataSet[numberOfTestSet:m,:],classLabelVector[numberOfTestSet:m],3)
        print "the classifier came back with: %d, the real answer is: %d"%(classifyResults,classLabelVector[i])
        if(classifyResults != classLabelVector[i]):
            errorCount += 1.0
    print "the total error rate is: %f" %(errorCount/float(numberOfTestSet))

# 将32*32 的手写数字转换为1*1024的行向量
def img2vector(filename):
    fr = open(filename)
    returnVect = zeros((1,1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])    #因为返回的矩阵是Str类型的所以要转换为int类型
    return returnVect


