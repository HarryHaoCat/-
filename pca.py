# -*-coding: utf-8 -*-

from numpy import*
# 将文本信息转换为矩阵返回
def loadDataSet(fileName):
    fr = open(fileName)
    stringArr = [line.strip().split('\t') for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]     #map(float,line)第一个参数是一个函数，第二个参数是输入的数据，函数作用于第二个输入的每一个对象并返回list
    return mat(datArr)

# PCA算法实现
def pca(dataMat,topNfeat):
    meanVals = mean(dataMat ,axis = 0)  #对矩阵的每一列取平均值
    meanRemoved = dataMat - meanVals   #去均值
    covMat = cov(meanRemoved,rowvar = 0)
    eigVals,eigVects = linalg.eig(mat(covMat))   #算得特征值(是个数组)和特征向量（是个矩阵）
    eigValInd = argsort(eigVals)               #对特征值进行从小到大的排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:,eigValInd]
    loadDataMat = meanRemoved * redEigVects  #将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    reconMat = (loadDataMat * redEigVects.T) + meanVals  #利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    return loadDataMat,reconMat

# 自适应PCA算法实现
def zsypca(dataMat):
    meanVals = mean(dataMat ,axis = 0)  #对矩阵的每一列取平均值
    meanRemoved = dataMat - meanVals   #去均值
    covMat = cov(meanRemoved,rowvar = 0)
    eigVals,eigVects = linalg.eig(mat(covMat))   #算得特征值(是个数组)和特征向量（是个矩阵）
    eigVals = real(eigVals)
    eigValInd = argsort(eigVals)               #对特征值进行从小到大的排序
    N = len(eigValInd)                         #特征值的个数
    eigValInd = eigValInd[:-(N+1):-1]
    newEigValInd = []
    sumEigVals = sum(eigVals)
    i = 0
    sumOfEigVls= 0
    while sumOfEigVls/float(sumEigVals) < 0.98:  # 压缩比约为98%
        sumOfEigVls += eigVals[i]
        newEigValInd.append(i)
        i += 1
    demension = i
    compressRatio = sumOfEigVls/float(sumEigVals)
    newEigValInd = array(newEigValInd)
    redEigVects = eigVects[:,newEigValInd]
    redEigVects = real(redEigVects)
    loadDataMat = meanRemoved * redEigVects  #将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    reconMat = (loadDataMat * redEigVects.T) + meanVals  #利用降维后的矩阵反构出原数据矩阵(用作测试，可跟未压缩的原矩阵比对)
    return loadDataMat,reconMat,demension ,compressRatio,redEigVects,meanVals


