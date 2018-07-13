# -*- coding: utf-8 -*-
# 测试函数
import myKNN
import pca
from numpy import*
resultLists = ['不喜欢', '魅力一般', '很有魅力']
percentTats = float(raw_input("玩视频游戏所耗时间百分比？"))
ffMiles = float(raw_input("每年获得的飞行常客旅程数？"))
iceCream = float(raw_input("每周消耗的冰淇淋公升数？"))
returnMat, classLabelVector = myKNN.file2matrix('datingTestSet2.txt')
dataMat = mat(returnMat)
loadDataMat,reconMat,demension ,compressRatio,redEigVects ,meanVals= pca.zsypca(dataMat)
#inX = ([ffMiles,percentTats,iceCream] - minVal)/ranges
inData = mat([ffMiles,percentTats,iceCream])
removeMean = inData - meanVals
inLoadDataMat = array(removeMean * redEigVects)
classifyResults = myKNN.classify0(inLoadDataMat, loadDataMat,classLabelVector, 3)
print classifyResults
print "你将可能喜欢这个人：",resultLists[classifyResults-1]