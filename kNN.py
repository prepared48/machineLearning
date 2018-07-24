#coding=utf-8
from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

# intX 需要被分类的数据
# dataSet 训练数据集
# labels 标签
# k 前K个
# 分类函数－－kNN分类的核心方法
def classify0(intX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(intX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat ** 2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1),reverse=True)
	print sortedClassCount
	return sortedClassCount[0][0]

# 解析数据集，最后一列示分类标签
def file2matrix(filename):
		fr = open(filename)
		arrayOLines = fr.readlines()
		numberOfLines = len(arrayOLines)
		returnMat = zeros((numberOfLines, 3))
		classLabelVector = []
		index = 0
		for line in arrayOLines:
			line = line.strip()
			listFromLine = line.split('\t')
			returnMat[index,:] = listFromLine[0:3]
			classLabelVector.append(int(listFromLine[-1]))
			index += 1
		return returnMat,classLabelVector

# 归一化特征值
# return normDataSet 归一化之后的数据集, ranges 数据集跨度, minVals 最小值
def autoNorm(dataSet):
	print dataSet
	minVals = dataSet.min(0)
	print(minVals)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	print normDataSet
	return normDataSet, ranges, minVals

# 分类器 针对约会网站的测试代码
def datingClassTest():
	horatio = 0.10
	datingDataMat, datingLabels = file2matrix('../machinelearninginaction/Ch02/datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*horatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
		print "the classifier came back with: %d, the real answer is : %d" % (classifierResult, datingLabels[i])
		if(classifierResult != datingLabels[i]) : errorCount += 1.0
	print "the total error rate is : %f" % (errorCount / float(numTestVecs))

# 约会网站预测函数
def classifyPerson():
	resultList = ['not at all','in small does','in large does']
	percentTats = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("frequent flier miles earned per year?"))
	iceCream = float(raw_input("liters o fice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('../machinelearninginaction/Ch02/datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print "You will probably like this person : ", resultList[classifierResult-1]















