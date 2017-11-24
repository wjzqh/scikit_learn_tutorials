#-*- coding:utf-8 -*-

import sys
import os
from numpy import *
import numpy as np
import operator
from Nbayes_lib import *

reload(sys)
sys.setdefaultencoding('utf-8')

k=3

#夹角余弦距离公式
def cosdist(vector1,vector2):
	return dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))
	
def classify(testdata,trainSet,listClassed,k):
	dataSetSize = trainSet.shape[0]
	distances = array(zeros(dataSetSize))
	for indx in xrange(dataSetSize):
		distances[indx]=cosdist(testdata,trainSet[indx])
	sortedDistIndicies = argsort(-distances)
	print "sortedDistIndicies:",sortedDistIndicies
	classCount = {}
	for i in range(k):
		voteIlabel = listClasses[sortedDistIndicies[i]]
		print "voteIlabel:",voteIlabel
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
		print "classCount:",classCount
	sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	print "sortedClassCount:",sortedClassCount
	return sortedClassCount[0][0]
	
dataSet,listClasses=loadDataSet()
nb=NBayes()
nb.train_set(dataSet,listClasses)
print classify(nb.tf[3],nb.tf,listClasses,k)