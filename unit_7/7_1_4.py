#-*- coding:utf-8 -*-
from numpy import *

def loadDataSet(filename):
	X = [];
	Y = [];
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		X.append(float(curLine[0]));
		Y.append(float(curLine[1]));
	return X,Y
	
xArr,yArr = loadDataSet("regdataset.txt")
#print xArr
#print yArr
m = len(xArr)
Xmat = mat(ones((m,2)))
for i in xrange(m):Xmat[i,1]=xArr[i]
Ymat=mat(yArr).T
print Xmat
print Ymat