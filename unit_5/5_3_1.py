import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def drawScatterbyLabel(plt,Input):
	m,n = np.shape(Input)
	target = Input[:,-1]
	for i in xrange(m):
		if target[i] == 0:
			plt.scatter(Input[i,0],Input[i,1],c='blue',marker='o')
		else:
			plt.scatter(Input[i,0],Input[i,1],c='red',marker='s')

def buildMat(dataSet):
	m,n = np.shape(dataSet)
	dataMat = np.zeros((m,n))
	dataMat[:,0] = 1
	dataMat[:,1:] = dataSet[:,:-1]
	return dataMat

def file2matrix(path,delimiter):
	recordlist = []
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	rowlist = content.splitlines()
	recordlist = [map(eval,row.split(delimiter)) for row in rowlist if row.strip()]
	return np.mat(recordlist)
Input = file2matrix("testSet.txt","\t")
target = Input[:,-1]
[m,n] = np.shape(Input)

dataMat = buildMat(Input)
#print dataMat

alpha = 0.01
steps = 5000

weights = np.ones((n,1))
#print weights

def logistic(wTx):
	return 1.0/(1.0+np.exp(-wTx))

weightlist = []
for k in xrange(steps):
	gradient = dataMat*np.mat(weights)
	output = logistic(gradient)
	errors = target-output
	weights = weights + alpha*dataMat.T*errors
	weightlist.append(weights)

drawScatterbyLabel(plt,Input)
X = linspace(-5,5,100)
Ylist = []
lenw = len(weightlist)
for indx in xrange(lenw):
	if indx%20 == 0:
		weight = weightlist[indx]
		Y =-(double(weights[0])+X*(double(weights[1])))/double(weights[2])
		plt.plot(X,Y)
		plt.annotate("hplane:"+str(indx),xy=(X[99],Y[99]))#分类超平面注释
plt.show()
