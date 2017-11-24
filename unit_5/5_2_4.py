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

alpha = 0.001
steps = 5000

weights = np.ones((n,1))
#print weights

def logistic(wTx):
	return 1.0/(1.0+np.exp(-wTx))

for k in xrange(steps):
	gradient = dataMat*np.mat(weights)
	output = logistic(gradient)
	errors = target-output
	weights = weights + alpha*dataMat.T*errors
#print weights

drawScatterbyLabel(plt,Input)
X = np.linspace(-5,5,100)
Y = -(np.double(weights[0])+X*(np.double(weights[1])))/np.double(weights[2])
plt.plot(X,Y)
plt.show()


def classifier(testData,weights):
	prob = logistic(sum(testData*weights))
	if prob > 0.5:
		return 1.0
	else:
		return 0.0

testdata = mat([-0.147324,2.874846])
m,n = shape(testdata)
testmat = zeros((m,n+1))
testmat[:,0] = 1
testmat[:,1:] = testdata
print classifier(testmat,weights)