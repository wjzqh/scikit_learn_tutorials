import numpy as np
import matplotlib.pyplot as plt

def drawScatterbyLabel(plt,Input):
	m,n = np.shape(Input)
	target = Input[:,-1]
	for i in xrange(m):
		if target[i] == 0:
			plt.scatter(Input[i,0],Input[i,1],c='blue',marker='o')
		else:
			plt.scatter(Input[i,0],Input[i,1],c='red',marker='s')
	plt.show()

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

drawScatterbyLabel(plt,Input)

dataMat = buildMat(Input)
#print dataMat

alpha = 0.01
steps = 500

weights = np.ones((n,1))
#print weights

def hardlim(dataSet):
	dataSet[np.nonzero(dataSet.A>0)[0]]=1
	dataSet[np.nonzero(dataSet.A<0)[0]]=0
	return dataSet

for k in xrange(steps):
	gradient = dataMat*np.mat(weights)
	output = hardlim(gradient)
	errors = target-output
	weights = weights + alpha*dataMat.T*errors
print errors