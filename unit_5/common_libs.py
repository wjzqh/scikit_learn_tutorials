import numpy as np

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

def logistic(wTx):
	return 1.0/(1.0+np.exp(-wTx))