#-*- coding:utf-8 -*-
from numpy import *
import sys
import os
import matplotlib.pyplot as plt

def plotscatter(Xmat,Ymat,yHat,plt):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(Xmat.tolist(),Ymat.tolist(),c='blue',marker='o')
	plt.plot(Xmat,yHat,'r')
	plt.show()
	
def loadDataSet(fileName):
	numFeat=len(open(fileName).readline().split('\t'))-1
	X=[];Y=[]
	fr=open(fileName)
	for line in fr.readlines():
		curLine=line.strip().split('\t')
		X.append([float(curLine[i]) for i in xrange(numFeat)])
		Y.append(float(curLine[-1]))
	return X,Y
	
xArr,yArr=loadDataSet("nolinear.txt")
print xArr
print yArr

miu=0.02
k=0.03

xMat=mat(xArr);yMat=mat(yArr).T
testArr=xArr
m,n=shape(xArr)
yHat=zeros(m)
for i in xrange(m):
	weights=mat(eye(m))
	for j in xrange(m):
		diffMat=testArr[i]-xMat[j,:]
		weights[j,j]=exp(diffMat*diffMat.T/(-miu*k**2))
	xTx=xMat.T*(weights*xMat)
	if linalg.det(xTx)!=0.0:
		ws=xTx.I*(xMat.T*(weights*yMat))
		yHat[i]=testArr[i]*ws
	else:
		print "This matrix is singular,cannot do inverse"
		sys.exit(0)
		
plotscatter(xMat[:,1],yMat,yHat,plt)