#-*- coding:utf-8 -*-
import os
import numpy as np
import sys
import operator
from numpy import *
sys.path.append(r'E:\scikit_learn\unit_5')
from common_libs import *
import matplotlib.pyplot as plt
from compiler.ast import flatten


Input = file2matrix("testSet.txt","\t")
#drawScatterbyLabel(plt,Input)
target = Input[:,-1]
target = flatten(target.tolist())

[m,n] = shape(Input)

dataMat = buildMat(Input)

steps = 500
weights = ones((n,1))

weightlist = []
for j in xrange(steps):
	dataIndex = range(m)
	for i in xrange(m):
		alpha = 2/(1.0+j+i)+0.0001
		randIndex = int(random.uniform(0,len(dataIndex)))
		vectSum = sum(dataMat[randIndex]*weights.T)
		grad = logistic(vectSum)
		errors = target[randIndex] - grad
		weights = weights + alpha*dataMat[randIndex].reshape(n,1)*errors
		del(dataIndex[randIndex])
	weightlist.append(weights)
lenwl = len(weightlist)
weightmat = zeros((lenwl,n))
i = 0
for weight in weightlist:
	weightmat[i,:] = list(weight)
	i+=1
fig = plt.figure()
axes1 = plt.subplot(311)
axes2 = plt.subplot(312)
axes3 = plt.subplot(313)
X = np.linspace(0,lenwl,lenwl)
axes1.plot(X,weightmat[:,0],color='blue',linewidth=1,linestyle="-")
axes1.set_ylabel('weight[0]')
axes2.plot(X,weightmat[:,1],color='red',linewidth=1,linestyle="-")
axes2.set_ylabel('weight[1]')
axes3.plot(X,weightmat[:,2],color='green',linewidth=1,linestyle="-")
axes3.set_ylabel('weight[2]')
plt.show()