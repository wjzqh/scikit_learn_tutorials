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
		alpha = 20/(1.0+j+i)+0.0001
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
axes1 = plt.subplot(211);axes2 = plt.subplot(212)
X1 = np.linspace(0,lenwl,lenwl)
axes1.plot(X1,-weightmat[:,0]/weightmat[:,2])
axes2.plot(X1,-weightmat[:,1]/weightmat[:,2])
plt.show()