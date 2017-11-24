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
drawScatterbyLabel(plt,Input)
target = Input[:,-1]
target = flatten(target.tolist())

[m,n] = shape(Input)

dataMat = buildMat(Input)

steps = 500
weights = ones((n,1))

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
print weights
weights = weights.tolist()
print weights
X = np.linspace(-5,5,100)
Y = -(double(weights[0])+X*(double(weights[1])))/double(weights[2])
plt.plot(X,Y)
plt.show()