#-*- coding:utf-8 -*-
import os
import numpy as np
import sys
import operator
from numpy import *
sys.path.append(r'E:\scikit_learn\unit_5')
from common_libs import *
import matplotlib.pyplot as plt


Input = file2matrix("testSet.txt","\t")
target = Input[:,-1]
[m,n] = shape(Input)

dataMat = buildMat(Input)

alpha = 0.001
steps = 500
weights = ones((n,1))
weightlist = []

for k in xrange(steps):
	gradient = dataMat*mat(weights)
	output = logistic(gradient)
	errors = target-output
	weights = weights + alpha*dataMat.T*errors
	weightlist.append(weights)

print weights
fig = plt.figure()
axes1 = plt.subplot(311)
axes2 = plt.subplot(312)
axes3 = plt.subplot(313)
weightmat = mat(zeros((steps,n)))
i = 0
for weight in weightlist:
	weightmat[i,:] = weight.T
	i+=1
X = linspace(0,steps,steps)
axes1.plot(X,weightmat[:,0],color='blue',linewidth=1,linestyle="-")
axes1.set_ylabel('weight[0]')
axes2.plot(X,weightmat[:,1],color='red',linewidth=1,linestyle="-")
axes2.set_ylabel('weight[1]')
axes3.plot(X,weightmat[:,2],color='green',linewidth=1,linestyle="-")
axes3.set_ylabel('weight[2]')
plt.show()