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
axes1 = plt.subplot(211)
axes2 = plt.subplot(212)
weightmat = mat(zeros((steps,n)))
i = 0
for weight in weightlist:
	weightmat[i,:] = weight.T
	i+=1
X = linspace(0,steps,steps)
axes1.plot(X[0:10],-weightmat[0:10,1]/weightmat[0:10,2],color='blue',linewidth=1,linestyle="-")
axes2.plot(X[10:],-weightmat[10:,1]/weightmat[10:,2],color='blue',linewidth=1,linestyle="-")
plt.show()