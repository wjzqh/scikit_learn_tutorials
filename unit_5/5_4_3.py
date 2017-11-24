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

alphalist = []
alphahlist = []
for j in xrange(steps):
	dataIndex = range(m)
	for i in xrange(m):
		alpha = 2/(1.0+j+i)+0.0001
		if j==0:alphalist.append(alpha)
		if i==0:alphahlist.append(alpha)
		randIndex = int(random.uniform(0,len(dataIndex)))

lenal = len(alphalist);lenalh = len(alphahlist)
#fig = plt.figure()
axes1 = plt.subplot(211);axes2 = plt.subplot(212)
X1 = np.linspace(0,lenal,lenal);X2 = np.linspace(0,lenalh,lenalh)
axes1.plot(X1,alphalist);axes2.plot(X2,alphahlist)
plt.show()