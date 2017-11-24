#-*- coding:utf-8 -*-

import sys
import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import operator

reload(sys)
sys.setdefaultencoding('utf-8')

def createDataSet():
		group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
		labels = ['A','A','B','B']
		return group,labels

dataSet,labels = createDataSet()
fig = plt.figure()
ax = fig.add_subplot(111)
indx =0
for point in dataSet:
		if labels[indx] == 'A':
			ax.scatter(point[0],point[1],c='blue',marker='o',linewidths=0,s=300)
			plt.annotate("("+str(point[0])+","+str(point[1])+")",xy=(point[0],point[1]))
		else:
			ax.scatter(point[0],point[1],c='red',marker='o',linewidths=0,s=300)
			plt.annotate("("+str(point[0])+","+str(point[1])+")",xy=(point[0],point[1]))
		indx +=1
plt.show()