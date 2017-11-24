#-*- coding:UTF-8 -*-
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from Kohonen import *

SOMNet = Kohonen()
SOMNet.loadDataSet("dataset2.txt");
#m,n=shape(SOMNet.dataMat)
#for i in xrange(m):
	#plt.scatter(SOMNet.dataMat[i,0],SOMNet.dataMat[i,1],c='blue',marker='o')
	#plt.plot(SOMNet.dataMat[i,0],SOMNet.dataMat[i,1],c='red',marker='^')
plt.show()
SOMNet.train()
print SOMNet.w
SOMNet.showCluster(plt)