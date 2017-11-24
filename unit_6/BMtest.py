#-*- coding:utf-8 -*-
from numpy import *
from BMNet import *
import copy
import matplotlib.pyplot as plt

bmNet=BoltzmannNet()
bmNet.loadDataSet("dataSet25.txt")
bmNet.train()
print "循环迭代",bmNet.iteration,"次"
print "最优解:",bmNet.bestdist
print "最佳路线:",bmNet.bestpath
bmNet.drawScatter(plt)
bmNet.drawPath(bmNet.bestpath,bmNet.dataMat,plt)
plt.show()

bmNet.TrendLine(plt)
plt.show()