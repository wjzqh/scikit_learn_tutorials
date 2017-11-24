#-*- coding:UTF-8 -*-
#Filename:BPTest.py

from numpy import *
import operator

from bpNet import *
import matplotlib.pyplot as plt

bpnet=BPNet()
bpnet.loadDataSet("testSet2.txt")
bpnet.dataMat=bpnet.normalize(bpnet.dataMat)
#print bpnet.dataMat

bpnet.drawClassScatter(plt)

bpnet.bpTrain()

print "bpnet.out_wb:",bpnet.out_wb
print "bpnet.hi_wb:",bpnet.hi_wb

x,z=bpnet.BPClassfier(-3.0,3.0)
bpnet.classfyLine(plt,x,z)
#print "bpnet.errlist:",bpnet.errlist
plt.show()

bpnet.TrendLine(plt)
plt.show()
