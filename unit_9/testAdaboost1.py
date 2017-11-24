#-*- encoding:utf-8 -*-

from numpy import *
import sys
from adaboostlib import *

dataArr,labelArr = loadDataSet('horseColicTraining.txt')
#print "dataArr:",dataArr
#print "labelArr:",labelArr
weakClassArr,aggClassEst=adaBoostTrain(dataArr,labelArr,numIt=10)
print "weakClassArr:",weakClassArr
print "aggClassEst:",aggClassEst