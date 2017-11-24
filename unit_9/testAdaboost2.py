#-*- encoding:utf-8 -*-
#Filename:testAdaboost2.py

from numpy import *
import sys
from adaboostlib import *
import matplotlib.pyplot as plt
dataArr,labelArr=loadDataSet('horseColicTraining.txt')
weakClassArr,aggClassEst=adaBoostTrain(dataArr,labelArr,numIt=10)
print "weakClassArr:",weakClassArr
plotROC(aggClassEst.T,labelArr.tolist()[0])

testArr,testLabelArr=loadDataSet('horseColicTest.txt')
ClassEst100=adaClassify(testArr,weakClassArr)
print "ClassEst100:",ClassEst100
errArr=mat(ones((67,1)))
totalError=errArr[ClassEst100!=mat(testLabelArr).T].sum()
print "totalError:",totalError