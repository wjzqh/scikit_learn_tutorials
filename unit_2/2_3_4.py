#-*- coding:utf-8 -*-
import sys
import os
import numpy as np
from Nbayes_lib import *

dataSet,listClasses = loadDataSet()
nb=NBayes()
nb.train_set(dataSet,listClasses)
nb.map2vocab(dataSet[1])
print nb.predict(nb.testset)