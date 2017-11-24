#-*- coding:utf-8 -*-
from numpy import *
import os

featuremat = mat([[88.5,96.8,104.1,111.3,117.7,124.0,130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7],
[88.5,96.8,104.1,111.3,117.7,124.0,130.0,135.4,140.2,145.3,151.9,159.5,165.9,169.8,171.6,172.3,172.7],
[12.54,14.65,16.64,18.98,21.26,24.06,27.33,30.46,33.74,37.69,42.49,48.08,53.37,57.08,59.35,60.68,61.40]])

mv1 = mean(featuremat[0])
print mv1
mv2 = mean(featuremat[1])
print mv2

cov = mean(multiply(featuremat[0]-mv1,featuremat[1]-mv2))
print cov

std1 = std(featuremat[0])
print std1
aaa = sqrt(mean(multiply(featuremat[0]-mv1, featuremat[0]-mv1)))
print aaa
std2 = std(featuremat[1])

corref = mean(multiply(featuremat[0]-mv1, featuremat[1]-mv2))/(std1*std2)
print corref
print corrcoef(featuremat)
r = range(10)
type(r)