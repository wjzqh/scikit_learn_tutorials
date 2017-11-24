from numpy import *
import os

a = mat([[1,2,3],[4,5,6]])
b = a.mean(axis=0)
print a-b
print finfo('float').max