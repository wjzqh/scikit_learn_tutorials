#_*_ coding:utf-8 _*_
import sys
import os
from numpy import *

reload(sys)
sys.setdefaultencoding("utf-8")

def file2matrix(path,delimiter):
	recordlist = []
	fp = open(path,"rb")
	content = fp.read()
	fp.close()
	rowlist = content.splitlines()
	recordlist=[map(eval,row.split(delimiter)) for row in rowlist if row.strip()]
	return mat(recordlist)
	
root = "testdata"
pathlist = os.listdir(root)
print pathlist
for path in pathlist:
	recordmat = file2matrix(root+"/"+path,"\t")
	print recordmat
	print shape(recordmat)

root = "testdata1"
import cPickle as pickle
file_obj = open(root+"/recordmat.dat","wb")
pickle.dump(recordmat,file_obj)
file_obj.close()

read_obj = open(root+"/recordmat.dat","rb")
readmat = pickle.load(read_obj)
print readmat
print shape(readmat)	