#-*- coding:utf-8 -*-
from sklearn.datasets.base import Bunch
import sys
import os
reload(sys)
sys.setdefaultencoding('utf-8')

bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])

def readfile(path):
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

wordbag_path = "train_word_bag/train_set.dat"
seg_path = "train_corpus_seg/"

catelist = os.listdir(seg_path)
bunch.target_name.extend(catelist)
for mydir in catelist:
	class_path = seg_path+mydir+"/"
	file_list = os.listdir(class_path)
	for file_path in file_list:
		fullname = class_path + file_path
		bunch.label.append(mydir)
		bunch.filenames.append(fullname)
		bunch.contents.append(readfile(fullname).strip())
		

import cPickle as pickle
if not os.path.exists("train_word_bag"):
	os.makedirs("train_word_bag")
file_obj = open(wordbag_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()
print "构建文本对象结束！！！"