#-*- coding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def readfile(path):
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

stopword_path = "train_word_bag/百度停用词列表.txt"
stpwrdlst = readfile(stopword_path).splitlines()
for decode_str in stpwrdlst:
	print decode_str
