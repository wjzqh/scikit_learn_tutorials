#-*- coding:utf-8 -*-
import sys
import os
from sklearn.datasets.base import Bunch
import cPickle as pickle
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

reload(sys)
sys.setdefaultencoding("utf-8")

def readbunchobj(path):
	file_obj = open(path,"rb")
	bunch = pickle.load(file_obj)
	file_obj.close()
	return bunch
	
def writebunchobj(path,bunchobj):
	file_obj = open(path,"wb")
	pickle.dump(bunchobj,file_obj)
	file_obj.close()
	
path = "train_word_bag/train_set.dat"
bunch = readbunchobj(path)

tfidfspace = Bunch(target_name=bunch.target_name,label=bunch.label,filename=bunch.filenames,tdm=[],vocabulary={})

def readfile(path):
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

stopword_path = "train_word_bag/百度停用词列表.txt"
stpwrdlst = readfile(stopword_path).splitlines()
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
transformer = TfidfTransformer()
tfidfspace.tdm= vectorizer.fit_transform(bunch.contents)
print tfidfspace.tdm
tfidfspace.vocabulary = vectorizer.vocabulary_

space_path = "train_word_bag/tfidfspace.dat"
writebunchobj(space_path,tfidfspace)

print "结束！！！"


