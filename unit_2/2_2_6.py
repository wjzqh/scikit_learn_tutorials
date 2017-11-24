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

#1.分词内容持久化
bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])

def readfile(path):
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

wordbag_path = "test_word_bag/test_set.dat"
seg_path = "test_word_bag/"

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
file_obj = open(wordbag_path,"wb")
pickle.dump(bunch,file_obj)
file_obj.close()


def readbunchobj(path):
	file_obj = open(path,"rb")
	bunch = pickle.load(file_obj)
	file_obj.close()
	return bunch
	
def writebunchobj(path,bunchobj):
	file_obj = open(path,"wb")
	pickle.dump(bunchobj,file_obj)
	file_obj.close()
	
#2.导入分词后的词向量Bunch对象
path = "test_word_bag/test_set.dat"
bunch = readbunchobj(path)
#3.构建测试集TF-IDF向量空间
testspace = Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,tdm=[],vocabulary={})
#4.导入训练集的词袋
space_path = "train_word_bag/tfidfspace.dat"
trainbunch = readbunchobj(space_path)
def readfile(path):
	fp = open(path, 'rb')
	content = fp.read()
	fp.close()
	return content

#5.使用TfidfVectorizer初始化向量空间模型
stopword_path = "train_word_bag/百度停用词列表.txt"
stpwrdlst = readfile(stopword_path).splitlines()
vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,vocabulary=trainbunch.vocabulary)
transformer = TfidfTransformer()

testspace.tdm= vectorizer.fit_transform(bunch.contents)
testspace.vocabulary = trainbunch.vocabulary
#创建词袋的持久化
space_path = "test_word_bag/testspace.dat"
writebunchobj(space_path,testspace)

