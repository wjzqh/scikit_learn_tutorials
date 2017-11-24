#-*- coding:utf-8 -*-
import sys
import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB

reload(sys)
sys.setdefaultencoding("utf-8")

def readbunchobj(path):
	file_obj = open(path,"rb")
	bunch = pickle.load(file_obj)
	file_obj.close()
	return bunch

#导入训练集向量空间
trainpath = "train_word_bag/tfidfspace.dat"
train_set = readbunchobj(trainpath)

#导入测试集向量空间
testpath = "test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

#应用朴素贝叶斯算法
#alha：0.001 alpha越小，迭代次数越多，精度越高
clf = MultinomialNB(alpha = 0.001).fit(train_set.tdm,train_set.label)

#预测分类结果
predicted = clf.predict(test_set.tdm)
total = len(predicted);rate = 0
for flabel,file_name,expct_cate in zip(test_set.label,test_set.filenames,predicted):
	if flabel!=expct_cate:
		rate+=1
		print file_name,":实际类别:",flabel,"-->预测类别:",expct_cate
#精度
print "error_rate:",float(rate)*100/float(total),"%"

