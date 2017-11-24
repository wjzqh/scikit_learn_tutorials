#-*- coding:utf-8 -*-
import sys
import os
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

seg_list = jieba.cut("小明1995年毕业于北京清华大学",cut_all=False)
print "Default Mode:", " ".join(seg_list) #默认切分

seg_list = jieba.cut("小明1995年毕业于北京清华大学")
print seg_list

seg_list = jieba.cut("小明1995年毕业于北京清华大学",cut_all=True)
print "Full Mode:", "/ ".join(seg_list)

str = "小明硕士毕业于中国科学院计算所，后在日本京都大学深造"
seg_list = jieba.cut_for_search(str)
print "search mode:", "/ ".join(seg_list)

for str in seg_list:
  print str