#-*- coding:utf-8 -*-
import sys
import os
import jieba

reload(sys)
sys.setdefaultencoding('utf-8')

seg_list = jieba.cut("С��1995���ҵ�ڱ����廪��ѧ",cut_all=False)
print "Default Mode:", " ".join(seg_list) #Ĭ���з�

seg_list = jieba.cut("С��1995���ҵ�ڱ����廪��ѧ")
print seg_list

seg_list = jieba.cut("С��1995���ҵ�ڱ����廪��ѧ",cut_all=True)
print "Full Mode:", "/ ".join(seg_list)

str = "С��˶ʿ��ҵ���й���ѧԺ�������������ձ�������ѧ����"
seg_list = jieba.cut_for_search(str)
print "search mode:", "/ ".join(seg_list)

for str in seg_list:
  print str