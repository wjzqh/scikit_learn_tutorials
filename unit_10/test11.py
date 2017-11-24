#coding=utf-8  
import numpy  
import theano  
import theano.tensor as T  
rng = numpy.random  
  
N = 10     # 我们为了测试，自己生成10个样本，每个样本是3维的向量，然后用于训练  
feats = 3  
D = (rng.randn(N, feats).astype(numpy.float32), rng.randint(size=N, low=0, high=2).astype(numpy.float32)) 
print D 