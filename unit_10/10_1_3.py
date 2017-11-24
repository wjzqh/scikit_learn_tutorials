#-*- coding:utf-8 -*-
import theano 
import theano.tensor as T
import numpy as np

X=T.matrix("X")
results,updates=theano.scan(lambda x_i:T.sqrt((x_i ** 2).sum()),sequences=[X])
compute_norm_lines=theano.function(inputs=[X],outputs=[results])

x=np.diag(np.arange(1,6,dtype=theano.config.floatX),1)

print compute_norm_lines(x)[0]

print np.sqrt((x ** 2).sum(1))