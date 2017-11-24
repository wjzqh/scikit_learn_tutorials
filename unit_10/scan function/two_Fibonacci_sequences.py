#==============================================================================
# taps vector -- two Fibonacci sequences
#==============================================================================
import theano
import theano.tensor as T
import numpy

Fibo_mat = T.matrix('Fibo_mat')
k = T.iscalar('n_steps')
result, update = theano.scan(lambda tm2,tm1: tm2 + tm1,
                              outputs_info=[dict(initial=Fibo_mat, taps=[-2,-1])],
                            n_steps=k)
Fibo_fn = theano.function([Fibo_mat,k], result, updates=update)
Fibo_init = numpy.asarray([[1,1],[1,2]], dtype=theano.config.floatX)
k_value = 12
print Fibo_fn(Fibo_init, k_value)