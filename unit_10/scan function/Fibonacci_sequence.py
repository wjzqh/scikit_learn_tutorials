#==============================================================================
# taps scalar -- Fibonacci sequence
#==============================================================================
import theano
import theano.tensor as T
import numpy

Fibo_arr = T.vector('Fibonacci')
k= T.iscalar('n_steps')
result, update = theano.scan(lambda tm2,tm1: tm2 + tm1,
                             outputs_info=[dict(initial=Fibo_arr, taps=[-2,-1])],
                            n_steps=k)
Fibo_fn = theano.function([Fibo_arr,k], result, updates=update)
Fibo_init = numpy.asarray([1,1], dtype=theano.config.floatX)
k_value = 12
print Fibo_fn(Fibo_init, k_value)