#==============================================================================
# theano.scan_module.until
#	theano.scan_module.until必须放在函数的return语句里，当条件满足时，scan停止迭代。
#==============================================================================
import theano
import theano.tensor as T
import numpy

print 'theano.scan_module.until:'
def prod_2(pre_value, max_value):
    return pre_value*2, theano.scan_module.until(pre_value*2 > max_value)

max_value = T.iscalar('max_value')
result, update = theano.scan(prod_2, outputs_info=T.constant(1.),
                             non_sequences=[max_value], n_steps=100)

prod_fn = theano.function([max_value], result, updates=update)
print prod_fn(400)