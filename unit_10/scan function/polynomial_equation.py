#==============================================================================
# polynomial -- c0*x^0 + c1*x^1 + c2*x^2 + c3*x^3...
#==============================================================================
import theano
import theano.tensor as T
import numpy

coefficients=T.vector('coeff')
x = T.iscalar('x')
sum_poly_init = T.fscalar('sum_poly')
result, update = theano.scan(lambda coefficients, power, sum_poly, x: T.cast(sum_poly +     
                             coefficients*(x**power),dtype=theano.config.floatX),
                             sequences=[coefficients, T.arange(coefficients.size)],
                            outputs_info=[sum_poly_init],
                            non_sequences=[x])

poly_fn = theano.function([coefficients,sum_poly_init,x], result, updates=update)

coeff_value = numpy.asarray([1.,3.,6.,5.], dtype=theano.config.floatX)
x_value = 3
poly_init_value = 0.
print poly_fn(coeff_value,poly_init_value, x_value)