import theano
import theano.tensor as T
import numpy as np

XWB = [[3,1],[-1,-2],[-1,5]]
X = np.array(XWB[0],dtype=theano.config.floatX)
W = np.array(XWB[1],dtype=theano.config.floatX)
B = np.array(XWB[2],dtype=theano.config.floatX)

logX = T.log(X)
print logX.eval()
meanW = T.mean(W)
print meanW.eval()
neqW = T.neq(W, B)
print neqW.eval()

x = T.dscalar('x')
y = x**2+10
grady = T.grad(y,x)
f = theano.function([x],grady)
print f(100)

Y_pred = T.nnet.softmax(T.dot(X,W)+B)
print Y_pred.eval()