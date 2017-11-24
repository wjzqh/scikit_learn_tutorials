import theano
import theano.tensor as T
import numpy as np

row = 3;column = 3

#-------------------定义符号函数代码------------------------#
#1.初始化一个Theano张量
A = theano.shared(
		value = np.ones((row,column),dtype = 'float64'),
		name = 'A',
		borrow = True
)

#使用Numpy array初始化
Xlist = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
B = theano.shared(
		value = np.array(Xlist,dtype = 'float64'),
		name = 'B',
		borrow = True
)
x = T.dmatrix('x')
y = T.dmatrix('y')
out = T.mean(x+y)

myfunc1 = theano.function(
		inputs = [],
		outputs = out,
		givens = [(x,A),(y,B)]
)
#遍历List
C = theano.shared(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype = theano.config.floatX))
idx = T.lscalar('idx')
z = T.vector('z')
f1 = T.sum(z)
myfunc2 = theano.function(
		[idx],
		outputs = f1,
		givens = {z:C[0:idx]})
		
#-------------------------输出函数运行结果-----------------------#
print A.get_value()
print np.shape(B.get_value())
print myfunc1()
print myfunc2(4)