""" exec1_2
"""
import matplotlib.pyplot as mpl
import numpy as np

NZ = []
X1 = []
X2 = []
U = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
V = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]

#np.random.normal(loc=0,scale=1)
for i in range(0, 100):
    # 10 Gaussian random numbers fo z's
    NZ.append(np.random.normal(0, 1, 10))
    # Scalar product of two cectors
    X1.append(np.dot(U, NZ[i]))
    X2.append(np.dot(V, NZ[i]))

X = range(1, 101)

#mpl.plot(X, X1, 'mo', X, X2, 'bo')
#mpl.xlim([0,101])
#mpl.legend(['x1', 'x2'])
#mpl.xlabel('N')
#mpl.ylabel('x')

mpl.plot(X1, X2, 'bo')
mpl.xlabel('x1')
mpl.ylabel('x2')
mpl.show()
