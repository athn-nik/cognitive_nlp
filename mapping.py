'''
We have
X source space
Y target space
We want the "mapping" matrix
'''

from numpy.linalg import svd
import numpy as np

h = np.matmul(x, y.T, out=None)

U,S,V = svd(h, full_matrices=True, compute_uv=True)
#U,_,V=svd(X*Y')

S = np.eye(U.shape[1], V.shape[0]) * S
#U*V'
tr_m = np.matmul(U, V.T, out=None)