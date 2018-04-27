import numpy as np
from numpy.linalg import svd

from scipy.stats import pearsonr


def rank_classification(target_vector,candidate_vectors,id):
    '''

    :param target_vector: 1*n
    :param candidate_vectors: m*n
    :param id :integer correct prediction
    :return: rank classification for target in [0,1]
    '''
    corr = []
    cds = candidate_vectors.shape[0]
    for i in range(cds):
        corr.append((pearsonr(candidate_vectors[i,:],target_vector),i))
    corr.sort(key=lambda tup: tup[0])
    pos = 0
    for i,j in enumerate(corr):
        if j[tuple[1]] == id:
            pos=i
            break
    assert pos
    rank_clas = abs(pos - cds)/cds
    return rank_clas

def regression_decoder(train_data,train_targets):
    '''

    :param train_data: #examples x #voxels matrix
    :param train_targets: # examples x #dimensions matrix
    :return:
     weighMatrix -   a #voxels+1 x #dimensions weight matrix
     r -             #dimensions vector with the regularization parameter
                    value for each dimension

    column i of weightMatrix has #voxels weights + intercept (last row)
    for predicting target i
    This function uses an efficient implementation of cross-validation within the
    training set to pick a different optimal value of the
    regularization parameter for each semantic dimension in the target vector
    This function uses kernel ridge regression with a linear
    kernel. This allows us to use the full brain as input features
    because we avoid the large inversion of the voxels/voxels matrix

    '''

    params = [1, .5, 5, 0.1, 10, 0.01, 100, 0.001, 1000, 0.0001, 10000, 0.00001,
              100000, 0.000001, 1000000]
    words = train_data.shape[0]
    emb_dim = train_targets.shape[1]

    h_x = np.ones((train_data.shape[0], train_data.shape[1]+1))
    h_x[:, :-1] = train_data
    train_data = h_x

    cv_err = np.zeros(len(params), emb_dim)
    K = np.matmul(train_data, train_data.T)
    U,D,V = svd(K)
    D = np.eye(U.shape[1], V.shape[0])*D
    for idx,reg_param in enumerate(params):
        dlambda = D + reg_param*np.eye(D.shape[0], D.shape[1])
        dlambdaInv = np.diag(1/np.diag(dlambda))
        klambdainv = np.matmul(np.matmul(V,dlambdaInv), U.T)

        K_p = np.matmul(train_data.T, klambdainv)
        S = np.matmul(train_data, K_p)

        weights = np.matmul(K_p, train_targets)

        # Snorm = repmat(1 - diag(S), 1, train_targets.shape[1])
        Snorm = np.kron(np.ones(1, train_targets.shape[1]), 1-np.diag(S))
        Y_diff = train_targets - np.matmul(train_data, weights)
        Y_diff = np.divide(Y_diff, Snorm)

        cv_err[idx, :]=(1/train_data.shape[0])*np.sum(np.multiply(Y_diff, Y_diff)) # elementwise

    minerridx = np.argmin(cv_err)
    # minerr = np.amin(cv_err)
    reg_dim = np.zeros(1, train_targets.shape[1])

    for i,j in enumerate(train_targets.shape[1]):

        reg_param = params[minerridx[i]]
        reg_dim[i]=reg_param

        dlambda = D + reg_param*np.eye(D.shape[0], D.shape[1])
        dlambdaInv = np.diag(1/np.diag(dlambda))
        klambdainv = np.matmul(np.matmul(V, dlambdaInv), U.T)

        weights[:,i] = np.matmul(np.matmul(train_data.T, klambdainv), train_targets[:,i])
    return weights,reg_dim