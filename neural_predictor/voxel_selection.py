import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def extract_meta_from_mat(mat_file):
    loaded = scio.loadmat(mat_file)
    return loaded['meta']


def parse_meta(meta):
    voxels2neigh = meta['voxelsToNeighbours'][0][0]
    nneigh = meta['numberOfNeighbors'][0][0].flatten()
    return voxels2neigh, nneigh


def search_light(v, nneigh, voxels2neigh, alpha):
    neigh = voxels2neigh[v, :nneigh[v]]
    regmat = np.eye(nneigh[v])
    return neigh, regmat


def split_folds(X, y, k=10):
    k_fold = KFold(k)
    X_cv_train = []
    y_cv_train = []
    X_cv_test = []
    y_cv_test = []
    for train_idx, test_idx in k_fold.split(X, y):
        X_cv_train.append(X[train_idx])
        y_cv_train.append(y[train_idx])
        X_cv_test.append(X[test_idx])
        y_cv_test.append(y[test_idx])
    return X_cv_train, y_cv_train, X_cv_test, y_cv_test


def load_embeddings(data_file):
    with open(data_file, 'r') as fd:
        lines = fd.readlines()
        words = [l.strip().split()[0] for l in lines]
        d = [map(float, l.strip().split()[1:]) for l in lines]
        xs = np.array(d)
    return words, xs


def main():
    meta = extract_meta_from_mat()