import numpy as np
import scipy.io as scio
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


def load_embeddings(data_file):
    with open(data_file, 'r') as fd:
        lines = fd.readlines()
        words = [l.strip().split()[0] for l in lines]
        d = [map(float, l.strip().split()[1:]) for l in lines]
        dims = len(d[0])
        vec_dict = dict(zip(words, d))
    return vec_dict, dims


def parse_voxels(loaded_mat):
    concepts = loaded_mat['keyConcept'].shape[0]
    return {
        loaded_mat['keyConcept'][c][0][0]: loaded_mat['examples'][c]
        for c in range(concepts)
    }


def parse_meta(loaded_mat):
    voxels2neigh = loaded_mat['meta']['voxelsToNeighbours'][0][0]
    nneigh = loaded_mat['meta']['numberOfNeighbors'][0][0].flatten()
    return voxels2neigh, nneigh


def load_voxels(mat_file):
    loaded = scio.loadmat(mat_file)
    vxs = parse_voxels(loaded)
    voxels2neigh, nneigh = parse_meta(loaded)
    return vxs, voxels2neigh, nneigh


def search_light(v, nneigh, voxels2neigh):
    neigh = voxels2neigh[v, :nneigh[v]]
    return neigh


def create_X_y(v, nneigh, voxels2neigh, semantic_vectors, mri_vectors):
    neigh = search_light(v, nneigh, voxels2neigh)
    X, y = [], []
    for concept in semantic_vectors.keys():
        X.append(mri_vectors[concept][neigh])
        y.append(semantic_vectors[concept])
    X, y = np.array(X), np.array(y)
    return X, y


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


def main(argv):
    # TODO change to argparse
    mat_file = argv[1]
    glove_file = argv[2]
    out_file = argv[3]
    alpha = 1.0
    n_folds = 10
    mri_vectors, voxels2neigh, nneigh = load_voxels(mat_file)
    semantic_vectors, semantic_dims = load_embeddings(glove_file)
    num_voxels = voxels2neigh.shape[0]
    num_concepts = len(mri_vectors.keys())
    scores = np.zeros((semantic_dims, num_voxels))
    for v in range(num_voxels):
        X, y = create_X_y(
            v, nneigh, voxels2neigh,
            semantic_vectors, mri_vectors)
        X_cv_train, y_cv_train, X_cv_test, y_cv_test = split_folds(
            X, y, k=n_folds)
        dot_sum = np.zeros(semantic_dims)
        for fold in range(len(X_cv_train)):
            X_tr = (StandardScaler(with_mean=True, with_std=True)
                    .fit_transform(X_cv_train[fold]))
            y_tr = (StandardScaler(with_mean=True, with_std=True)
                    .fit_transform(y_cv_train[fold]))
            X_te = (StandardScaler(with_mean=True, with_std=True)
                    .fit_transform(X_cv_test[fold]))
            ridge = Ridge(alpha=alpha, normalize=False).fit(X_tr, y_tr)
            y_pred = ridge.predict(X_te)
            y_pred_z = (StandardScaler(with_mean=True, with_std=True)
                    .fit_transform(y_pred))
            dot_sum += y_pred_z * y_cv_test[fold]
        scores[:, v] = dot_sum / float(num_concepts)
    np.save(out_file, scores)


if __name__ == '__main__':
    main(sys.argv)
