import errno
import numpy as np
import scipy.io as scio
import sys
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from utils import load_pickle, disc_pr, load_data_meta, extract_sent_embed
import argparse
import glob
import string
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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


def parse_meta1(meta):
    voxels2neigh = np.expand_dims(meta['voxelsToNeighbours'], axis=0)[0]
    nneigh = np.expand_dims(meta['numberOfNeighbours'], axis=0)[0]
    return voxels2neigh, nneigh


def load_voxels(mat_file):
    loaded = scio.loadmat(mat_file)
    vxs = parse_voxels(loaded)
    voxels2neigh, nneigh = parse_meta(loaded)
    return vxs, voxels2neigh, nneigh


def search_light(v, nneigh, voxels2neigh):
    neigh = voxels2neigh[v, :nneigh[v]] - 1
    assert np.all(neigh >= 0)
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


def voxel_scores(mri_vectors, semantic_vectors, meta):
    alpha = 1.0
    n_folds = 10
    voxels2neigh, nneigh = parse_meta1(meta)
    semantic_dims = next(iter(semantic_vectors.values())).shape[0]
    num_voxels = voxels2neigh.shape[0]
    num_concepts = len(mri_vectors.keys())
    scores = np.zeros((semantic_dims, num_voxels))
    for v in range(num_voxels):
        print('Voxel {} of {}'.format(v, num_voxels))
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
            dot_sum += (y_pred_z * y_cv_test[fold]).sum(axis=0)
        scores[:, v] = dot_sum / float(num_concepts)
    return scores

#def calcu

def load_exp1(data_dir):
    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    exp_id = int((data_dir.split('/')[-2]).split('_')[0][-1])
    assert exp_id == 1
    fld = data_dir
    # Run one participant
    data_files = sorted(glob.glob(fld + '/*'))
    dt_fls_grouped = [tuple(data_files[i:i + 2]) for i in
                      range(0, len(data_files), 2)]
    print(fld)
    disc_pr()

    # for every file wordcloud pictures and sentences cases
    for data_group in dt_fls_grouped:
        data_dict, metadata = load_data_meta(data_group)
        word_dict = dict()
        for word, _ in data_dict.items():
            word_dict[word] = w2vec_dict[word]
        yield data_group[0], data_dict, word_dict, metadata


def load_avg_exp1(data_dir):
    avg_dict = {}
    data_group = None
    word_dict = {}
    meta = {}
    for dg, data_dict, wd, metadata in load_exp1(data_dir):
        for k, v in data_dict.items():
            if k in avg_dict:
                avg_dict[k] += v
            else:
                avg_dict[k] = v
        word_dict = wd
        meta = metadata
        data_group = dg
    for k, v in avg_dict.items():
        avg_dict[k] /= 3.0
    return data_group, avg_dict, word_dict, meta
        

def load_exp23(data_dir):
    exp_id = int((data_dir.split('/')[-2]).split('_')[0][-1])
    assert exp_id == 2 or exp_id == 3
    fld = data_dir
    # Run one participant
    data_files = sorted(glob.glob(fld + '/*'))
    dt_fls_grouped = [tuple(data_files[i:i + 2]) for i in
                      range(0, len(data_files), 2)]

    # for every file here data and meta
    for data_group in dt_fls_grouped:
        print('\t{}'.format(data_group))
        data_dict, metadata = load_data_meta(data_group)
        word_dict = dict()
        for sent, _ in data_dict.items():
            word_dict[sent] = extract_sent_embed(sent)
        yield data_group[0], data_dict, word_dict, metadata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=True)
    args = parser.parse_args()
    print(args.data_dir)
    # assert 'data_processed' not in args.data_dir, 'You should rename your {} to data_processed'.format(args.data_dir)

    exp = int((args.data_dir.split('/')[-2]).split('_')[0][-1])
    assert exp == 1 or exp == 2 or exp == 3
    assert 'exp' in args.data_dir.split('/')[-2]
    if exp == 1:
        #data_gen = load_exp1(args.data_dir)
        #disc_pr()
        # how to access a generator silly boy :*
        #for x in data_gen:
        #    print(x[0])
        #    out_file = os.path.join('./', 'voxels_scores', '{}.npy'.format(x[0].split('.')[0]))
        #    out_dir = '/'.join(out_file.split('/')[:-1])
        #    mkdir_p(out_dir)
        #    vscores = voxel_scores(x[1], x[2], x[3])
        #    np.save(out_file, vscores)
        data_group, data_dict, word_dict, meta = load_avg_exp1(args.data_dir)
        out_file = os.path.join('./', 'voxels_scores_avg', '{}.npy'.format(data_group.split('.')[0]))
        out_dir = '/'.join(out_file.split('/')[:-1])
        mkdir_p(out_dir)
        vscores = voxel_scores(data_dict, word_dict, meta)
        np.save(out_file, vscores)
    elif exp == 2 or exp == 3:
        load_exp23(args.data_dir)
    else:
        raise ValueError("Illegal value for data folder .Select from{1,2,3}")
