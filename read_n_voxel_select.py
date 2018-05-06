import glob
import scipy.io as spio
import argparse
from utils import disc_pr, check_list, save_pickle
import random
import numpy as np
from itertools import groupby

import sys

def load_exp1(data_dir,voxel_dir):
    main_dir = glob.glob(data_dir + '/*')
    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    exp_id = int((data_dir.split('/')[-1]).split('_')[0][-1])
    assert exp_id == 1
    for fld in main_dir:
        # for every participant
        data_files = sorted(glob.glob(fld + '/*'))
        dt_fls_grouped = [tuple(data_files[i:i + 2]) for i in
                          range(0, len(data_files), 2)]
        print(fld)
        disc_pr()

        # for every file wordcloud pictures and sentences cases
        for data_group in dt_fls_grouped:
            data_dict, metadata = load_pickle(data_group)
            word_dict = dict()
            for word, _ in data_dict.items():
                word_dict[word] = w2vec_dict[word]
            yield data_group[0], data_dict, word_dict, metadata


def load_exp23(data_dir,voxel_dir):
    main_dir = glob.glob(data_dir + '/*')

    exp_id = int((data_dir.split('/')[-1]).split('_')[0][-1])
    assert exp_id == 2 or exp_id == 3

    for fld in main_dir:
        # for every participant

        data_files = sorted(glob.glob(fld + '/*'))
        dt_fls_grouped = [tuple(data_files[i:i + 2]) for i in
                          range(0, len(data_files), 2)]
        stable_voxels = np.load()
        # for every file here data and meta
        for data_group in dt_fls_grouped:
            print('\t{}'.format(data_group))
            data_dict, metadata = load_pickle(data_group)
            word_dict = dict()
            for sent, _ in data_dict.items():
                word_dict[sent] = extract_sent_embed(sent)
            return data_group[0], data_dict, word_dict, metadata


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=True)
    parser.add_argument('-v', '-voxel_dir', dest="voxel_dir", required=True)
    args = parser.parse_args()
    # print("I am reading the files from the directory ",args.data_dir)
    # print(data_dir.split['/'])
    exp = int((args.data_dir).split('/')[-1][-1])
    assert 'exp' in (args.data_dir).split('/')[-1]

    if exp == 1:
        read_data_e1(args.data_dir,args.voxel_dir)
    elif exp == 2 or exp ==3:
        read_data_e2(args.data_dir,args.voxel_dir)
    else:
        raise ValueError("Illegal value for data folder .Select from{1,2,3}")