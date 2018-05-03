import numpy as np
from decoder import regression_decoder
import argparse
import heapq
from utils import load_pickle
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=False)
    args = parser.parse_args()
    # assert 'data_processed' not in args.data_dir, 'You should rename your {} to data_processed'.format(args.data_dir)
    scores_clouds = np.load('/home/nathan/Desktop/M01/data_180concepts_pictures.npy')

    max_vxl_scr_clouds = np.amax(scores_clouds, axis=0)
    print(max_vxl_scr_clouds.shape)

    #vxl_id = heapq.nlargest(5000, range(len(max_vxl_scr)), max_vxl_scr.take) order preserved O(klogn)
    stable_vxl_wcl = np.argpartition(max_vxl_scr_clouds, -5000)[-5000:] # O(n) order unpreserved presrved with sort after in O(klogk_+n)

    wcld = load_pickle('/home/nathan/Desktop/emnlp18/data_processed/exp1_proc/M01/data_180concepts_pictures.mat.pkl')


    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    wcld_wds = dict()
    word_dict= dict()
    for word, v in wcld.items():
        wcld_wds[word[0]] = v[stable_vxl_wcl]
    for word, _ in wcld.items():
        word_dict[word[0]] = w2vec_dict[word[0]]

    wd_seq = word_dict.keys()
    train_data = np.zeros((len(wd_seq),5000))
    train_targs = np.zeros((len(wd_seq),300))
    for i,w in enumerate(wd_seq):
        train_data[i,:] = wcld_wds[w]
        train_targs[i,:] = word_dict[w]
    # toy examples
    #wds = np.random.rand(4, 300)

    sum1 = train_data.sum(axis=0)
    for x in range(train_data.shape[1]):
        train_data[:,x]-= sum1[x]
    sum2 = train_data.sum(axis=0)
    for x in range(train_targs.shape[1]):
        train_targs[:, x] -= sum2[x]
    #vxl = np.random.rand(4, 5000)
    weights,l = regression_decoder(train_data,train_targs)
    print(l)
    scores_clouds = np.save('/home/nathan/Desktop/M01/weights_pictures.npy',weights)
    # exp = int((args.data_dir.split('/')[-1]).split('_')[0][-1])
    # assert exp == 1 or exp == 2 or exp == 3
    # assert 'exp' in args.data_dir.split('/')[-1]
    # if exp == 1:
    #     data_gen = load_exp1(args.data_dir)
    #
    # elif exp == 2 or exp == 3:
    #     pass
    # else:
    #     raise ValueError("Illegal value for data folder .Select from{1,2,3}")

