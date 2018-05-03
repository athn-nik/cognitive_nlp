import numpy as np
from decoder import regression_decoder
import argparse
import heapq
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=False)
    args = parser.parse_args()
    # assert 'data_processed' not in args.data_dir, 'You should rename your {} to data_processed'.format(args.data_dir)
    scores_sents = np.load('/home/nathan/Desktop/M01/data_180concepts_sentences.npy')
    scores_pics = np.load('/home/nathan/Desktop/M01/data_180concepts_pictures.npy')
    scores_clouds = np.load('/home/nathan/Desktop/M01/data_180concepts_wordclouds.npy')

    max_vxl_scr_pics=np.amax(scores_pics, axis=0)
    max_vxl_scr_sents = np.amax(scores_sents, axis=0)
    max_vxl_scr_clouds = np.amax(scores_clouds, axis=0)
    print(max_vxl_scr_clouds.shape)
    #vxl_id = heapq.nlargest(5000, range(len(max_vxl_scr)), max_vxl_scr.take) order preserved O(klogn)
    stable_vxl = np.argpartition(scores, -5000)[-5000:] # O(n) order unpreserved presrved with sort after in O(klogk_+n)
    sys.exit(0)
    # toy examples
    wds = np.random.rand(4, 300)
    vxl = np.random.rand(4, 5000)
    weights,l = regression_decoder(vxl,wds)

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

