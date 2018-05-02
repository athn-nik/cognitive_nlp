import numpy as np
from decoder import regression_decoder
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=False)
    args = parser.parse_args()
    # assert 'data_processed' not in args.data_dir, 'You should rename your {} to data_processed'.format(args.data_dir)
    x=np.load('/home/nathan/Desktop/M01/data_180concepts_sentences.npy')
    print(x)
    print(x.shape)
    sys.exit(1)
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

