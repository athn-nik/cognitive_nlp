import argparse
import glob
from utils import load_pickle,disc_pr
import numpy as np

def _exp1(data_dir):
    main_dir = glob.glob(data_dir + '/*/*')

    for fl in main_dir:
        data = load_pickle(fl)
        k, v = data.popitem()
        print(v)
        sys.exit()
        np.zeros((len(data),len(v)))

        for key, value in data.items():

        disc_pr()


def _exp2(data_dir):
    main_dir = glob.glob(data_dir + '/*/*')

    for fl in main_dir:
        data = load_pickle(fl)
        k, v = data.popitem()
        print(v)
        sys.exit()
        np.zeros((len(data), len(v)))

        for key, value in data.items():

        disc_pr()


def _exp3(data_dir):
    main_dir = glob.glob(data_dir + '/*/*')

    for fl in main_dir:
        data = load_pickle(fl)
        k, v = data.popitem()
        print(v)
        sys.exit()
        np.zeros((len(data), len(v)))

        for key, value in data.items():

        disc_pr()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','-data_dir',dest="data_dir",required=True)
    args = parser.parse_args()
    # print("I am reading the files from the directory ",args.data_dir)
    #print(data_dir.split['/'])
    assert (args.data_dir).strip().split('/')[1] == 'data_processed', 'You should rename your data_directory to data_processed'

    exp = int((args.data_dir).split('/')[-1][-1])
    assert 'exp' in (args.data_dir).split('/')[-1]

    if  exp == 1:
        _exp1(args.data_dir)
    elif exp == 2:
        _exp2(args.data_dir)
    elif exp == 3:
        _exp3(args.data_dir)
    else :
        raise ValueError("Illegal value for data folder .Select from{1,2,3}")