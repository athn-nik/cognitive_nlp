# coding=utf-8
import numpy as np
from decoder import regression_decoder
import argparse
import heapq
from utils import load_pickle,save_pickle
from sklearn.preprocessing import StandardScaler


def evaluation(i1,p1,i2,p2,metric='pearson'):
 #print("Cosine Similarity Calculation...")
  #Normalize vectors
    '''i1[:]= [x*x for x in i1]
    magni1=np.sum(i1)
    i1[:]= [x/magni1 for x in i1]
    i2[:]= [x*x for x in i2]
    magni2=np.sum(i2)
    i2[:]= [x/magni2 for x in i2]
    p1[:]= [x*x for x in p1]
    magnp1=np.sum(p1)
    p1[:]= [x/magnp1 for x in p1]
    p2[:]= [x*x for x in p2]
    magnp2=np.sum(p2)
    p2[:]= [x/magnp2 for x in p2]'''
 #print(spatial.distance.cosine(p1,i2),spatial.distance.cosine(p2,i1),spatial.distance.cosine(i2,p2),spatial.distance.cosine(i1,p1))
    if metric=='cosine':
        bad=2-spatial.distance.cosine(p1,i2)-spatial.distance.cosine(p2,i1)
        good=2-spatial.distance.cosine(i2,p2)-spatial.distance.cosine(i1,p1)
    elif metric=='pearson':
        bad=scipy.stats.pearsonr(p1,i2)+scipy.stats.pearsonr(p2,i1)
        good=scipy.stats.pearsonr(i2,p2)+scipy.stats.pearsonr(i1,p1)
    else:
        print("You have given wrong parameter regarding similarity metric!")
        print("give pearson or cosine")
        sys.exit()
    if (bad<=good):
        return 1
    else :
        return 0




if __name__ == '__main__':
    '''
    
    Example run :
    
    python train_decoder.py - i /path/to/final_data/exp_id/M01/data_180concepts_wordclouds.pkl
    
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=True)
    args = parser.parse_args()

    data_structure = '/'.join((args.data_dir).split('/')[-4:-2])
    assert  'final_data/exp1' or 'final_data/exp3' or 'final_data/exp2' == data_structure not in args.data_dir, \
           'You should rename your {} to data_processed'.format(args.data_dir)

    # load data
    data_dict = load_pickle(args.data_dir)

    # weights name and path to save them
    if args.data_dir.split('/')[-3][-1]=='1':
        weight_file = '/'.join(args.data_dir.split('/')[:-1])+'/weights_'+(args.data_dir.split('/')[-1]).split('_')[-1].split('.')[0]
        print(weight_file)
    else:
        weight_file = '/'.join(args.data_dir.split('/')[:-1]) + '/weights_sentences'

    # load data and convert to numpy
    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    wd_seq = data_dict.keys()

    train_data = np.zeros((len(wd_seq),5000))
    train_targs = np.zeros((len(wd_seq),300))

    for i,w in enumerate(wd_seq):
        train_data[i,:] = data_dict[w]
        train_targs[i,:] = w2vec_dict[w]

    # Normalization of data across different dimensions
    sum1 = train_data.sum(axis=0)
    for x in range(train_data.shape[1]):
        train_data[:,x]-= sum1[x]
    sum2 = train_data.sum(axis=0)
    for x in range(train_targs.shape[1]):
        train_targs[:, x] -= sum2[x]

    # Train to learn the weights
    weights,l = regression_decoder(train_data,train_targs)
    scores_clouds = np.save(weight_file+'.npy',weights)