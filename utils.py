import pickle
import os
import string
import numpy as np


TRANS = str.maketrans('', '', string.punctuation.replace('-', ''))


def save_pickle(data,savepath):
    fdr = '/'.join(savepath.strip().split('/')[:-1])
    print(fdr)
    if not os.path.exists(fdr):
        os.makedirs(fdr)
    with open(savepath+'.pkl', 'wb') as fn:
        pickle.dump(data, fn)

def load_pickle(file):

    with open(file, 'rb') as fn:
        data = pickle.load(fn)
    return data

def disc_pr():
    print("***********************************")


def check_list(lst):
    for id,el in enumerate(lst[1:179]):
        if not(int(el)-int(lst[id-1])==1 and int(lst[id+1])-int(el)==1):
            return True
    return False

def extract_sent_embed(sent):

    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    with open('./stimuli/stopwords.txt') as f:
        stp_wds = f.read().splitlines()
    sent = (sent.translate(TRANS)).lower().split(' ')
    sent_proc=[]
    for wd in sent:
        if wd not in stp_wds:
            if '-' in wd:
                sent_proc.extend(wd.split('-'))
            else:
                sent_proc.extend(wd)
    avg_vec=np.zeros((1,300))
    for wd in sent_proc:
        avg_vec += w2vec_dict[wd]

    avg_vec/=len(sent_proc)

def load_data_meta(data_tuple):
    data = dict()
    meta = dict()
    data_cleared = dict()
    for fl in data_tuple:
        if 'meta' in fl:
            meta = load_pickle(fl)
        else:
            data = load_pickle(fl)
    assert data,meta
    for k,v in data.items():
        data_cleared[k[0]] = v

    return data_cleared,meta