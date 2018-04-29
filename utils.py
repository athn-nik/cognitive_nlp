import pickle
import os

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