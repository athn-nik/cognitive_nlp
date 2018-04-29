import glob
import string
import re, string
import numpy as np
import os
import pickle

def save_pickle(data,savepath):
    fdr = '/'.join(savepath.strip().split('/')[:-1])
    print(fdr)
    if not os.path.exists(fdr):
        os.makedirs(fdr)
    with open(savepath+'.pkl', 'wb') as fn:
        pickle.dump(data, fn)
def preprocess_file(filepath):
    translator = str.maketrans('', '', string.punctuation.replace('-',''))
    words = []
    with open(file,'r') as fl:
        for raw_line in fl:
            l = raw_line.strip()
            l = (l.translate(translator)).lower().split(' ')
            print(l)
            for wd in l:
                print(wd)
                if '-' in wd:
                    l.extend(wd.split('-'))
                    print(l)
                    l.remove(wd)
                    print(l)
            words.extend(l)
    voc_raw = list(set(words))
    voc_raw.remove('')
    vocab = dict()
    for item in voc_raw:
        vocab[item] = 1
    return vocab



c=0

fileSize = os.path.getsize('../word_embeddings/glove.42B.300d.txt')

progress = 0
with open('../word_embeddings/glove.42B.300d.txt', 'r') as embds:
    model = {}
    for id,line in enumerate(embds):
        progress = progress + len(line)
        progressPercent = round(100*((1.0*progress)/fileSize),2)
        if id % 10000 == 0:
            print(progressPercent)
        splitLine = line.split()
        word = splitLine[0]
        if word in vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
            c+=1
            voc_raw.remove(word)
save_pickle(model,'./word2vec')
