import glob
import string
import re, string
import numpy as np
import os

translator = str.maketrans('', '', string.punctuation)
words = []
for file in glob.glob('./*[!.py]'):
    print(file)
    with open(file,'r') as fl:
        for raw_line in fl:
            l = raw_line.strip()
            l = (l.translate(translator)).lower().split(' ')
            words.extend(l)

voc_raw = list(set(words))
vocab = dict()
for item in voc_raw:
    vocab[item] = 1
c=0
print(vocab)

fileSize = os.path.getsize(fileName)

progress = 0
with open('../word_embeddings/glove.42B.300d.txt', 'r') as embds:
    model = {}
    for id,line in enumerate(embds):
        progress = progress + len(line)
        progressPercent = (1.0*progress)/fileSize
        if id % 10000 == 0:
            print(progressPercent)
        splitLine = line.split()
        word = splitLine[0]
        if word in vocab:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
            c+=1
