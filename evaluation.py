import numpy as np
import pandas as pd
from utils import fetch_MEN

w_pics = np.load('./weights/M01/weights_pictures.npy')
w_wcld = np.load('./weights/M01/weights_wordclouds.npy')
w_sents = np.load('./weights/M01/weights_sentences.npy')

x=fetch_MEN()
print(x['X'])
wd_pairs=x['X']
sim_score=x['y']
for x in wd_pairs:
    if word == 'theatre':
        word = 'theater'
    if word == 'harbour':
        word = 'harbor'
    if word == 'colour':
        word = 'color'