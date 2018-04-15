import os,sys
import numpy as np
import pickle
'''
idx=0
#sem_feat=[[]]
sem_feat=[[]]*60
noun=np.loadtxt('noun.txt',dtype=bytes).astype(str)
with open("/home/n_athan/Desktop/diploma/code/glove.6B.50d.txt","r") as w2v:
 dictio=pickle.load(w2v)
 #print(dictio)
 for wd in noun:
  print(wd)
  print(dictio[wd])
  sem_feat[idx]=dictio[wd]
  idx+=1
print(len(sem_feat))
print(len(sem_feat[0]))
np.savetxt('sem_feat_glove.txt', np.array(sem_feat,np.dtype(float)), fmt='%f')


'''
def find_in_file(nou):
 with open("/home/n_athan/Desktop/diploma/code/glove.6B.50d.txt","r") as w2v:
  for line in w2v:
   if line.strip().split(" ")[0]==nou:
    help=line.strip().split(" ")[1:]
    vec=[float(x) for x in help]
 return vec 


sem_feat=[[]]*60
nouns=np.loadtxt('noun.txt',dtype=bytes).astype(str)
idx=0
for nou in nouns:
 sem_feat[idx]=find_in_file(nou)
 idx+=1
np.savetxt('sem_feat_glove.txt', np.array(sem_feat,np.dtype(float)), fmt='%f')
