#!/usr/bin/env python3
'''
Multiple Linear regression to obtain weights for prediction
inputs: semantic features vectors for nine participants
        P1-P9
output: cvi activation of voxel v for intermediate semantic feature i
(1-25) sensor-motor verbs
'''

#from extr_page import noun,sem_feat
import timeit
import scipy.io
from sklearn import linear_model
from scipy.stats.stats import pearsonr
import numpy as np
import itertools
from scipy import spatial
import glob
import sys
import re
from heapq import nlargest
import matplotlib
import random
import numpy as np
from manual_tools import ridge_mod
########################################################################
##############Cosine similarity computation for evaluation##############
########################################################################

if __name__ == '__main__':
#########################################################################
#Start computations for every participant(1-9) for every test pair(1770)#
#########################################################################
 noun=np.loadtxt('noun.txt',dtype=bytes).astype(str)
 stab_vox=[250,300,400,600,700,800]
 for s_v in stab_vox:
  for parts in range(1,10) :
   print("Processing data for Participant "+str(parts))
   mat = scipy.io.loadmat('../data/FMRI/data-science-P'+str(parts)+'.mat')
  #############################################################
  ###############Data Split and merge formatting################
  ##############################################################
   length=len(mat['data'][0].item()[0])
   #trial data are 6x60=360-2x6=348(test words excluded)
   fmri_data_for_trial=np.zeros((360,length))
   noun_for_trial=[]
   k=0
   j=0
   t1=0
   t2=0
   for x in range (0,360):
    fmri_data_for_trial[j,:]=mat['data'][x][0][0]
    noun_for_trial=noun_for_trial+[mat['info']['word'][0][x][0]]
    j+=1
   k=0
   tempo=np.zeros((60,6),dtype=int)
   for x in noun:
    tempo[k,:]=[i for i, j in enumerate(noun_for_trial) if j == x]
    k+=1
   combs=set(itertools.combinations([0,1,2,3,4,5],2))
   combs=list(combs)
  ########################################################################
  #################Voxel Stability Selection Starts#######################
  ########################################################################
   vox=np.zeros((length,6,60))
   stab_score=np.zeros((length))
   for x in range(0,length):#voxel
    sum_vox=0
    for y in range(0,60):#noun
     vox[x,0,y]=fmri_data_for_trial[tempo[y,0],x]
     vox[x,1,y]=fmri_data_for_trial[tempo[y,1],x]
     vox[x,2,y]=fmri_data_for_trial[tempo[y,2],x]
     vox[x,3,y]=fmri_data_for_trial[tempo[y,3],x]
     vox[x,4,y]=fmri_data_for_trial[tempo[y,4],x]
     vox[x,5,y]=fmri_data_for_trial[tempo[y,5],x]
     # compute the correlation
    for z in combs:
     sum_vox+=pearsonr(vox[x,z[0],:],vox[x,z[1],:])[0]
     #sum_vox+=np.pearsonr(vox[x,z[0],:], vox[x,z[1],:])[0, 1]
    stab_score[x]=sum_vox/15#no of possible correlations
   #stab_vox=nlargest(500,range(len(stab_score)),stab_score.take)
   stab_vox=np.argsort(stab_score)[::-1][:s_v]
   np.savetxt('../stable_voxels/st_vox'+str(parts)+'_'+str(s_v)+'.txt',stab_vox,fmt='%d')
