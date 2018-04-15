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
import os

if __name__ == '__main__':
    ###################################################################
    ######Load semantic features and handle execution requests#########
    ###################################################################
    skip_or_cooc='../sem_feat.txt'
    if skip_or_cooc=='sem_feat_glove.txt':
        latent_dims=50
    elif skip_or_cooc=='../sem_feat.txt':
        latent_dims=25
    sem_feat=np.loadtxt(skip_or_cooc,dtype=float)
    noun=np.loadtxt('../noun.txt',dtype=bytes).astype(str)
    #########################################################################
    #Start computations for every participant(1-9) for every test pair(1770)#
    #########################################################################
    #no of voxels to use
    stable_voxels_range=[50,100,150,200,225,250,300,400,500]
    reg_par=[(0.8,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (0.9,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (0.8,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0,2.0),\
    (1.0,2.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (1.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (1.2,1.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0),\
    (290.0,1.0,5.0,10.0,12.0,15.0,20.0,50.0,100.0,150.0,200.0,300.0,400.0,500.0)]
    for stable_voxels in stable_voxels_range:
        for parts in range(1,10):
            print("Processing data for Participant "+str(parts))
            mat = scipy.io.loadmat('/home/n_athan/Desktop/diploma/data/FMRI/data-science-P'+str(parts)+'.mat')
            #it goes to 2nd trial and accesses i'th voxel
            #trials are 60 concrete nouns*6 times=360
            #extract data and noun for that data from .mat file
            print('Data reading & processing starts...')
            length=len(mat['data'][0].item()[0])
            #trial data are 6x60=360-2x6=348(test words excluded)
            fmri_data_for_trial=np.zeros((360,length))
            noun_for_trial=[]
            k=0
            j=0
            colToCoord=np.zeros((length,3))
            coordToCol=np.zeros((mat['meta']['dimx'][0][0][0][0],mat['meta']['dimy'][0][0][0][0],mat['meta']['dimz'][0][0][0][0]))

            colToCoord=mat['meta']['colToCoord'][0][0]
            coordToCol=mat['meta']['coordToCol'][0][0]
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
                print('Data reading & processing ends...')
            ########################################################################
            #################Voxel Stability Selection Starts#######################
            ########################################################################
            print('Voxel Selection starts...')
            #print(test_pairs.index(test_words))
            stab_vox=np.loadtxt('../stable_voxels/st_vox'+str(parts)+'_'+str(stable_voxels)+'.txt',dtype=int)
            print('I loaded the voxels NOT calculated them!')
            #stab_vox=np.loadtxt('./stable_voxels/st_vox'+str(parts)+'/'+noun[test_words[0]]+'_'+noun[test_words[1]]+'.txt',dtype=int)
            print('Voxel Selection ends...')
            #################################################################
            ########Data preproccesing and mean normalization################
            #################################################################
            print('Mean normalization and global representation construction starts...')
            fmri_data_proc=np.zeros((60,stable_voxels))
            fmri_data_final=np.zeros((60,stable_voxels))
            for x in range(0,60):
                fmri_data_proc[x,:] =fmri_data_for_trial[tempo[x,0],stab_vox]+fmri_data_for_trial[tempo[x,1],stab_vox]+fmri_data_for_trial[tempo[x,3],stab_vox]+fmri_data_for_trial[tempo[x,2],stab_vox]+fmri_data_for_trial[tempo[x,4],stab_vox]+fmri_data_for_trial[tempo[x,5],stab_vox]
                fmri_data_proc[x,:]/=6
            mean_data=np.sum(fmri_data_proc,axis=0)
            mean_data/=60
            std=np.std(fmri_data_proc)
            fmri_data_final=np.zeros((60,stable_voxels))
            mean_data=np.tile(mean_data,(60,1))
            fmri_data_final=fmri_data_proc-mean_data
            fmri_data_final/=std
            print('Mean normalization and global representation construction ends...')
            #########################################################################
            ##########################Training section###############################
            #########################################################################
            print('Training starts...')
            mle_est=np.zeros((stable_voxels,latent_dims+1))#zeros 25
            semantic=np.zeros((60,latent_dims))
            sem_feat=np.array(sem_feat)
            temp=np.ones((60,latent_dims+1))
            temp[:,:-1]=sem_feat
            k=0
            for x in range(60):
                semantic[k,:]=sem_feat[x,:]
                k+=1
            bias=[]
            model = linear_model.RidgeCV(reg_par[int(parts-1)],fit_intercept=True,normalize=False)#####Ridge(alpha=0.5)
            model.fit(semantic,fmri_data_final)
            mle_est=model.coef_ #TODO remove [x,:
            bias=model.intercept_
            bias=np.array(bias)
            bias=np.reshape(np.array(bias),(stable_voxels,1))
            mle_est=np.append(mle_est,bias,1)
            vox_folder=str(stable_voxels)+'st_vox'
            if not os.path.exists('../train_all/'+vox_folder):
                os.makedirs('../train_all/'+vox_folder)
            np.savetxt('../train_all/'+vox_folder+'/coeffs'+str(parts)+'.txt',mle_est,fmt='%f')
