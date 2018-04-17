#!/usr/bin/env python3
'''
Multiple Linear regression to obtain weights for prediction
inputs: semantic features vectors for nine participants
        P1-P9
output: cvi activation of voxel v for intermediate semantic feature i
(1-25) sensor-motor verbs
'''

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

#######################################################################
##############Cosine similarity computation for evaluation##############
########################################################################


def evaluation(i1,p1,i2,p2,metric):
    print("Cosine Similarity Calculation...")

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

    ###################################################################
    ######Load semantic features and handle execution requests#########
    ###################################################################

    sem_feat=np.loadtxt('sem_feat.txt',dtype=float)
    noun=np.loadtxt('noun.txt',dtype=bytes).astype(str)
    test_pairs=set(itertools.combinations(list(range(60)),2))
    test_pairs=list(test_pairs)
    outFile=open("../outputs/bsl_model.txt", 'w') #w for truncating
    #outFile.write("Test Words             Accuracy\n")
    help=re.findall("\d",sys.argv[1])
    if help==[]:
        no_parts=list(range(1,10))
    else:
        no_parts=list(help)
    if (sys.argv[2]=='-tr'):
        train_option=1
    elif (sys.argv[2]=='-notr'):
        train_option=0
    help=float(sys.argv[3])
    var=float(len(test_pairs))
    help=help*var
    test_pairs=test_pairs[0:int(help)]
    #print(help)
    if (len(sys.argv)>4 and sys.argv[4]=='-st_vox'):
        calc_st=1
    elif (len(sys.argv)<=4):
        calc_st=0
    #########################################################################
    #Start computations for every participant(1-9) for every test pair(1770)#
    #########################################################################
    alpha=[]
    for parts in no_parts :
        print("Processing data for Participant "+str(parts))
        mat = scipy.io.loadmat('../data/data-science-P'+str(parts)+'.mat')
        outFile.write("Participant "+str(parts)+"\n")
        outFile.write("Test Words             Cosine similarity\n")
        acc=0
        for test_words in test_pairs:
            rate=100*((test_pairs.index(test_words))/len(test_pairs))
            print("%.1f" % rate,end='\r')
            ##############################################################
            ###############Data Split and merge formatting################
            ##############################################################
            test_1=noun[test_words[0]]
            test_2=noun[test_words[1]]
            #print("Combination of test words are "+str(test_1)+" "+str(test_2))

            #it goes to 2nd trial and accesses i'th voxel
            #trials are 60 concrete nouns*6 times=360
            #extract data and noun for that data from .mat file

            length=len(mat['data'][0].item()[0])
            #trial data are 6x60=360-2x6=348(test words excluded)
            fmri_data_for_trial=np.zeros((348,length))
            fmri_data_raw=np.zeros((360,length))
            noun_for_trial=[]
            test_data1=np.zeros((6,length))
            test_data2=np.zeros((6,length))
            k=0
            j=0
            colToCoord=np.zeros((length,3))
            coordToCol=np.zeros((mat['meta']['dimx'][0][0][0][0],mat['meta']['dimy'][0][0][0][0],\
                                mat['meta']['dimz'][0][0][0][0]))

            colToCoord=mat['meta']['colToCoord'][0][0]
            coordToCol=mat['meta']['coordToCol'][0][0]
            t1=0
            t2=0
            for x in range (0,360):
                fmri_data_raw[k,:]=mat['data'][x][0][0]
                k+=1
                if mat['info'][0][x][2][0]==test_1:
                    test_data1[t1,:]=mat['data'][x][0][0]
                    t1+=1
                elif mat['info'][0][x][2][0]==test_2:
                    test_data2[t2,:]=mat['data'][x][0][0]
                    t2+=1
                else:
                    fmri_data_for_trial[j,:]=mat['data'][x][0][0]
                    noun_for_trial=noun_for_trial+[mat['info']['word'][0][x][0]]
                    j+=1
                k=0
            tempo=np.zeros((58,6),dtype=int)
            for x in noun:
                if ((x!=test_1) and (x!=test_2)):
                    tempo[k,:]=[i for i, j in enumerate(noun_for_trial) if j == x]
                    k+=1
            combs=set(itertools.combinations([0,1,2,3,4,5],2))
            combs=list(combs)

            ########################################################################
            #################Voxel Stability Selection Starts#######################
            ########################################################################

            #print(test_pairs.index(test_words))
            if (calc_st):
                vox=np.zeros((length,6,58))
                fd=open('/home/n_athan/Desktop/diploma/code/stable_voxels/st_vox'+str(parts)+'.pkl','wb')
                #print(fmri_data_for_trial[tempo[0,:],0])
                stab_score=np.zeros((length))
            for x in range(0,length):#voxel
                sum_vox=0
                for y in range(0,58):#noun
                    vox[x,0,y]=fmri_data_for_trial[tempo[y,0],x]
                    vox[x,1,y]=fmri_data_for_trial[tempo[y,1],x]
                    vox[x,2,y]=fmri_data_for_trial[tempo[y,2],x]
                    vox[x,3,y]=fmri_data_for_trial[tempo[y,3],x]
                    vox[x,4,y]=fmri_data_for_trial[tempo[y,4],x]
                    vox[x,5,y]=fmri_data_for_trial[tempo[y,5],x]
                    # compute the correlation
                for z in combs:
                    sum_vox+=pearsonr(vox[x,z[0],:],vox[x,z[1],:])[0]
                stab_score[x]=sum_vox/15#no of possible correlations
                stab_vox=np.argsort(stab_score)[::-1][:500]
                np.savetxt('./stable_voxels/st_vox'+str(parts)+'/'+noun[test_words[0]]+ \
                '_'+noun[test_words[1]]+'.txt',stab_vox,fmt='%d')
            else:
                stab_vox=np.loadtxt('./stable_voxels/st_vox'+str(parts)+'/'+ \
                noun[test_words[0]]+'_'+noun[test_words[1]]+'.txt',dtype=int)
            #################################################################
            ########Data preproccesing and mean normalization################
            #################################################################
            test_data1=np.sum(test_data1,axis=0)
            test_data1/=6
            test_data2=np.sum(test_data2,axis=0)
            test_data2/=6

            fmri_data_proc=np.zeros((58,500))
            fmri_data_final=np.zeros((58,500))
            for x in range(0,58):
                fmri_data_proc[x,:] = fmri_data_for_trial[tempo[x,0],stab_vox] \
                                    +fmri_data_for_trial[tempo[x,1],stab_vox] \
                                    +fmri_data_for_trial[tempo[x,3],stab_vox]+fmri_data_for_trial[tempo[x,2],stab_vox] \
                                    +fmri_data_for_trial[tempo[x,4],stab_vox]+fmri_data_for_trial[tempo[x,5],stab_vox]
                fmri_data_proc[x,:]/=6
            #proc
            mean_data=np.sum(fmri_data_proc,axis=0)+test_data1[stab_vox]+test_data2[stab_vox]
            mean_data/=60
            fmri_data_final=np.zeros((58,500))
            mean_data=np.tile(mean_data,(58,1))
            fmri_data_final=fmri_data_proc-mean_data
            test_data1=test_data1[stab_vox]-mean_data[0,:]
            test_data2=test_data2[stab_vox]-mean_data[0,:]
            test_data1=test_data1.reshape((500,1))
            test_data2=test_data2.reshape((500,1))
            #print(test_data1.shape)
            #print(test_data2.shape)
            #########################################################################
            ##########################Training section###############################
            #########################################################################

            mle_est=np.zeros((500,26))#zeros 25
            semantic=np.zeros((58,25))
            sem_feat=np.array(sem_feat)
            temp=np.ones((60,26))
            temp[:,:-1]=sem_feat
            k=0
            for x in range(60):
            if ((noun[x]!=test_1) and (noun[x]!=test_2)) :
                semantic[k,:]=sem_feat[x,:]
                k+=1
            bias=[]
            #semantic=np.tile(semantic,(58,1)
            #print(semantic.shape)
            if (train_option):
                k=0
                for x in range(500):
                    y=fmri_data_final[:,x]
                    y=y.reshape((58,1))
                    #print(y.shape)
                    res=ridge_mod(y,semantic,[3],0)
                    res=res.reshape((26))
                    mle_est[k,:]=res
                    k+=1
                    #print("got here")
                np.savetxt('./mle_estimates/coeffs'+str(parts)+'.txt',mle_est,fmt='%f')
            else:
                mle_est=np.loadtxt('./mle_estimates/coeffs'+str(parts)+'.txt',dtype=float)

            #######################################################################
            #####################Evaluation section################################
            #######################################################################
            i1=test_data1
            i1=i1.reshape((500,1))
            i2=test_data2
            i2=i2.reshape((500,1))
            #we want to found the noun
            #the noun contained in info
            #mat['info'][0][i][2] contains the word i want for what the trial is set
            idx1=np.where(noun==test_1)
            idx2=np.where(noun==test_2)
            sf1=temp[idx1,:]
            sf2=temp[idx2,:]
            sf1=sf1.reshape((26,1))
            sf2=sf2.reshape((26,1))
            p1=np.dot(mle_est,sf1)
            p2=np.dot(mle_est,sf2)
            acc=acc+evaluation(i1,p1,i2,p2,'cosine')
        accuracy=acc/(len(test_pairs))
        outFile.write("\n"+"Total Accuracy "+str(accuracy)+"alpha = "+str(alpha)+"\n")
    outFile.close()









