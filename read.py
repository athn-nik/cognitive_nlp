import argparse
import numpy as np
import itertools
import scipy.io
from scipy.stats.stats import pearsonr
try:
    import cPickle as pickle
except ImportError:
    import pickle

def stable_voxel_selection(calc_st,fmri_data,length,no_stable,trial_ids):
    combs = set(itertools.combinations([0,1,2,3,4,5],2))
    combs = list(combs)
    stab_vox = None
    if (calc_st):
        vox=np.zeros((length,6,60))
        #print(fmri_data_for_trial[tempo[0,:],0])
        stab_score=np.zeros((length))
        for x in range(0,length):#voxel
            sum_vox=0
            for y in range(0,60):#noun
                vox[x,0,y]=fmri_data[trial_ids[y,0],x]
                vox[x,1,y]=fmri_data[trial_ids[y,1],x]
                vox[x,2,y]=fmri_data[trial_ids[y,2],x]
                vox[x,3,y]=fmri_data[trial_ids[y,3],x]
                vox[x,4,y]=fmri_data[trial_ids[y,4],x]
                vox[x,5,y]=fmri_data[trial_ids[y,5],x]
                # compute the correlation
            for z in combs:
                sum_vox+=pearsonr(vox[x,z[0],:],vox[x,z[1],:])[0]
            stab_score[x]=sum_vox/15#no of possible correlations
            #stab_vox=nlargest(500,range(len(stab_score)),stab_score.take)
            stab_vox=np.argsort(stab_score)[::-1][:no_stable]
            np.savetxt('./stable_voxels/st_vox1_' + str(no_stable)+'.txt',stab_vox,fmt='%d')
    else:
        stab_vox=np.loadtxt('./stable_voxels/st_vox1'+'_'+str(no_stable)+'.txt',dtype=int)
        print('I loaded the voxels NOT calculated them!')
    return stab_vox


def deduplicate_ordered(lst):
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]

def main(calc_stability,data_dir,stable_voxels_number):
    parts=1
    mat = scipy.io.loadmat('./data-science-P'+str(parts)+'.mat')
    voxels_number = len(mat['data'][0].item()[0])
    fmri_data_for_trial=np.zeros((360,voxels_number))
    noun_for_trial=[]
    k=0
    j=0
    colToCoord=np.zeros((voxels_number,3))
    coordToCol=np.zeros((mat['meta']['dimx'][0][0][0][0],mat['meta']['dimy'][0][0][0][0],mat['meta']['dimz'][0][0][0][0]))
    colToCoord=mat['meta']['colToCoord'][0][0]
    coordToCol=mat['meta']['coordToCol'][0][0]
    for x in range (0,360):
        fmri_data_for_trial[j,:]=mat['data'][x][0][0]
        noun_for_trial=noun_for_trial+[mat['info']['word'][0][x][0]]
        j+=1

    tempo=np.zeros((60,6),dtype=int)
    nouns = deduplicate_ordered(noun_for_trial)
    assert len(nouns) == 60
    
    for x in nouns:
        tempo[k,:]=[i for i, j in enumerate(noun_for_trial) if j == x]
        k+=1
    
    stable_voxels = stable_voxel_selection(calc_stability,fmri_data_for_trial,voxels_number,stable_voxels_number,tempo)
    fmri_data_proc=np.zeros((60,stable_voxels_number))
    fmri_data_final=np.zeros((60,stable_voxels_number))
    for x in range(0,60):

        fmri_data_proc[x,:] = fmri_data_for_trial[tempo[x,0],stable_voxels]+fmri_data_for_trial[tempo[x,1],stable_voxels]+\
        fmri_data_for_trial[tempo[x,3],stable_voxels]+fmri_data_for_trial[tempo[x,2],stable_voxels]\
        +fmri_data_for_trial[tempo[x,4],stable_voxels]+fmri_data_for_trial[tempo[x,5],stable_voxels]
        
        fmri_data_proc[x,:]/=6

    mean_data=np.sum(fmri_data_proc,axis=0)
    mean_data/=60
    
    #fmri_data_final=np.zeros((60,stable_voxels_number))
    mean_data=np.tile(mean_data,(60,1))
    fmri_data_final=(fmri_data_proc-mean_data)
    noun_image_dict = dict()
    
    for n_id,n in enumerate(nouns):
        noun_image_dict[n] = fmri_data_final[n_id]
    return noun_image_dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir',dest="data_dir",required=True)
    parser.add_argument('-calc_st',dest="calc_st",default=False,required=False)
    parser.add_argument('-st_vox',dest="stable_voxels",default=500,required=False)

    args = parser.parse_args()
    dict_final = main(args.calc_st, args.data_dir, args.stable_voxels)
    with open('word_voxels.p', 'wb') as fd:
        pickle.dump(dict_final, fd)
