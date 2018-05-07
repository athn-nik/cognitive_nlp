import numpy as np
from decoder import regression_decoder
import argparse
import heapq
from utils import load_pickle,save_pickle
from sklearn.preprocessing import StandardScaler


def evaluation(i1,p1,i2,p2,metric='pearson'):
 #print("Cosine Similarity Calculation...")
  #Normalize vectors
    '''i1[:]= [x*x for x in i1]
    magni1=np.sum(i1)
    i1[:]= [x/magni1 for x in i1]
    i2[:]= [x*x for x in i2]
    magni2=np.sum(i2)
    i2[:]= [x/magni2 for x in i2]
    p1[:]= [x*x for x in p1]
    magnp1=np.sum(p1)
    p1[:]= [x/magnp1 for x in p1]
    p2[:]= [x*x for x in p2]
    magnp2=np.sum(p2)
    p2[:]= [x/magnp2 for x in p2]'''
 #print(spatial.distance.cosine(p1,i2),spatial.distance.cosine(p2,i1),spatial.distance.cosine(i2,p2),spatial.distance.cosine(i1,p1))
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '-data_dir', dest="data_dir", required=False)
    args = parser.parse_args()
    # assert 'data_processed' not in args.data_dir, 'You should rename your {} to data_processed'.format(args.data_dir)
    fdrs=[1,2,3,4,5,6,7,8,9]
    for fld in fdrs:
        data_dict = dict()
        data_file1 = load_pickle('/home/nathan/Desktop/emnlp18/data_processed/exp1_proc/P01/data_180concepts_sentences.mat.pkl')
        data_file2 = load_pickle('/home/nathan/Desktop/emnlp18/data_processed/exp1_proc/P01/data_180concepts_wordclouds.mat.pkl')
        data_file3 = load_pickle('/home/nathan/Desktop/emnlp18/data_processed/exp1_proc/P01/data_180concepts_pictures.mat.pkl')

        data_dict1 = dict()
        for word, v in data_file1.items():
            data_dict1[word[0]] = v
        data_dict2 = dict()
        for word, v in data_file2.items():
            data_dict2[word[0]] = v
        data_dict3 = dict()
        for word, v in data_file3.items():
            data_dict3[word[0]] = v

        for word, v in data_dict3.items():
            data_dict[word] = (data_dict1[word]+data_dict2[word]+data_dict3[word])/3

        scores = np.load('/home/nathan/Desktop/voxels_scores/data_processed/exp1_proc/P01/data_180concepts_wordclouds.npy')


        max_vxl_scr = np.amax(scores, axis=0)

        vxl_id = heapq.nlargest(5000, range(len(max_vxl_scr)), max_vxl_scr.take) # order preserved O(klogn)
        #stable_vxl_wcl = np.argpartition(max_vxl_scr_clouds, -5000)[-5000:] # O(n) order unpreserved presrved with sort after in O(klogk_+n)


        save_pickle(data_dict,'/home/nathan/Desktop/final_data/exp1/P01/data_180concepts_average')
    sys.exit()
    word_dict= dict()
    w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    for word, _ in wcld.items():
        word_dict[word[0]] = w2vec_dict[word[0]]
    wd_seq = word_dict.keys()
    train_data = np.zeros((len(wd_seq),5000))
    train_targs = np.zeros((len(wd_seq),300))
    for i,w in enumerate(wd_seq):
        train_data[i,:] = wcld_wds[w]
        train_targs[i,:] = word_dict[w]
    # toy examples
    #wds = np.random.rand(4, 300)


    # Normalization of data across different dimensions

    sum1 = train_data.sum(axis=0)
    for x in range(train_data.shape[1]):
        train_data[:,x]-= sum1[x]
    sum2 = train_data.sum(axis=0)
    for x in range(train_targs.shape[1]):
        train_targs[:, x] -= sum2[x]
    #vxl = np.random.rand(4, 5000)
    weights,l = regression_decoder(train_data,train_targs)
    print(l)
    scores_clouds = np.save('/home/nathan/Desktop/M01/weights_pictures.npy',weights)
    # exp = int((args.data_dir.split('/')[-1]).split('_')[0][-1])
    # assert exp == 1 or exp == 2 or exp == 3
    # assert 'exp' in args.data_dir.split('/')[-1]
    # if exp == 1:
    #     data_gen = load_exp1(args.data_dir)
    #
    # elif exp == 2 or exp == 3:
    #     pass
    # else:
    #     raise ValueError("Illegal value for data folder .Select from{1,2,3}")

