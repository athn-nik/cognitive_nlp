import numpy as np
import pandas as pd
from utils import fetch_MEN,fetch_embeds
from scipy.stats import spearmanr,pearsonr,ttest_ind,ttest_rel
from scipy import spatial
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from time import time
from utils import save_pickle,load_pickle
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import manifold

import sys
import glob

data_men=fetch_MEN()

# fetch_WS353(which="similarity")
# fetch_SimLex999()
#fetch_RW()
# fetch_RG65()

men_wds = data_men['X'].tolist()
men_scores = data_men['y'].tolist()
men_scores = [y for x in men_scores for y in x]

s_v=5000
emb_dim=300

#men_voc =list(set([item for sublist in train_wds for item in sublist]+[item for sublist in test_wds for item in sublist]))

glove_model=load_pickle('../word_embeddings/glove300d.42B.MEN.pkl')

data_dir = sys.argv[1]
n_neighbors = None
n_components = None
method = None
for part in ["M01" ,"M02", "M03", "M04" ,"M05", "M06" ,"M07", "M08" ,"M09" ,"M10", "M13" ,"M14" ,"M15" ,"M16" ,"M17" ,"P01"]:

    weights_lst = glob.glob(data_dir + '/' + part + '/weights/*')

    print("Participant ID : ",part)

    for wt in weights_lst:

        print("Similarity for : ", wt)

        weights_extracted=np.load(wt)


        neural_data1=np.zeros((len(men_wds),s_v+1))
        neural_data2=np.zeros((len(men_wds),s_v+1))

        test_data_glove_1 = np.zeros((len(men_wds), 300))
        test_data_glove_2 = np.zeros((len(men_wds), 300))
        real = []

        i=0

        for i,x in enumerate(men_wds):
            e1_t = glove_model[x[0]]
            e2_t = glove_model[x[1]]

            test_data_glove_1[i, :] = np.array(e1_t).reshape(300)
            test_data_glove_2[i, :] = np.array(e2_t).reshape(300)

            pred_1_t = np.dot(weights_extracted,e1_t)
            pred_2_t = np.dot(weights_extracted,e2_t)

            neural_data1[i, :] = pred_1_t.reshape(s_v+1)
            neural_data2[i, :] = pred_2_t.reshape(s_v+1)
            real.append(float(men_scores[i]))

        if n_neighbors is not None and N-components is not None and method is not None:
            neural_data1 = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                                method=method).fit_transform(neural_data1)

            neural_data2 = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                                method=method).fit_transform(neural_data2)

            test_data_glove_1 = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                                                method=method).fit_transform(test_data_glove_1)
            test_data_glove_2 = manifold.LocallyLinearEmbedding(n_neighbors, n_components, eigen_solver='auto',
                                                                method=method).fit_transform(test_data_glove_2)

        sum1 = 0
        estimated_similarity = []
        bsl = []

        for i in range(test_data_glove_1.shape[0]):
            est_sim = 1 - spatial.distance.cosine(neural_data1[i, :], neural_data2[i, :])
            estimated_similarity.append(est_sim)


        for i in range(test_data_glove_1.shape[0]):
            bsl_sim = 1 - spatial.distance.cosine(test_data_glove_1[i,:], test_data_glove_2[i,:])
            bsl.append(bsl_sim)

        c = 0

        for i in range(len(real)):
            if abs(estimated_similarity[i]-real[i])<abs(bsl[i]-real[i]):
                c += 1
        print(c*1.0/len(men_wds))
        real = [(float(i)-min(real)) / (max(real)-min(real)) for i in real]
        estimated_similarity = [(float(i)-min(estimated_similarity)) / (max(estimated_similarity)-min(estimated_similarity)) for i in estimated_similarity]

        real_low = [x for x in real if x<0.1]
        real_low_index = [real.index(x) for x in real if x<0.1]
        real_high = [x for x in real if x>0.85]
        real_high_index = [real.index(x) for x in real if x>0.85]

        estima_low = [estimated_similarity[idx] for idx in real_low_index]
        estima_high = [estimated_similarity[idx] for idx in real_high_index]
        bsl_low = [bsl[idx] for idx in real_low_index]
        bsl_high = [bsl[idx] for idx in real_high_index]
        real = (np.array(real)).reshape(len(men_wds),1)
        estimated_similarity = (np.array(estimated_similarity)).reshape(len(men_wds),1)
        bsl = (np.array(bsl)).reshape(len(men_wds),1)

        # Calculation of scores
        # All dataset
        ########################################################################
        ########################################################################
        neural_corr = spearmanr(estimated_similarity,real)[0]
        neural_corr_low = spearmanr(estima_low, real_low)[0]
        neural_corr_high =spearmanr(estima_high, real_high)[0]
        print("Neural correlation: ", neural_corr)
        print("Neural correlation LOW: ", neural_corr_low)
        print("Neural correlation HIGH: ", neural_corr_high)

        text_corr=spearmanr(bsl,real)[0]
        text_corr_low = spearmanr(bsl_low, real_low)[0]
        text_corr_high = spearmanr(bsl_high, real_high)[0]
        print("Text-derived correlation: ", text_corr)
        print("Text-derived correlation LOW: ", text_corr_low)
        print("Text-derived correlation HIGH: ", text_corr_high)


        # sum_pear+=res2[0]
        # sum_spear+=res
