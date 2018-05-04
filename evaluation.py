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

w_pics = np.load('./weights/M01/weights_pictures.npy')
w_wcld = np.load('./weights/M01/weights_wordclouds.npy')
w_sents = np.load('./weights/M01/weights_sentences.npy')

wt_lst = [w_pics,w_wcld,w_sents]

data_train=fetch_MEN("dev")
data_test=fetch_MEN("test")

train_wds=data_train['X'].tolist()
train_scores=data_train['y'].tolist()
train_scores = [y for x in train_scores for y in x]


test_wds=data_test['X'].tolist()
test_scores=data_test['y'].tolist()
test_scores = [y for x in test_scores for y in x]

s_v=5000
emb_dim=300
#men_voc =list(set([item for sublist in train_wds for item in sublist]+[item for sublist in test_wds for item in sublist]))

glove_model=load_pickle('../word_embeddings/glove300d.42B.MEN.pkl')


for wt in wt_lst:
    sum_pear=sum_spear=sum_pear_l=sum_spear_l=sum_pear_m=sum_spear_m=sum_pear_h=sum_spear_h=0
    sum_bsl_s=sum_bsl_p=bsl_sum_pear_l=bsl_sum_spear_l=bsl_sum_pear_h=bsl_sum_spear_h=0
    sum_pear_hl=sum_spear_hl=0
    sum_pear_l_fus=sum_spear_l_fus=sum_pear_h_fus=sum_spear_h_fus=sum_pear_hl_fus=sum_spear_hl_fus=0
    sum_pear_fused=sum_spear_fused=0

    weights_extracted=wt
    train_data=np.zeros((len(train_wds),s_v+1))
    train_data1=np.zeros((len(train_wds),s_v+1))
    train_data2=np.zeros((len(train_wds),s_v+1))

    targets=np.zeros((len(train_wds),1))
    i=0
    for i,x in enumerate(train_wds):
        # if x[0] == 'theatre' or x[1] == 'theatre':
        #     word = 'theater'
        # if x[0] == 'harbour' or x[1] == 'harbour':
        #     word = 'harbor'
        # if x[0] == 'colour' or x[1] == 'colour':
        #     word = 'color'
        e1_t=glove_model[x[0]]
        e2_t=glove_model[x[1]]
        #e1_t=e1_t.reshape(emb_dim,1)
        #e2_t=e2_t.reshape(emb_dim,1)

        pred_1_t=np.dot(weights_extracted,e1_t)
        pred_2_t=np.dot(weights_extracted,e2_t)

        diff_t=abs(pred_2_t-pred_1_t)**2
        dist_t=diff_t.reshape(len(diff_t))
        train_data1[i,:]=pred_1_t.reshape(s_v+1)
        train_data2[i,:]=pred_2_t.reshape(s_v+1)
        targets[i,0]=float(train_scores[i])

    train_data = (abs(train_data1-train_data2))**2
    train_data=(StandardScaler(with_mean=True, with_std=True).fit_transform(train_data))

    model = linear_model.Ridge(alpha=1,fit_intercept=True,normalize=True)

    model.fit(train_data,targets)
    mle_est=model.coef_
    #bias=model.intercept_
    #bias=np.array(bias)
    #mle_est=np.append(mle_est,bias)

    sum1=0
    estimated_similarity=[]
    real=[]
    bsl=[]
    test_data=np.zeros((len(test_wds),s_v+1))
    test_data1=np.zeros((len(test_wds),s_v+1))
    test_data2=np.zeros((len(test_wds),s_v+1))
    j=0
    for j,x in enumerate(test_wds):
        e1_te=glove_model[x[0]]
        #e1_te=e1_te.reshape(s_v,1)#26
        e2_te=glove_model[x[1]]
        #e2_te=e2_te.reshape(s_v,1)

        try:
            bsl_pair=1-spatial.distance.cosine(glove_model[x[0]],glove_model[x[1]])
            bsl.append(bsl_pair)
        except KeyError:
            bsl.append(0.5)

        pred_1_te=np.dot(weights_extracted,e1_te)
        pred_2_te=np.dot(weights_extracted,e2_te)


        diff_te=abs(pred_2_te-pred_1_te)**2
        diff_te=np.append(diff_te,1)
        helper=diff_te.reshape(len(diff_te))
        test_data1[j,:]=pred_1_te.reshape(s_v+1)
        test_data2[j,:]=pred_2_te.reshape(s_v+1)

        #test_data[j,:]=helper
        #est_sim=np.dot(diff_te,mle_est)
        #estimated_similarity.append(est_sim)
        real.append(float(test_scores[j]))
    test_data = abs(test_data1-test_data2)**2
    test_data = (StandardScaler(with_mean=True, with_std=True).fit_transform(test_data))
    #predictions=model.predict(test_data)
    #for i in predictions:
        #print(i)
    #    estimated_similarity.append(i[0])
    for i in range(test_data.shape[0]):
        est_sim=np.dot(test_data[i,:],mle_est.T)
        estimated_similarity.append(est_sim[0])
    #print(len(estimated_similarity))

    #bsl=[x[0] for x in bsl]
    c=0

    for i in range(len(real)):
        #print(real[i],estimated_similarity[i],bsl[i])
        if abs(estimated_similarity[i]-real[i])<abs(bsl[i]-real[i]):
            c+=1
    print(c*1.0/len(test_wds))
    real_low=[x for x in real if x<0.1]
    real_low_index=[real.index(x) for x in real if x<0.1]
    real_high=[x for x in real if x>0.85]
    real_high_index=[real.index(x) for x in real if x>0.85]

    estima_low= [estimated_similarity[idx] for idx in real_low_index]
    estima_high= [estimated_similarity[idx] for idx in real_high_index]
    bsl_low= [bsl[idx] for idx in real_low_index]
    bsl_high= [bsl[idx] for idx in real_high_index]
    real=(np.array(real)).reshape(len(test_wds),1)
    estimated_similarity=(np.array(estimated_similarity)).reshape(len(test_wds),1)
    bsl=(np.array(bsl)).reshape(len(test_wds),1)

    # Calculation of scores
    # All dataset
    ########################################################################
    ########################################################################

    res=spearmanr(estimated_similarity,real)[0]
    res2=pearsonr(estimated_similarity,real)[0]
    #bsl=np.array(bsl)
    #
    bsl_s=spearmanr(bsl,real)[0]
    bsl_p=pearsonr(bsl,real)[0]
    print(res,bsl_s)
    print(res2,bsl_p)


    sum_pear+=res2[0]
    sum_spear+=res
