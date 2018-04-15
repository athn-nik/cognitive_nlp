import gensim
from scipy import spatial
import numpy as np
from tqdm import tqdm
cte_no=['airplane', 'ant', 'apartment', 'arch', 'arm', 'barn', 'bear', 'bed', 'bee',
'beetle', 'bell', 'bicycle', 'bottle', 'butterfly', 'car', 'carrot', 'cat', 'celery',
'chair', 'chimney', 'chisel', 'church', 'closet', 'coat', 'corn', 'cow', 'cup', 'desk',
'dog', 'door', 'dress', 'dresser', 'eye', 'fly', 'foot', 'glass', 'hammer', 'hand',
'horse', 'house', 'igloo', 'key', 'knife', 'leg', 'lettuce', 'pants', 'pliers',
'refrigerator', 'saw', 'screwdriver', 'shirt', 'skirt', 'spoon', 'table', 'telephone',
'tomato', 'train', 'truck', 'watch', 'window']



model = gensim.models.KeyedVectors.load_word2vec_format('../utils/embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
# if you vector file is in binary format, change to binary=True
bsl_similarity=np.zeros((len(cte_no),len(cte_no)))
for noun in tqdm(cte_no):
    for index_x,x in enumerate(cte_no[(cte_no.index(noun)+1):]):
        result = 1 - spatial.distance.cosine(model[noun],model[x])
        index_x+=len(cte_no)-len(cte_no[(cte_no.index(noun)+1):])
        bsl_similarity[cte_no.index(noun),index_x]=result
bsl_similarity[range(bsl_similarity.shape[0]), range(bsl_similarity.shape[0])] = 1.0
#convert array appropriately
i_lower = np.tril_indices(len(cte_no), -1)
bsl_similarity[i_lower] = bsl_similarity.T[i_lower]
symmetric=np.allclose(bsl_similarity, bsl_similarity.T, atol=1e-8)
#scipy.io.savemat('./sm_'+sv+'.mat', mdict={'arr': sim_matrix})
# print(symmetric)
print(bsl_similarity)
