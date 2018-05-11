import os
import pickle
import sys
import utils
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


EXP1_PARTS = ["M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09",
              "M10", "M13", "M14", "M15", "M16", "M17", "P01"]
EXP2_PARTS = ["M02", "M04", "M07", "M08", "M09", "M14", "M15", "P01"]
EXP3_PARTS = ["M02", "M03", "M04", "M07", "M15", "P01"]

data_dir = sys.argv[1]
exp = sys.argv[2]
voxel_type = sys.argv[3]

assert exp in ['exp2', 'exp3']
assert voxel_type in ['average', 'pictures', 'sentences', 'wordclouds']
glove_embeddings = utils.load_pickle('./stimuli/word2vec.pkl')

if exp == 'exp2':
    sentences_file = 'examples_384sentences.pkl'
    EXP_PARTS = EXP2_PARTS
elif exp == 'exp3':
    EXP_PARTS = EXP3_PARTS
else:
    print('Invalid experiment {}'.format(exp))
    sys.exit(1)

participant_results = {}

for part in EXP_PARTS:
    print("Participant ID : ", part)
    weights_dir = os.path.join(data_dir, 'exp1', part, 'weights')
    weights_file = os.path.join(weights_dir,
                                'weights_{}.npy'.format(voxel_type))
    weights = np.load(weights_file)
    sentence_embeddings_file = os.path.join(data_dir, exp, part, sentences_file)
    sentence_embeddings = utils.load_pickle(sentence_embeddings_file)
    result_list = []
    avg_sp_voxels = 0
    avg_pear_voxels = 0
    num_elem = 0
    for sentence, sentence_from_mri in sentence_embeddings.items():
        sentence_from_glove = utils.extract_sent_embed(
            sentence.strip())
        sentence_from_voxels = np.dot(weights, sentence_from_glove)[:-1]
        sp_voxels = spearmanr(sentence_from_mri, sentence_from_voxels)
        avg_sp_voxels += sp_voxels[0]
        pear_voxels = pearsonr(sentence_from_mri, sentence_from_voxels)
        pear_voxels += pear_voxels[0]
        d = {
            'sentence': sentence,
            'sentence_mri': sentence_from_mri,
            'sentence_glove': sentence_from_glove,
            'sentence_voxels': sentence_from_voxels,
            'spearman_voxels_mri': sp_voxels,
            'pearson_voxels_mri': pear_voxels
        }
        num_elem += 1
        result_list.append(d)
    participant_results[part] = {
        'all': result_list,
        'avg_spearman_voxels_mri': avg_sp_voxels / float(num_elem),
        'avg_pearson_voxels_mri': avg_pear_voxels / float(num_elem)
    }
    print('{} voxels - sentence Spearman {}'.format(
        part,
        participant_results[part]['avg_spearman_voxels_mri']))
    print('{} voxels - sentence Pearson {}'.format(
        part,
        participant_results[part]['avg_pearson_voxels_mri']))

with open('{}_{}_composisionality_results'.format(exp, voxel_type), 'wb') as fd:
    pickle.dump(participant_results, fd)


"""
data_dir = sys.argv[1]
for part in ["M01" ,"M02", "M03", "M04" ,"M05", "M06" ,"M07", "M08" ,"M09" ,"M10", "M13" ,"M14" ,"M15" ,"M16" ,"M17" ,"P01"]:
    weights_lst = glob.glob(data_dir + '/' + part + '/weights/*')
    print("Participant ID : ",part)
    for wt in weights_lst:

        weights_extracted=np.load(wt)

        '''
        load embeddings
        use extract_sent_embed from utils.py
        it also filters the sentence
        '''

        '''
        construct representation of sentences
        1.load weights
        2.multiply weights with word vector
        3.average embeddings or other measures i.e.weighted average or multiplicative
        '''

        '''
        4.correlate with real
        '''

        '''
        baseline (do the same with glove)
        '''
        '''
        3-4 compositionality measures
        '''
"""
