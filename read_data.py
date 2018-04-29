import glob
import scipy.io as spio
import argparse
from utils import disc_pr,check_list,save_pickle
import random
import numpy as np
from itertools import groupby

import sys

def read_data_e1(data_dir):

    main_dir = glob.glob(data_dir+'/*/*')

    for fl in main_dir:
        # print("Participant id is: ",fl.strip().split('/')[-2])
        ff = spio.loadmat(fl,squeeze_me=True)
        ff_nv2 = spio.loadmat(fl,squeeze_me=False)
        assert check_list(ff['labelsConcept']), "False ordered data"
        mtd = ff_nv2['meta']
        # print(mtd.dtype)
        participant = fl.strip().split("/")[-2]
        exp = fl.strip().split("/")[-3]
        print(fl.split('/')[-1])
        if 'data' in fl.split('/')[-1]:
            ff['labelsPOS']=[ff['keyPOS'][x-1] for x in ff['labelsPOS']]
            pos = ff['labelsPOS']
            wds = ff['keyConcept']
            vxl = ff['examples']
            cnc = ff['labelsConcreteness']
            mtd = ff['meta']
            data_dict={}

            for el in ff['labelsConcept']:
                id=el-1
                data_dict[(wds[id],pos[id],cnc[id])]=vxl[id]
                #print((wds[id],pos[id],cnc[id]))
            save_pickle(data_dict,'../data_processed/'+exp+'_proc/'+participant+'/'+fl.strip().split("/")[-1])
            save_pickle(mtd,'../data_processed/'+exp+'_proc/'+participant+'/'+fl.strip().split("/")[-1]+'_meta')


def read_data_e2(data_dir):

    main_dir = glob.glob(data_dir+'/*/*')

    print(main_dir)
    for fl in main_dir:
        # print("Participant id is: ",fl.strip().split('/')[-2])
        participant = fl.strip().split("/")[-2]
        exp = fl.strip().split("/")[-3]
        print(fl.split('/')[-1])

        if 'example' in fl.split('/')[-1]:
            ff = spio.loadmat(fl,squeeze_me=True)
            ff_2 = spio.loadmat(fl,squeeze_me=False)
            disc_pr()
            sents = ff['keySentences']
            
            part_topic_id = ff['labelsPassageForEachSentence']
            topic_id = ff['labelsPassageCategory']
            topics = ff['keyPassageCategory']
            part_of_topics =ff['keyPassages']
            vxl = ff['examples']
            mtd = ff_2['meta']
            topic_id = [x for x, number in zip(topic_id, len(topic_id)*[4]) for _ in range(number)]
            data_dict={}
            for id,el in enumerate(part_topic_id):
                data_dict[(sents[id],part_of_topics[el-1],topics[topic_id[id]-1])]=vxl[id]

        
            # (Sentence,subtopic(Apple),topic(Fruit)): voxels
            save_pickle(data_dict, '../data_processed/' + exp + '_proc/' + participant + '/' + fl.strip().split("/")[-1])
            save_pickle(mtd, '../data_processed/' + exp + '_proc/' + participant + '/' + fl.strip().split("/")[-1] + '_meta')


def read_data_e3(data_dir):

    main_dir = glob.glob(data_dir+'/*/*')
    
    for fl in main_dir:
        print("Participant id is: ",fl.strip().split('/')[-2])
        participant = fl.strip().split("/")[-2]
        exp = fl.strip().split("/")[-3]

        if 'example' in fl.split('/')[-1]:
            ff = spio.loadmat(fl,squeeze_me=True)
            ff_v2 = spio.loadmat(fl,squeeze_me=True)

            disc_pr()
            sents = ff['keySentences']

            vxl = ff['examples']
            mtd = ff_v2['meta']

            sen_lbl = ff['labelsPassageForEachSentence'].tolist()
            zipped = list(zip(list(set(sen_lbl)),ff['labelsPassageCategory'].tolist()))
            freq = [sen_lbl.count(key) for key in list(set(sen_lbl))]
            final_list_lbls = []
            for idx,el in enumerate(zipped):
                for x in range(freq[idx]):
                    final_list_lbls.append(el)
            print(len(final_list_lbls))

            for i,j in enumerate(final_list_lbls):
                final_list_lbls[i]=(sents[i],final_list_lbls[i][0],ff['keyPassageCategory'][final_list_lbls[i][1]-1])
            print(final_list_lbls)
            data_dict={}
            for i,j in enumerate(final_list_lbls):
                data_dict[j] = vxl[i]
            save_pickle(data_dict, '../data_processed/' + exp + '_proc/' + participant + '/' + fl.strip().split("/")[-1])
            save_pickle(mtd, '../data_processed/' + exp + '_proc/' + participant + '/' + fl.strip().split("/")[-1] + '_meta')

        disc_pr()
       


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','-data_dir',dest="data_dir",required=True)
    args = parser.parse_args()
    # print("I am reading the files from the directory ",args.data_dir)
    #print(data_dir.split['/'])
    exp = int((args.data_dir).split('/')[-1][-1])
    assert 'exp' in (args.data_dir).split('/')[-1]

    if  exp == 1:
        read_data_e1(args.data_dir)
    elif exp == 2:
        read_data_e2(args.data_dir)
    elif exp == 3:
        read_data_e3(args.data_dir)
    else :
        raise ValueError("Illegal value for data folder .Select from{1,2,3}")