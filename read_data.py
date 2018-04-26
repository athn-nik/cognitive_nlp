import glob
import scipy.io as spio
import argparse
from utils import disc_pr,check_list,save_pickle
import random
import numpy as np
from itertools import groupby



def read_data_e1(data_dir):

    main_dir = glob.glob(data_dir+'*/*')
    for fl in main_dir:
        # print("Participant id is: ",fl.strip().split('/')[-2])
        ff = spio.loadmat(fl,squeeze_me=True)
        ff_nv2 = spio.loadmat(fl,squeeze_me=False)
        disc_pr()
        print(ff.keys())
        disc_pr()
        assert check_list(ff['labelsConcept']), "False ordered data"
        disc_pr()
        disc_pr()
        mtd = ff_nv2['meta']
        print(mtd.dtype)
        disc_pr()
        disc_pr()
        disc_pr()
        if 'data' in fl.split('/')[-1]:
            ff['labelsPOS']=[ff['keyPOS'][x-1] for x in ff['labelsPOS']]
            pos = ff['labelsPOS']
            wds = ff['keyConcept']
            vxl = ff['examples']
            cnc = ff['labelsConcreteness']
            data_dict={}
            #print(mtd.dtype)
            print(len(mtd.item()))
            #data_dict_meta = wds:mtd
            # print(wds)
            # print(vxl)
            for el in ff['labelsConcept']:
                id=el-1
                data_dict[(wds[id],pos[id],cnc[id])]=vxl[id]
                #print((wds[id],pos[id],cnc[id]))
            # save_pickle(data_dict,fl)
            # save_pickle(mtd,fl)


def read_data_e2(data_dir):

    main_dir = glob.glob(data_dir+'*/*')
    print(main_dir)
    for fl in main_dir:
        # print("Participant id is: ",fl.strip().split('/')[-2])
        if 'example' in fl.split('/')[-1]:
            ff = spio.loadmat(fl,squeeze_me=True)
            ff_2 = spio.loadmat(fl,squeeze_me=False)
            print(ff.keys())
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
                #print((sents[id],part_of_topics[el-1],topics[topic_id[id]-1]))
                data_dict[(sents[id],part_of_topics[el-1],topics[topic_id[id]-1])]=vxl[id]
            #data_dict_meta[wds[id]]=ff['meta']
        
        
        # (Sentence,subtopic(Apple),topic(Fruit)): voxels

        #save_pickle(data_dict,fl)
        disc_pr()
        #assert check_list(ff['labelsConcept']), "False ordered data"

        # if 'data' in fl.split('/')[-1]:
        #     ff['labelsPOS']=[ff['keyPOS'][x-1] for x in ff['labelsPOS']]
        #     pos = ff['labelsPOS']
        #     wds = ff['keyConcept']
        #     vxl = ff['examples']
        #     cnc = ff['labelsConcreteness']
        #     mtd = ff['meta']
        #     data_dict={}
        #     data_dict_meta={}
        #     print(mtd.item()[8])
        #     print(mtd.dtype)


        #     for el in ff['labelsConcept']:
        #         id=el-1
        #         data_dict[(wds[id],pos[id],cnc[id])]=vxl[id]
        #         data_dict_meta[wds[id]]=ff['meta']

def read_data_e3(data_dir):

    main_dir = glob.glob(data_dir+'*/*')
    
    for fl in main_dir:
        print("Participant id is: ",fl.strip().split('/')[-2])
        if 'example' in fl.split('/')[-1]:
            ff = spio.loadmat(fl,squeeze_me=True)
            ff_v2 = spio.loadmat(fl,squeeze_me=True)

            print(ff.keys())
            disc_pr()
            sents = ff['keySentences']

            print(ff['keyPassageCategory'])
            topics = ff['keyPassages']
            vxl = ff['examples']
            mtd = ff_v2['meta']
            print(ff['keyPassages'])
            print(ff['keySentences'])

            print(ff['labelsPassageCategory'])
            print(ff['labelsPassageForEachSentence'])

            data_dict = {}
            i=1
            sents_processed = len(sents)*['']
            sen_lbl = ff['labelsPassageForEachSentence'].tolist()
            print(len(ff['labelsPassageCategory']))
            zipped = list(zip(list(set(sen_lbl)),ff['labelsPassageCategory'].tolist()))
            disc_pr()
            print(sen_lbl)
            disc_pr()
            freq = [sen_lbl.count(key) for key in list(set(sen_lbl))]
            final_list_lbls = []
            print(freq)
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

        disc_pr()
       


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','-data_dir',dest="data_dir",required=True)
    parser.add_argument('-e',"--experiment",dest="exp",type=int,required=True)
    args = parser.parse_args()
    # print("I am reading the files from the directory ",args.data_dir)
    if args.exp == 1:
        read_data_e1(args.data_dir)
    elif args.exp == 2:
        read_data_e2(args.data_dir)
    elif args.exp == 3:
        read_data_e3(args.data_dir)
    else :
        raise ValueError("Illegal value for -e args.Select from{1,2,3}")