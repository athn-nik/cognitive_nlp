import glob
import scipy.io as spio
import argparse
from utils import disc_pr,check_list,save_pickle



def read_data_e1(data_dir):

    main_dir = glob.glob(data_dir+'*/*')
    for fl in main_dir:
        print("Participant id is: ",fl.strip().split('/')[-2])
        print(fl)
        ff = spio.loadmat(fl,squeeze_me=True)
        disc_pr()
        print(ff.keys())
        disc_pr()
        assert check_list(ff['labelsConcept']), "False ordered data"

        if 'data' in fl.split('/')[-1]:
            ff['labelsPOS']=[ff['keyPOS'][x-1] for x in ff['labelsPOS']]
            pos = ff['labelsPOS']
            wds = ff['keyConcept']
            vxl = ff['examples']
            cnc = ff['labelsConcreteness']
            mtd = ff['meta']
            data_dict={}
            data_dict_meta={}
            print(mtd.item()[8])
            print(mtd.dtype)


            for el in ff['labelsConcept']:
                id=el-1
                data_dict[(wds[id],pos[id],cnc[id])]=vxl[id]
                data_dict_meta[wds[id]]=ff['meta']
            save_pickle(data_dict,fl)
# data
# dict_keys(['__header__', '__version__', '__globals__', 
# 'labels_task', 'labelsConcept', 'keyConcept', 'labelsPOS',
#  'keyPOS', 'labelsConcreteness', 'meta', 'examples', 
#  'examples_task', 'tstats_task'])

#  ict_keys(['__header__', '__version__',
#   '__globals__', 'labelsConcept', 'keyConcept',
#    'labelsConcreteness', 'meta', 'examples'])
def read_data_e2(data_dir):

    main_dir = glob.glob(data_dir+'*/*')
    for fl in main_dir:
        print("Participant id is: ",fl.strip().split('/')[-2])
        if 'example' in fl.split('/')[-1]:
            ff = spio.loadmat(fl,squeeze_me=True)
            print(ff.keys())
            disc_pr()
            sents = ff['keySentences']
            
            part_topic_id = ff['labelsPassageForEachSentence']
            topic_id = ff['labelsPassageCategory']
            topics = ff['keyPassageCategory']
            part_of_topics =ff['keyPassages']
            vxl = ff['examples']
            print(part_topic_id,topic_id)
            topic_id = [x for x, number in zip(topic_id, len(topic_id)*[4]) for _ in range(number)]
            print(part_of_topics,topics)
            print(topic_id)
            data_dict={}
            for id,el in enumerate(part_topic_id):
                print((sents[id],part_of_topics[el-1],topics[topic_id[id]-1]))
                data_dict[(sents[id],part_of_topics[el-1],topics[topic_id[id]-1])]=vxl[id]
            #data_dict_meta[wds[id]]=ff['meta']
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
            print(ff.keys())
            disc_pr()
            sents = ff['keySentences']

            print(ff['keyPassageCategory'])
            topics = ff['keyPassages']
            vxl = ff['examples']
            print(ff['keyPassages'])
            print(ff['keySentences'])

            print(ff['labelsPassageCategory'])
            print(ff['labelsPassageForEachSentence'])
            print(len(ff['labelsPassageForEachSentence']))

            data_dict = {}
            for id,el in enumerate(sents):
                data_dict[(sents[id], topics[id])]=vxl[id]

            
        disc_pr()
       


if __name__ == '__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','-data_dir',dest="data_dir",required=True)
    parser.add_argument('-e',"--experiment",dest="exp",type=int,required=True)
    args = parser.parse_args()
    print("I am reading the files from the directory ",args.data_dir)
    if args.exp == 1:
        read_data_e1(args.data_dir)
    elif args.exp == 2:
        read_data_e2(args.data_dir)
    elif args.exp == 3:
        read_data_e3(args.data_dir)
    else :
        raise ValueError("Illegal value for -e args.Select from{1,2,3}")