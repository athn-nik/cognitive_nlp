w2vec_dict = load_pickle('./stimuli/word2vec.pkl')

data_dir = sys.argv[1]
for part in ["M01" ,"M02", "M03", "M04" ,"M05", "M06" ,"M07", "M08" ,"M09" ,"M10", "M13" ,"M14" ,"M15" ,"M16" ,"M17" ,"P01"]:
    weights_lst = glob.glob(data_dir + '/' + part + '/weights/*')
    print("Participant ID : ",part)
    for wt in weights_lst:

        weights_extracted=np.load(wt)

        '''
        load embeddings
        '''
        '''
        construct representation of sentences
        '''
        '''
        correlate with real
        '''
        '''
        baseline (do the same with glove)
        '''
        '''
        3-4 compositionality measures
        '''