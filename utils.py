# -*- coding: utf-8 -*-
import pickle
import os
import string
import numpy as np
import tempfile
import pandas as pd
import sys
import hashlib
import errno
import time
import shutil
from sklearn.datasets.base import Bunch

TRANS = str.maketrans('', '', string.punctuation.replace('-', ''))
TEMP = tempfile.gettempdir()

if sys.version_info[0] == 3:
    import pickle
    import io
    import urllib

    _basestring = str
    cPickle = pickle
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    _urllib = urllib
    izip = zip

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string.encode('utf-8'))
        return m.hexdigest()
else:
    import cPickle
    import StringIO
    import urllib
    import urllib2
    import urlparse
    import types
    import itertools

    _basestring = basestring
    cPickle = cPickle
    StringIO = BytesIO = StringIO.StringIO
    izip = itertools.izip

    class _module_lookup(object):
        modules = [urlparse, urllib2, urllib]

        def __getattr__(self, name):
            for module in self.modules:
                if hasattr(module, name):
                    attr = getattr(module, name)
                    if not isinstance(attr, types.ModuleType):
                        return attr
            raise NotImplementedError(
                'This function has not been imported properly')

    module_lookup = _module_lookup()

    class _urllib():
        request = module_lookup
        error = module_lookup
        parse = module_lookup

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string)
        return m.hexdigest()

def movetree(src, dst):
    """Move an entire tree to another directory. Any existing file is
    overwritten"""
    names = os.listdir(src)

    # Create destination dir if it does not exist
    _makedirs(dst)
    errors = []

    for name in names:
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname) and os.path.isdir(dstname):
                movetree(srcname, dstname)
                os.rmdir(srcname)
            else:
                shutil.move(srcname, dstname)
        except (IOError, os.error) as why:
            errors.append((srcname, dstname, str(why)))
        # catch the Error from the recursive movetree so that we can
        # continue with other files
        except Exception as err:
            errors.extend(err.args[0])
    if errors:
        raise Exception(errors)

def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None,
                 initial_size=0, total_size=None, verbose=1):
    """Download a file chunk by chunk and show advancement
    Parameters
    ----------
    response: _urllib.response.addinfourl
        Response to the download request in order to get file size
    local_file: file
        Hard disk file where data should be written
    chunk_size: int, optional
        Size of downloaded chunks. Default: 8192
    report_hook: bool
        Whether or not to show downloading advancement. Default: None
    initial_size: int, optional
        If resuming, indicate the initial size of the file
    total_size: int, optional
        Expected final size of download (None means it is unknown).
    verbose: int, optional
        verbosity level (0 means no message).
    Returns
    -------
    data: string
        The downloaded file.
    """


    try:
        if total_size is None:
            total_size = response.info().get('Content-Length').strip()
        total_size = int(total_size) + initial_size
    except Exception as e:
        if verbose > 1:
            print("Warning: total size could not be determined.")
            if verbose > 2:
                print("Full stack trace: %s" % e)
        total_size = None
    bytes_so_far = initial_size

    # t0 = time.time()
    if report_hook:
        pbar = tqdm(total=total_size, unit="b", unit_scale=True)

    while True:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                # sys.stderr.write('\n')
                pbar.close()
            break

        local_file.write(chunk)
        if report_hook:
            pbar.update(len(chunk)) # This is better because works in ipython
            # _chunk_report_(bytes_so_far, total_size, initial_size, t0)

    if report_hook:
        pbar.close()

    return

def _makedirs(path):  # https://stackoverflow.com/a/600612/223267
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def save_pickle(data,savepath):
    fdr = '/'.join(savepath.strip().split('/')[:-1])
    if not os.path.exists(fdr):
        os.makedirs(fdr)
    with open(savepath+'.pkl', 'wb') as fn:
        pickle.dump(data, fn)

def load_pickle(file):

    with open(file, 'rb') as fn:
        data = pickle.load(fn)
    return data

def disc_pr():
    print("***********************************")


def check_list(lst):
    for id,el in enumerate(lst[1:179]):
        if not(int(el)-int(lst[id-1])==1 and int(lst[id+1])-int(el)==1):
            return True
    return False

def fetch_embeds(wds_list,embed_fl='../word_embeddings/glove.42B.300d.txt'):
    wds_vec=dict()
    if isinstance(wds_list,str):
        wds_list=[wds_list]
    with open(embed_fl,'r') as fl:
        for line in fl:
            wd = line.strip().split()[0]
            vec = [float(x) for x in line.strip().split()[1:]]
            if wd in wds_list:
                wds_vec[wd] = vec
                wds_list.remove(wd)
                if wds_list == []:
                    return wds_vec

    return "Not all words found."


def loadGloveModel(gloveFile='../word_embeddings/glove.42B.300d.txt'):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    return model





def extract_sent_embed(sent, glove_embeddings=None):
    if glove_embeddings is None:
        w2vec_dict = load_pickle('./stimuli/word2vec.pkl')
    w2vec_dict = glove_embeddings
    with open('./stimuli/stopwords.txt') as f:
        stp_wds = f.read().splitlines()
    sent = (sent.translate(TRANS)).lower().split(' ')

    sent_proc=[]
    for wd in sent:
        if wd not in stp_wds:
            if '-' in wd:
                split_words = wd.split('-')
                for w in split_words:
                    sent_proc.append(w)
            else:
                sent_proc.append(wd)
    avg_vec=np.zeros((1,300))
    for wd in sent_proc:
        avg_vec += w2vec_dict[wd]

    #avg_vec/=len(sent_proc)
    avg_vec = avg_vec.reshape((300,))
    return avg_vec

def load_data_meta(data_tuple):
    data = dict()
    meta = dict()
    data_cleared = dict()
    for fl in data_tuple:
        if 'meta' in fl:
            meta = load_pickle(fl)
        else:
            data = load_pickle(fl)
    assert data,meta
    # clear or not? morning call
    for k,v in data.items():
        data_cleared[k[0]] = v

    return data_cleared,meta


def _get_dataset_dir(sub_dir=None, data_dir=None, default_paths=None,
                     verbose=1):
    """ Create if necessary and returns data directory of given dataset.
    Parameters
    ----------
    sub_dir: string
        Name of sub-dir
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    default_paths: list of string, optional
        Default system paths in which the dataset may already have been
        installed by a third party software. They will be checked first.
    verbose: int, optional
        verbosity level (0 means no message).
    Returns
    -------
    data_dir: string
        Path of the given dataset directory.
    Notes
    -----
    This function retrieves the datasets directory (or data directory) using
    the following priority :
    1. defaults system paths
    2. the keyword argument data_dir
    3. the global environment variable WEB_SHARED_DATA
    4. the user environment variable WEB_DATA
    5. web_data in the user home folder
    """
    # We build an array of successive paths by priority
    # The boolean indicates if it is a pre_dir: in that case, we won't add the
    # dataset name to the path.
    paths = []


    # Search given environment variables
    if default_paths is not None:
        for default_path in default_paths:
            paths.extend([(d, True) for d in default_path.split(':')])

    # Check data_dir which force storage in a specific location
    if data_dir is not None:
        paths.extend([(d, False) for d in data_dir.split(':')])
    else:
        global_data = os.getenv('WEB_SHARED_DATA')
        if global_data is not None:
            paths.extend([(d, False) for d in global_data.split(':')])

        local_data = os.getenv('WEB_DATA')
        if local_data is not None:
            paths.extend([(d, False) for d in local_data.split(':')])

        paths.append((os.path.expanduser('~/web_data'), False))

    if verbose > 2:
        print('Dataset search paths: %s' % paths)

    # Check if the dataset exists somewhere
    for path, is_pre_dir in paths:
        if not is_pre_dir and sub_dir:
            path = os.path.join(path, sub_dir)
        if os.path.islink(path):
            # Resolve path
            path = readlinkabs(path)
        if os.path.exists(path) and os.path.isdir(path):
            if verbose > 1:
                print('\nDataset found in %s\n' % path)
            return path

    # If not, create a folder in the first writeable directory
    errors = []
    for (path, is_pre_dir) in paths:
        if not is_pre_dir and sub_dir:
            path = os.path.join(path, sub_dir)
        if not os.path.exists(path):
            try:
                _makedirs(path)
                if verbose > 0:
                    print('\nDataset created in %s\n' % path)
                return path
            except Exception as exc:
                short_error_message = getattr(exc, 'strerror', str(exc))
                errors.append('\n -{0} ({1})'.format(
                    path, short_error_message))

    raise OSError('Web tried to store the dataset in the following '
                  'directories, but:' + ''.join(errors))


def _uncompress_file(file_, delete_archive=True, verbose=1):
    """Uncompress files contained in a data_set.
    Parameters
    ----------
    file: string
        path of file to be uncompressed.
    delete_archive: bool, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True
    verbose: int, optional
        verbosity level (0 means no message).
    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    if verbose > 0:
        print('Extracting data from %s...' % file_)
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        with open(file_, "rb") as fd:
            header = fd.read(4)
        processed = False
        if zipfile.is_zipfile(file_):
            z = zipfile.ZipFile(file_)
            z.extractall(data_dir)
            z.close()
            processed = True
        elif ext == '.gz' or header.startswith(b'\x1f\x8b'):
            import gzip
            gz = gzip.open(file_)
            if ext == '.tgz':
                filename = filename + '.tar'
            out = open(filename, 'wb')
            shutil.copyfileobj(gz, out, 8192)
            gz.close()
            out.close()
            # If file is .tar.gz, this will be handle in the next case
            if delete_archive:
                os.remove(file_)
            file_ = filename
            filename, ext = os.path.splitext(file_)
            processed = True
        if tarfile.is_tarfile(file_):
            with contextlib.closing(tarfile.open(file_, "r")) as tar:
                tar.extractall(path=data_dir)
            processed = True
        if not processed:
            raise IOError(
                    "[Uncompress] unknown archive file format: %s" % file_)
        if delete_archive:
            os.remove(file_)
        if verbose > 0:
            print('   ...done.')
    except Exception as e:
        if verbose > 0:
            print('Error uncompressing file: %s' % e)
        raise

def _get_as_pd(url, dataset_name, **read_csv_kwargs):
    return pd.read_csv(_fetch_file(url, dataset_name, verbose=0), **read_csv_kwargs)

def fetch_MEN(which="all", form="natural"):
    """
    Fetch MEN dataset for testing similarity and relatedness
    ----------
    which : "all", "test" or "dev"
    form : "lem" or "natural"
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores
    Published at http://clic.cimec.unitn.it/~elia.bruni/MEN.html.

    """
    if which == "dev":
        data = _get_as_pd('https://www.dropbox.com/s/c0hm5dd95xapenf/EN-MEN-LEM-DEV.txt?dl=1',
                          'similarity', header=None, sep=" ")
    elif which == "test":
        data = _get_as_pd('https://www.dropbox.com/s/vdmqgvn65smm2ah/EN-MEN-LEM-TEST.txt?dl=1',
                          'similarity/EN-MEN-LEM-TEST', header=None, sep=" ")
    elif which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/b9rv8s7l32ni274/EN-MEN-LEM.txt?dl=1',
                          'similarity', header=None, sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    if form == "natural":
        # Remove last two chars from first two columns
        data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    elif form != "lem":
        raise RuntimeError("Not recognized form argument")

    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(np.float) / 5.0)

def fetch_WS353(which="all"):
    """
        Fetch WS353 dataset for testing attributional and
        relatedness similarity
        Parameters
        ----------
        which : 'all': for both relatedness and attributional similarity,
                'relatedness': for relatedness similarity
                'similarity': for attributional similarity
                'set1': as divided by authors
                'set2': as divided by authors
        References
        ----------
        Finkelstein, Gabrilovich, "Placing Search in Context: The Concept Revisited†", 2002
        Agirre, Eneko et al., "A Study on Similarity and Relatedness Using Distributional and WordNet-based Approaches",
        2009
        Returns
        -------
        data : sklearn.datasets.base.Bunch
            dictionary-like object. Keys of interest:
            'X': matrix of 2 words per column,
            'y': vector with scores,
            'sd': vector of std of scores if available (for set1 and set2)
    """
    if which == "all":
        data = _get_as_pd('https://www.dropbox.com/s/eqal5qj97ajaycz/EN-WS353.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "relatedness":
        data = _get_as_pd('https://www.dropbox.com/s/x94ob9zg0kj67xg/EN-WSR353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "similarity":
        data = _get_as_pd('https://www.dropbox.com/s/ohbamierd2kt1kp/EN-WSS353.txt?dl=1',
                          'similarity', header=None, sep="\t")
    elif which == "set1":
        data = _get_as_pd('https://www.dropbox.com/s/opj6uxzh5ov8gha/EN-WS353-SET1.txt?dl=1',
                          'similarity', header=0, sep="\t")
    elif which == "set2":
        data = _get_as_pd('https://www.dropbox.com/s/w03734er70wyt5o/EN-WS353-SET2.txt?dl=1',
                          'similarity', header=0, sep="\t")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(np.float)

    # We have also scores
    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(np.float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    else:
        return Bunch(X=X.astype("object"), y=y)


def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)
    References
    ----------
    Rubenstein, Goodenough, "Contextual correlates of synonymy", 1965
    Notes
    -----
    Scores were scaled by factor 10/4
    """
    data = _get_as_pd('https://www.dropbox.com/s/chopke5zqly228d/EN-RG-65.txt?dl=1',
                      'similarity', header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float) * 10.0 / 4.0)


def fetch_RW():
    """
    Fetch Rare Words dataset for testing attributional similarity
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores
    References
    ----------
    Published at http://www-nlp.stanford.edu/~lmthang/morphoNLM/.
    Notes
    -----
    2034 word pairs that are relatively rare with human similarity scores. Rare word selection: our choices of
    rare words (word1) are based on their frequencies – based on five bins (5, 10], (10, 100], (100, 1000],
    (1000, 10000], and the affixes they possess. To create a diverse set of candidates, we randomly
    select 15 words for each configuration (a frequency bin, an affix). At the scale of Wikipedia,
    a word with frequency of 1-5 is most likely a junk word, and even restricted to words with
    frequencies above five, there are still many non-English words. To counter such problems,
    each word selected is required to have a non-zero number of synsets in WordNet(Miller, 1995).
    """
    data = _get_as_pd('https://www.dropbox.com/s/xhimnr51kcla62k/EN-RW.txt?dl=1',
                      'similarity', header=None, sep="\t").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float),
                 sd=np.std(data[:, 3:].astype(np.float)))


def fetch_multilingual_SimLex999(which="EN"):
    """
    Fetch Multilingual SimLex999 dataset for testing attributional similarity
    Parameters
    -------
    which : "EN", "RU", "IT" or "DE" for language
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,
    References
    ----------
    Published at http://technion.ac.il/~ira.leviant/MultilingualVSMdata.html.
    Notes
    -----
    Scores for EN are different than the original SimLex999 dataset.
    Authors description:
    Multilingual SimLex999 resource consists of translations of the SimLex999 word similarity data set to
    three languages: German, Italian and Russian. Each of the translated datasets is scored by
    13 human judges (crowdworkers) - all fluent speakers of its language. For consistency, we
    also collected human judgments for the original English corpus according to the same protocol
    applied to the other languages. This dataset allows to explore the impact of the "judgement language"
    (the language in which word pairs are presented to the human judges) on the resulted similarity scores
    and to evaluate vector space models on a truly multilingual setup (i.e. when both the training and the
    test data are multilingual).
    """
    if which == "EN":
        data = _get_as_pd('https://www.dropbox.com/s/nczc4ao6koqq7qm/EN-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "DE":
        data = _get_as_pd('https://www.dropbox.com/s/ucpwrp0ahawsdtf/DE-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "IT":
        data = _get_as_pd('https://www.dropbox.com/s/siqjagyz8dkjb9q/IT-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    elif which == "RU":
        data = _get_as_pd('https://www.dropbox.com/s/3v26edm9a31klko/RU-MSIM999.txt?dl=1',
                          'similarity', header=None, encoding='utf-8', sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    scores = data.values[:, 2:].astype(np.float)
    y = np.mean(scores, axis=1)
    sd = np.std(scores, axis=1)

    return Bunch(X=X.astype("object"), y=y, sd=sd)


def fetch_SimLex999():
    """
    Fetch SimLex999 dataset for testing attributional similarity
    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,
        'conc': matrix with columns conc(w1), conc(w2) and concQ the from dataset
        'POS': vector with POS tag
        'assoc': matrix with columns denoting free association: Assoc(USF) and SimAssoc333
    References
    ----------
    Hill, Felix et al., "Simlex-999: Evaluating semantic models with (genuine) similarity estimation", 2014
    Notes
    -----
     SimLex-999 is a gold standard resource for the evaluation of models that learn the meaning of words and concepts.
     SimLex-999 provides a way of measuring how well models capture similarity, rather than relatedness or
     association. The scores in SimLex-999 therefore differ from other well-known evaluation datasets
     such as WordSim-353 (Finkelstein et al. 2002). The following two example pairs illustrate the
     difference - note that clothes are not similar to closets (different materials, function etc.),
     even though they are very much related: coast - shore 9.00 9.10, clothes - closet 1.96 8.00
    """
    data = _get_as_pd('https://www.dropbox.com/s/0jpa1x8vpmk3ych/EN-SIM999.txt?dl=1',
                      'similarity', sep="\t")

    # We basically select all the columns available
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    return Bunch(X=X.astype("object"), y=y, sd=sd, conc=conc, POS=POS, assoc=assoc)

def _fetch_file(url, data_dir=TEMP, uncompress=False, move=False,md5sum=None,
                username=None, password=None, mock=False, handlers=[], resume=True, verbose=0):
    """Load requested dataset, downloading it if needed or requested.
    This function retrieves files from the hard drive or download them from
    the given urls. Note to developpers: All the files will be first
    downloaded in a sandbox and, if everything goes well, they will be moved
    into the folder of the dataset. This prevents corrupting previously
    downloaded data. In case of a big dataset, do not hesitate to make several
    calls if needed.
    Parameters
    ----------
    dataset_name: string
        Unique dataset name
    resume: bool, optional
        If true, try to resume partially downloaded files
    uncompress: bool, optional
        If true, will uncompress zip
    move: str, optional
        If True, will move downloaded file to given relative path.
        NOTE: common usage is zip_file_id/zip_file.zip together
        with uncompress set to True
    md5sum: string, optional
        MD5 sum of the file. Checked if download of the file is required
    username: string, optional
        Username used for basic HTTP authentication
    password: string, optional
        Password used for basic HTTP authentication
    handlers: list of BaseHandler, optional
        urllib handlers passed to urllib.request.build_opener. Used by
        advanced users to customize request handling.
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    resume: bool, optional
        If true, try resuming download if possible
    verbose: int, optional
        verbosity level (0 means no message).
    Returns
    -------
    files: list of string
        Absolute paths of downloaded files on disk
    """

    # TODO: move to global scope and rename
    def _fetch_helper(url, data_dir=TEMP, resume=True, overwrite=False,
                md5sum=None, username=None, password=None, handlers=[],
                verbose=1):
        if not os.path.isabs(data_dir):
            data_dir = _get_dataset_dir(data_dir)

        # Determine data path
        _makedirs(data_dir)

        # Determine filename using URL
        parse = _urllib.parse.urlparse(url)
        file_name = os.path.basename(parse.path)
        if file_name == '':
            file_name = md5_hash(parse.path)

        temp_file_name = file_name + ".part"
        full_name = os.path.join(data_dir, file_name)
        temp_full_name = os.path.join(data_dir, temp_file_name)
        if os.path.exists(full_name):
            if overwrite:
                os.remove(full_name)
            else:
                return full_name
        if os.path.exists(temp_full_name):
            if overwrite:
                os.remove(temp_full_name)
        t0 = time.time()
        local_file = None
        initial_size = 0

        try:
            # Download data
            url_opener = _urllib.request.build_opener(*handlers)
            request = _urllib.request.Request(url)
            request.add_header('Connection', 'Keep-Alive')
            if username is not None and password is not None:
                if not url.startswith('https'):
                    raise ValueError(
                        'Authentication was requested on a non  secured URL (%s).'
                        'Request has been blocked for security reasons.' % url)
                # Note: HTTPBasicAuthHandler is not fitted here because it relies
                # on the fact that the server will return a 401 error with proper
                # www-authentication header, which is not the case of most
                # servers.
                encoded_auth = base64.b64encode(
                    (username + ':' + password).encode())
                request.add_header(b'Authorization', b'Basic ' + encoded_auth)
            if verbose > 0:
                displayed_url = url.split('?')[0] if verbose == 1 else url
                print('Downloading data from %s ...' % displayed_url)
            if resume and os.path.exists(temp_full_name):
                # Download has been interrupted, we try to resume it.
                local_file_size = os.path.getsize(temp_full_name)
                # If the file exists, then only download the remainder
                request.add_header("Range", "bytes=%s-" % (local_file_size))
                try:
                    data = url_opener.open(request)
                    content_range = data.info().get('Content-Range')
                    if (content_range is None or not content_range.startswith(
                            'bytes %s-' % local_file_size)):
                        raise IOError('Server does not support resuming')
                except Exception:
                    # A wide number of errors can be raised here. HTTPError,
                    # URLError... I prefer to catch them all and rerun without
                    # resuming.
                    if verbose > 0:
                        print('Resuming failed, try to download the whole file.')
                    return _fetch_helper(
                        url, data_dir, resume=False, overwrite=overwrite,
                        md5sum=md5sum, username=username, password=password,
                        handlers=handlers, verbose=verbose)
                local_file = open(temp_full_name, "ab")
                initial_size = local_file_size
            else:
                data = url_opener.open(request)
                local_file = open(temp_full_name, "wb")
            _chunk_read_(data, local_file, report_hook=(verbose > 0),
                         initial_size=initial_size, verbose=verbose)
            # temp file must be closed prior to the move
            if not local_file.closed:
                local_file.close()
            shutil.move(temp_full_name, full_name)
            dt = time.time() - t0
            if verbose > 0:
                print('...done. (%i seconds, %i min)' % (dt, dt // 60))
        except _urllib.error.HTTPError as e:
            if verbose > 0:
                print('Error while fetching file %s. Dataset fetching aborted.' %
                      (file_name))
            if verbose > 1:
                print("HTTP Error: %s, %s" % (e, url))
            raise
        except _urllib.error.URLError as e:
            if verbose > 0:
                print('Error while fetching file %s. Dataset fetching aborted.' %
                      (file_name))
            if verbose > 1:
                print("URL Error: %s, %s" % (e, url))
            raise
        finally:
            if local_file is not None:
                if not local_file.closed:
                    local_file.close()
        if md5sum is not None:
            if (_md5_sum_file(full_name) != md5sum):
                raise ValueError("File %s checksum verification has failed."
                                 " Dataset fetching aborted." % local_file)
        return full_name

    if not os.path.isabs(data_dir):
        data_dir = _get_dataset_dir(data_dir)


    # There are two working directories here:
    # - data_dir is the destination directory of the dataset
    # - temp_dir is a temporary directory dedicated to this fetching call. All
    #   files that must be downloaded will be in this directory. If a corrupted
    #   file is found, or a file is missing, this working directory will be
    #   deleted.
    parse = _urllib.parse.urlparse(url)
    file_name = os.path.basename(parse.path)

    files_pickle = cPickle.dumps([(file_, url) for file_, url in zip([file_name], [url])])
    files_md5 = hashlib.md5(files_pickle).hexdigest()
    temp_dir = os.path.join(data_dir, files_md5)

    # Create destination dir
    _makedirs(data_dir)

    # Abortion flag, in case of error
    abort = None

    # 2 possibilities:
    # - the file exists in data_dir, nothing to do (we have to account for move parameter here)
    # - the file does not exists: we download it in temp_dir

    # Target file in the data_dir
    target_file = os.path.join(data_dir, file_name)

    # Change move so we always uncompress to some folder (this is important for
    # detecting already downloaded files)
    # Ex. glove.4B.zip -> glove.4B/glove.4B.zip
    if uncompress and not move:
        dirname, _ = os.path.splitext(file_name)
        move = os.path.join(dirname, os.path.basename(file_name))

    if (abort is None
        and not os.path.exists(target_file)
        and (not move or (move and uncompress and not os.path.exists(os.path.dirname(os.path.join(data_dir, move)))))
            or (move and not uncompress and not os.path.exists(os.path.join(data_dir, move)))):

        # Target file in temp dir
        temp_target_file = os.path.join(temp_dir, file_name)
        # We may be in a global read-only repository. If so, we cannot
        # download files.
        if not os.access(data_dir, os.W_OK):
            raise ValueError('Dataset files are missing but dataset'
                             ' repository is read-only. Contact your data'
                             ' administrator to solve the problem')

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        dl_file = _fetch_helper(url, temp_dir, resume=resume,
                              verbose=verbose, md5sum=md5sum,
                              username=username,
                              password=password,
                              handlers=handlers)

        if (abort is None and not os.path.exists(target_file) and not
                os.path.exists(temp_target_file)):
            if not mock:
                warnings.warn('An error occured while fetching %s' % file_)
                abort = ("Dataset has been downloaded but requested file was "
                         "not provided:\nURL:%s\nFile:%s" %
                         (url, target_file))
            else:
                _makedirs(os.path.dirname(temp_target_file))
                open(temp_target_file, 'w').close()

        if move:
            move = os.path.join(data_dir, move)
            move_dir = os.path.dirname(move)
            _makedirs(move_dir)
            shutil.move(dl_file, move)
            dl_file = move
            target_file = dl_file

        if uncompress:
            try:
                if os.path.getsize(dl_file) != 0:
                    _uncompress_file(dl_file, verbose=verbose)
                else:
                    os.remove(dl_file)
                target_file = os.path.dirname(target_file)
            except Exception as e:
                abort = str(e)
    else:
        if verbose > 0:
            print("File already downloaded, skipping")

        if move:
            target_file = os.path.join(data_dir, move)

        if uncompress:
            target_file = os.path.dirname(target_file)

    if abort is not None:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise IOError('Fetching aborted: ' + abort)
    # If needed, move files from temps directory to final directory.
    if os.path.exists(temp_dir):
        # XXX We could only moved the files requested
        # XXX Movetree can go wrong
        movetree(temp_dir, data_dir)
        shutil.rmtree(temp_dir)
    return target_file
