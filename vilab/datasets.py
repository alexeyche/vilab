
import os, shutil, re, string, urllib, fnmatch
from os.path import join as pj
from env import Env, make_dir
import logging
import numpy as np
import gzip, zipfile, tarfile
import pickle as pkl
import cPickle as cPkl


def _get_datafolder_path():
	env = Env()	
	return env.dataset()

def _unpickle(f):
    fo = open(f, 'rb')
    d = cPkl.load(fo)
    fo.close()
    return d


def _download_mnist_binarized(datapath):
    """
    Download the fized binzarized MNIST dataset if it is not present.
    :return: The train, test and validation set.
    """
    datafiles = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/"
                 "datasets/binarized_mnist/binarized_mnist_train.amat",
        "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                 "binarized_mnist/binarized_mnist_valid.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                "binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
        logging.info("Downloading {} data into {}".format(split, datapath))
        local_file = datapath + '/binarized_mnist_%s.npy'%(split)
        datasplits[split] = np.loadtxt(urllib.urlretrieve(datafiles[split])[0])

    f = gzip.open(datapath +'/mnist.pkl.gz', 'w')
    pkl.dump([datasplits['train'],datasplits['valid'],datasplits['test']],f)


def _download_mnist_realval(dataset):
    origin = (
        "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz"
    )
    logging.info("Downloading data from {}".format(origin))
    urllib.urlretrieve(origin, dataset)

def load_mnist_realval(dataset=_get_datafolder_path()+'/mnist_real/mnist.pkl.gz'):
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_realval(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f)
    f.close()
    x_train, targets_train = train_set[0], train_set[1]
    x_valid, targets_valid = valid_set[0], valid_set[1]
    x_test, targets_test = test_set[0], test_set[1]
    return x_train, targets_train, x_valid, targets_valid, x_test, targets_test


def load_mnist_binarized(dataset=_get_datafolder_path()+'/mnist_binarized/mnist.pkl.gz'):
    '''
    Loads the fixed binarized MNIST dataset provided by Hugo Larochelle.
    :param dataset: path to dataset file
    :return: None
    '''
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_binarized(datasetfolder)

    f = gzip.open(dataset, 'rb')
    x_train, x_valid, x_test = pkl.load(f)
    f.close()
    return x_train, x_valid, x_test

def _download_mnist_binarized_small(datapath):
    datafiles = {
        "train": "http://www.cs.toronto.edu/~larocheh/public/"
                 "datasets/binarized_mnist/binarized_mnist_train.amat",
        "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                 "binarized_mnist/binarized_mnist_valid.amat",
        "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                "binarized_mnist/binarized_mnist_test.amat"
    }
    datasplits = {}
    for split in datafiles.keys():
        logging.info("Downloading {} data into {}".format(split, datapath))
        local_file = datapath + '/binarized_mnist_small_%s.npy'%(split)
        datasplits[split] = np.loadtxt(urllib.urlretrieve(datafiles[split])[0])[:1000]

    f = gzip.open(datapath +'/mnist.pkl.gz', 'w')
    pkl.dump([datasplits['train'],datasplits['valid'],datasplits['test']],f)


def load_mnist_binarized_small(dataset=_get_datafolder_path()+'/mnist_binarized_small/mnist.pkl.gz'):
    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_mnist_binarized_small(datasetfolder)

    f = gzip.open(dataset, 'rb')
    x_train, x_valid, x_test = pkl.load(f)
    f.close()
    return x_train, x_valid, x_test



def _download_iris(datapath):
    origin = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    logging.info("Downloading data from {}".format(origin))
    make_dir(os.path.dirname(datapath))
    urllib.urlretrieve(origin, datapath)
    

def load_iris_dataset(dataset=_get_datafolder_path() + "/iris/iris.csv"):
    if not os.path.exists(dataset):
        _download_iris(dataset)
    test_data = []
    test_data_classes = []
    for f1, f2, f3, f4, cl in np.recfromcsv(dataset, delimiter=","):
        test_data.append(np.asarray([[f1, f2, f3, f4]]))
        test_data_classes.append(cl)
    
    test_data = np.concatenate(test_data)
    uniq_classes = list(set(test_data_classes))
    test_data_classes = np.asarray([ uniq_classes.index(cl) for cl in test_data_classes ])

    return test_data, test_data_classes

def load_toy_dataset():
    test_data = np.asarray([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [0.0, 1.0, 1.0, 1.0]
    ])
    test_data_classes = [0, 1, 2, 3]
    return test_data, test_data_classes

