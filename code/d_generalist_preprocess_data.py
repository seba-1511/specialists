
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is an experiment that will train a specified generalist network.
"""

import cPickle as pk
import numpy as np
import os
import random

from cifar_net import get_custom_vgg
from keras.datasets import cifar100
from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.data import DataIterator, load_cifar10
from neon.transforms.cost import Misclassification
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from scipy import linalg
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import array2d, as_float_array

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

DATASET_NAME = 'cifar100'
EXPERIMENT_DIR = 'experiments/' + DATASET_NAME + '/'
VALIDATION = True


def split_train_set(X_train, y_train):
    return (X_train[:-5000], y_train[:-5000]), (X_train[-5000:], y_train[-5000:])


def load_data():
    if DATASET_NAME == 'cifar10':
        (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
        nout = 16
    elif DATASET_NAME == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)
        nout = 128
    elif DATASET_NAME == 'svhn':
        from scipy.io import loadmat
        train = loadmat('../data/svhm_train.mat')
        test = loadmat('../data/svhn_test.mat')
        (X_train, y_train), (X_test, y_test) = (train['X'], train['y']), (test['X'], test['y'])
        s = X_train.shape
        X_train = X_train.reshape(-1, s[-1]).transpose()
        s = X_test.shape
        X_test = X_test.reshape(-1, s[-1]).transpose()
        temp = np.empty(X_train.shape, dtype=np.uint)
        np.copyto(temp, X_train)
        X_train = temp
        temp = np.empty(X_test.shape, dtype=np.uint)
        np.copyto(temp, X_test)
        X_test = temp
        nout = 16
    return (X_train, y_train), (X_test, y_test), nout

class ZCA(BaseEstimator, TransformerMixin):

    """
    Taken from: https://gist.github.com/duschendestroyer/5170087
    """
    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = array2d(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

if __name__ == '__main__':
    # hyperparameters
    batch_size = 128
    num_epochs = args.epochs
    num_epochs = 74 if num_epochs == 10 else num_epochs
    rng_seed = 1234
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    filename = DATASET_NAME + '_preprocessed.pkl'
    if os.path.isfile(filename):
        with open(filename, 'rb') as prep:
            (X_train, y_train), (X_test, y_test), nout = pk.load(prep, pk.HIGHEST_PROTOCOL)
    else:
        (X_train, y_train), (X_test, y_test), nout = load_data()
        zca = ZCA()
        zca.fit(X_train)
        X_train = zca.transform(X_train)
        X_test = zca.transform(X_test)
        dataset_preprocessed = ((X_train, y_train), (X_test, y_test), nout)
        with open(filename, 'wb') as prep:
            pk.dump(dataset_preprocessed, prep, pk.HIGHEST_PROTOCOL)


    if VALIDATION:
        (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
    model, opt, cost = get_custom_vgg(nout=nout)


    train_set = DataIterator(X_train, y_train, nclass=nout, lshape=(3, 32, 32))
    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))

    callbacks = Callbacks(model, train_set, args, eval_set=test_set)
    if VALIDATION:
        valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))
        callbacks = Callbacks(model, train_set, args, eval_set=valid_set)

    model.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    print 'Validation: ', VALIDATION
    print 'Train misclassification error: ', model.eval(train_set, metric=Misclassification())
    if VALIDATION:
        print 'Valid misclassification error: ', model.eval(valid_set, metric=Misclassification())
    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    if args.save_path is not None:
        save_obj(model.serialize(), EXPERIMENT_DIR + args.save_path)

