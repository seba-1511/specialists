#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is an experiment that will train a specified generalist network.
"""

import numpy as np

from neon.backends import gen_backend
from neon.data import DataIterator, load_cifar10
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj

from keras.datasets import cifar100

from cifar_net import get_custom_vgg, get_allconv, get_dummy

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

DATASET_NAME = 'cifar100'
EXPERIMENT_DIR = 'experiments/' + DATASET_NAME + '/'
VALIDATION = False


def split_train_set(X_train, y_train):
    return (X_train[:-5000], y_train[:-5000]), (X_train[-5000:], y_train[-5000:])

if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    num_epochs = args.epochs
    num_epochs = 74 if num_epochs == 10 else num_epochs
    rng_seed = 1234

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    if DATASET_NAME == 'cifar10':
        (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
        nout = 16
    elif DATASET_NAME == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)
        nout = 128

    if VALIDATION:
        (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
    model, opt, cost = get_custom_vgg(nout=nout)

    train_set = DataIterator(X_train, y_train, nclass=nout, lshape=(3, 32, 32))
    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))

    callbacks = Callbacks(model, train_set, output_file=args.output_file,
                          valid_set=test_set, valid_freq=args.validation_freq,
                          progress_bar=args.progress_bar)
    if VALIDATION:
        valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))
        callbacks = Callbacks(model, train_set, output_file=args.output_file,
                            valid_set=valid_set, valid_freq=args.validation_freq,
                            progress_bar=args.progress_bar)

    model.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    print 'Validation: ', VALIDATION
    print 'Train misclassification error: ', model.eval(train_set, metric=Misclassification())
    if VALIDATION:
        print 'Valid misclassification error: ', model.eval(valid_set, metric=Misclassification())
    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    if args.save_path is not None:
        save_obj(model.serialize(), EXPERIMENT_DIR + args.save_path)

