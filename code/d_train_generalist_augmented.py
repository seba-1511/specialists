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
from keras.preprocessing.image import ImageDataGenerator

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

    # Setup Data Augmentation:
    print 'Setting up data augmentation...'
    augmenter = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
    )

    # Setup Model and Datasets
    print 'Loading data...'
    if DATASET_NAME == 'cifar10':
        (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
        nout = 16
    elif DATASET_NAME == 'cifar100':
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
        X_train_keras = X_train.copy()
        X_train = X_train.reshape(50000, 3072)
        X_test = X_test.reshape(10000, 3072)
        nout = 128

    if VALIDATION:
        (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)

    model, opt, cost = get_custom_vgg(nout=nout)

    # train_set = DataIterator(X_train, y_train, nclass=nout, lshape=(3, 32, 32))
    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))

    if VALIDATION:
        valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))
        callbacks = Callbacks(model, train_set, output_file=args.output_file,
                            valid_set=valid_set, valid_freq=args.validation_freq,
                            progress_bar=args.progress_bar)

    # Training Procedure
    print 'Begin training...'
    print 'Calibrating augmenter...'
    augmenter.fit(X_train_keras)
    print 'Setting up model...'
    model.cost = cost
    model.set_shortcut()
    model.optimizer = opt
    model.total_cost = be.empty((1, 1))
    while model.epoch_index < num_epochs and not model.finished:
        # Data Augmentation
        print 'Augmenting Data...'
        X_train_augmented = np.empty(X_train_keras.shape)
        y_train_augmented = np.empty(y_train.shape)
        for X, y in augmenter.flow(X_train_keras, y_train):
            X_train_augmented = np.vstack((X_train_augmented, X))
            y_train_augmented = np.vstack((y_train_augmented, y))
        print 'Reshaping augmented batch...'
        X_train_augmented = X_train_augmented.reshape(X_train_augmented.shape[0], 3072)

        print 'Creating iterator...'
        train_set = DataIterator(X_train_augmented, y_train_augmented, nclass=nout, lshape=(3, 32, 32))

        if not VALIDATION:
            callbacks = Callbacks(model, train_set, output_file=args.output_file,
                            valid_set=test_set, valid_freq=args.validation_freq,
                            progress_bar=args.progress_bar)

        print 'Fitting data...'
        model._epoch_fit(train_set, callbacks)

        callbacks.on_epoch_end(model.epoch_index)
        model.epoch_index += 1
    callbacks.on_train_end()


    # Print Results
    print 'Validation: ', VALIDATION
    print 'Train misclassification error: ', model.eval(train_set, metric=Misclassification())
    if VALIDATION:
        print 'Valid misclassification error: ', model.eval(valid_set, metric=Misclassification())
    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    if args.save_path is not None:
        save_obj(model.serialize(), EXPERIMENT_DIR + 'augmented_' + args.save_path)

