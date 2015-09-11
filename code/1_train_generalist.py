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

from cifar_net import get_custom_vgg, get_allconv, get_dummy

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()


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

    (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
    (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
    # That's only for the allconv net:
    # nout = 16
    model, opt, cost =get_custom_vgg(nout=nout)

    # TODO: Split train to get a validation set
    train_set = DataIterator(X_train, y_train, nclass=nout, lshape=(3, 32, 32))
    valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))
    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))

    callbacks = Callbacks(model, train_set, output_file=args.output_file,
                          valid_set=valid_set, valid_freq=args.validation_freq,
                          progress_bar=args.progress_bar)

    model.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    print 'Train misclassification error: ', model.eval(train_set, metric=Misclassification())
    print 'Valid misclassification error: ', model.eval(valid_set, metric=Misclassification())
    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    save_obj(model.serialize(), args.save_path)

