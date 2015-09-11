#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is an experiment that will train a specified generalist network.
"""

import numpy as np

from neon.backends import gen_backend
from neon.data import DataIterator, load_cifar10
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

from cifar_net import get_custom_vgg, get_allconv, get_dummy

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()


if __name__ == '__main__':
    # hyperparameters
    batch_size = 128
    num_epochs = args.epochs
    num_epochs = 74 if num_epochs == 10 else num_epochs
    rng_seed = 1234
    rng_seed = args.rng_seed

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
    # That's only for the allconv net:
    nout = 16
    model = get_allconv(nout=nout)
    # TODO: Split train to get a validation set
    train_set = DataIterator(X_train, y_train, nclass=nout, lshape=(3, 32, 32))
    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))

    opt_schedule = Schedule(step_config=20, change=0.1)
    opt_gdm = GradientDescentMomentum(
        learning_rate=0.0002,
        momentum_coef=0.90,
        schedule=opt_schedule,
    )
    opt_gdmwd = GradientDescentMomentum(
        learning_rate=0.0001,
        momentum_coef=0.90,
        wdecay=0.0005,
        schedule=opt_schedule,
    )
    opt_allconv = GradientDescentMomentum(
        learning_rate=0.5,
        schedule=opt_schedule,
        momentum_coef=0.9,
        wdecay=0.0001,
    )
    opt = MultiOptimizer({
        'default': opt_gdmwd,
        'Bias': opt_gdm,
    })

    scaled_cost = GeneralizedCost(costfunc=CrossEntropyMulti(epsilon=0.0005, scale=1000))
    general_cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    callbacks = Callbacks(model, train_set, output_file=args.output_file,
                          valid_set=test_set, valid_freq=args.validation_freq,
                          progress_bar=args.progress_bar)

    model.fit(train_set, optimizer=opt_allconv, num_epochs=num_epochs, cost=general_cost, callbacks=callbacks)

    print 'Train misclassification error: ', model.eval(train_set, metric=Misclassification())
    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    # TODO: Export trained model to a given .pkl
