#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is an experiment that will train a specified generalist network.
"""

from neon.backends import gen_backend
from neon.data import DataIterator, load_cifar10
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

from cifar_net import get_model


if __name__ == '__main__':
    # hyperparameters
    batch_size = 128
    num_epochs = args.epochs

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=args.rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
    model = get_model(nout=nout)
    # TODO: Split train to get a validation set
    train_set = DataIterator(X_test, y_train, nclass=nout)
    test_set = DataIterator(X_test, y_test, nclass=nout)

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
    opt = MultiOptimizer({
        'default': opt_gdmwd,
        'Bias': opt_gdm,
    })

    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    callbacks = Callbacks(model, train_set, output_file=args.output_file,
                      valid_set=test_set, valid_freq=args.validation_freq,
                      progress_bar=args.progress_bar)

    model.fit(train_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

    print 'Test misclassification error: ', model.eval(test_set, metric=Misclassification())

    # TODO: Export trained model to a given .pkl
