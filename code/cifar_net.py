#!/usr/bin/env python
# -*- coding: utf-8 -*-

from neon.layers import (
    Affine,
    Conv,
    Pooling,
    Dropout,
    BatchNorm,
    Activation,
)
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, Schedule, MultiOptimizer
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
from neon.transforms import (
    Misclassification,
    Rectlin,
    Softmax,
)
from neon.initializers import GlorotUniform, Uniform, Constant


def get_custom_vgg(archive_path=None, nout=10):
    """
    This functions creates and returns the network that will be used for the
    generalists and the specialists.
    """
    conv_init = GlorotUniform()
    fc_init = GlorotUniform()
    last_init = GlorotUniform()
    layers = [
        # Initially Convs are: 128, 256, 512 Affines: 4096, 4096, no dropout between convs
        # 34.45% on 100: 96, 128, 256, do: 0.5, 0.5, 0.5, fc: 2048, 2048, wd: 0.00075, 74e, sched[0.00005, 0.000001, 0.0000005 for 20, 40, 60]
        Conv((3, 3, 96), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 96), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Dropout(0.5),

        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Dropout(0.5),

        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Dropout(0.5),

        Affine(nout=2048, init=fc_init, bias=Constant(1), activation=Rectlin(), batch_norm=True),
        Dropout(0.5),
        Affine(nout=2048, init=fc_init, bias=Constant(1), activation=Rectlin(), batch_norm=True),
        Dropout(0.5),
        Affine(nout=nout, init=last_init, bias=Constant(-7), activation=Softmax()),
        # Activation(Softmax()),
    ]
    model = Model(layers=layers)
    if archive_path is not None:
        # Set model weights
        model.load_weights(archive_path)
    # opt_schedule = Schedule(step_config=20, change=0.1)
    opt_schedule = Schedule(
        step_config=[20, 40, 60, 80, 100, 120, 140, 160],
        change=[0.00005, 0.000001, 0.0000005, 0.00000025, 0.0000001, 0.00000005, 0.000000025, 0.00000001]
    )
    opt_gdm = GradientDescentMomentum(
        learning_rate=0.0002,
        momentum_coef=0.90,
        schedule=opt_schedule,
    )
    opt_gdmwd = GradientDescentMomentum(
        learning_rate=0.0001,
        momentum_coef=0.90,
        wdecay=0.00075, # original: 0.0005
        schedule=opt_schedule,
    )
    opt = MultiOptimizer({
        'default': opt_gdmwd,
        'Bias': opt_gdm,
    })
    # cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(epsilon=0.0005, scale=1000))
    return model, opt, cost

def get_allconv(archive_path=None, nout=10):
    init_uni = GlorotUniform()
    layers = []
    layers.append(Dropout(keep=0.8))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
    layers.append(Dropout(keep=0.5))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
    layers.append(Dropout(keep=0.5))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((1, 1, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((1, 1, nout), init=init_uni, activation=Rectlin()))
    layers.append(Pooling(6, op="avg"))
    layers.append(Activation(Softmax()))
    model = Model(layers=layers)
    if archive_path is not None:
        model.load_weights(archive_path)
    # opt_schedule = Schedule(step_config=[20, 40, 60, 80, 100, 120, 140, 160], change=[0.1, 0.05, 0.025, 0.01, 0.005, 0.0001, 0.00005, 0.00001])
    opt_schedule = Schedule(step_config=20, change=0.1)
    opt = GradientDescentMomentum(
        learning_rate=0.5,
        schedule=opt_schedule,
        momentum_coef=0.9,
        # wdecay=0.00077,
        wdecay=0.00075,
    )
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    return model, opt, cost

def get_dummy(archive_path=None, nout=10):
    act = Rectlin()
    init = GlorotUniform()
    model = Model(layers=[Affine(100, init=init, activation=act), Affine(nout, init=init, activation=act)])
    opt = GradientDescentMomentum(learning_rate=0.5, momentum_coef=0.0)
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())
    return model, opt, cost


