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
    # TODO: Check that init works properly.
    conv_init = GlorotUniform()
    fc_init = GlorotUniform()
    last_init = GlorotUniform()
    layers = [
        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 512), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Conv((3, 3, 512), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1, batch_norm=True),
        Pooling((2, 2), strides=2, op='max'),

        Affine(nout=4096, init=fc_init, bias=Constant(1), activation=Rectlin(), batch_norm=True),
        Dropout(0.5),
        Affine(nout=4096, init=fc_init, bias=Constant(1), activation=Rectlin(), batch_norm=True),
        Dropout(0.5),
        Affine(nout=nout, init=last_init, bias=Constant(-7), activation=Softmax()),
        # Activation(Softmax()),
    ]
    model = Model(layers=layers)
    if archive_path is not None:
        # Set model weights
        model.load_weights(archive_path)
    opt_schedule = Schedule(step_config=20, change=0.1)
    opt_gdm = GradientDescentMomentum(
        learning_rate=0.0002,
        momentum_coef=0.90,
        schedule=opt_schedule,
    )
    opt_gdmwd = GradientDescentMomentum(
        learning_rate=0.0001,
        momentum_coef=0.90,
        wdecay=0.0005, # original: 0.0005
        schedule=opt_schedule,
    )
    opt = MultiOptimizer({
        'default': opt_gdmwd,
        'Bias': opt_gdm,
    })
    cost = GeneralizedCost(costfunc=CrossEntropyMulti(epsilon=0.0005, scale=1000))
    return model, opt, cost

def get_allconv(archive_path=None, nout=10):
    init_uni = GlorotUniform()
    layers = []
    layers.append(Dropout(keep=.8))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 96), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
    layers.append(Dropout(keep=.5))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin(), pad=1, strides=2))
    layers.append(Dropout(keep=.5))
    layers.append(Conv((3, 3, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((1 ,1, 192), init=init_uni, batch_norm=True, activation=Rectlin()))
    layers.append(Conv((1,1,16), init=init_uni, activation=Rectlin()))
    layers.append(Pooling(6, op="avg"))
    layers.append(Activation(Softmax()))
    model = Model(layers=layers)
    if archive_path is not None:
        model.load_weights(archive_path)
    opt_schedule = Schedule(step_config=20, change=0.1)
    opt = GradientDescentMomentum(
        learning_rate=0.5,
        schedule=opt_schedule,
        momentum_coef=0.9,
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


