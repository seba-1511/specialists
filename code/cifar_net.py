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
        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 128), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 256), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 512), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 512), init=conv_init, bias=Constant(0), activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Affine(nout=4096, init=fc_init, bias=Constant(1), activation=Rectlin()),
        BatchNorm(),
        Dropout(0.5),
        Affine(nout=4096, init=fc_init, bias=Constant(1), activation=Rectlin()),
        BatchNorm(),
        Dropout(0.5),
        Affine(nout=nout, init=last_init, bias=Constant(-7), activation=Rectlin()),
        Activation(Softmax()),
    ]
    # Instanciate model
    if archive_path != None:
        raise('Loading weights not implemented yet.')
    # Set model weights

    return Model(layers=layers)

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
    return Model(layers=layers)

def get_dummy(archive_path=None, nout=10):
    return Model(
        layers=[
            Affine(100),
            Affine(100),
        ]
    )

