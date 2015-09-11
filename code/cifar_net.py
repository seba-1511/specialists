#!/usr/bin/env python
# -*- coding: utf-8 -*-

from neon.layers import (
    Affine,
    Conv,
    Pooling,
    Dropout,
    BatchNorm,
)
from neon.models import Model
from neon.transforms import (
    Misclassification,
    Rectlin,
    Softmax,
)
from neon.initializers import GlorotUniform


def get_model(archive_path=None, nout=10):
    """
    This functions creates and returns the network that will be used for the
    generalists and the specialists.
    """
    # TODO: Check that init works properly.
    conv_init = GlorotUniform()
    fc_init = GlorotUniform()
    last_init = GlorotUniform()
    layers = [
        Conv((3, 3, 128), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 128), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 256), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 256), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Conv((3, 3, 512), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Conv((3, 3, 512), init=conv_init, activation=Rectlin(), pad=1),
        BatchNorm(),
        Pooling((2, 2), strides=2, op='max'),

        Affine(4096, init=fc_init, activation=Rectlin()),
        BatchNorm(),
        Dropout(0.5),
        Affine(4096, init=fc_init, activation=Rectlin()),
        BatchNorm(),
        Dropout(0.5),
        Affine(nout, init=last_init, activation=Softmax()),
    ]
    # Instanciate model
    if archive_path != None:
        raise('Loading weights not implemented yet.')
    # Set model weights

    return Model(layers=layers)
