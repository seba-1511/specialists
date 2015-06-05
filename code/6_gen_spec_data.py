#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO:
#: Import cudanet backend
#: Import Datapar and try it

import numpy as np

from sklearn.metrics import confusion_matrix

from neon.datasets import CIFAR100
from neon.backends.cpu import CPU
from neon.backends.par import NoPar
from neon.models import MLP
from neon.transforms import (
    RectLin,
    Softmax,
    CrossEntropy,
)
from neon.layers import (
    DataLayer,
    ConvLayer,
    PoolingLayer,
    CrossMapResponseNormLayer,
    DropOutLayer,
    FCLayer,
    CostLayer,
)


def load_model(name):
    gdm = {}
    wt_init0 = {}
    layers = [
        DataLayer(
            name='d0',
            is_local=True,
            nofm=3,
            ofmshape=[32, 32],
        ),
        ConvLayer(
            name='layer1',
            lrule_init=gdm,
            weight_init=wt_init0,
            nofm=128,
            fshape=[8, 8],
            activation=RectLin(),
            pad=4,
        ),
        PoolingLayer(
            name='layer2',
            op='max',
            fshape=[4, 4],
            stride=2,
        ),
        CrossMapResponseNormLayer(
            name='layer3',
            ksize=5,
            alpha=0.0001,
            beta=0.9,
        ),
        DropOutLayer(
            name='do1',
            keep=0.8,
        ),

        ConvLayer(
            name='layer4',
            lrule_init=gdm,
            nofm=192,
            fshape=[8, 8],
            weight_init=wt_init0,
            activation=RectLin(),
            pad=3,
        ),
        PoolingLayer(
            name='layer5',
            op='max',
            fshape=[4, 4],
            stride=2,
        ),
        CrossMapResponseNormLayer(
            name='layer6',
            ksize=5,
            alpha=0.0001,
            beta=1.9365,
        ),
        DropOutLayer(
            name='do2',
            keep=0.5,
        ),

        ConvLayer(
            name='layer7',
            lrule_init=gdm,
            nofm=256,
            fshape=[5, 5],
            weight_init=wt_init0,
            activation=RectLin(),
            pad=3,
        ),
        PoolingLayer(),
        CrossMapResponseNormLayer(),
        DropOutLayer(),
        ConvLayer(),

        FCLayer(),
        DropOutLayer(),

        FCLayer(),
        CostLayer(),
    ]

    return MLP(
        serialized_path='saved_models/' + name,
        deserialize_path='saved_models/' + name,
        num_epochs=74,
        batch_norm=True,
        batch_size=128,
        layers=layers,
    )


def load_data():
    data = CIFAR100(repo_path='~/data')
    data.backend = CPU()
    data.backend.actual_batch_size = 128
    par = NoPar()
    par.associate(data.backend)
    data.load()
    return data, par


def soft_cm(targets, preds):
    return [[]]


if __name__ == '__main__':
    model = load_model('maxout_copy_100.prm')
    import pdb
    pdb.set_trace()
    data, par = load_data()
    targets = data.targets['test']
    pred_probs = model.predict_proba(data.inputs['test'])
    pred_classes = np.argmax(pred_probs, axis=1)
    cm_categorical = confusion_matrix(targets, pred_classes)
    cm_soft = soft_cm(targets, pred_probs)
