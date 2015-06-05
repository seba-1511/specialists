#!/usr/bin/env python
# -*- coding: utf-8 -*-


from neon.models import MLP
from neon.backends.par import NoPar
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


def load_cifar100_train32_test50():
    name = 'maxout_copy_100.prm'
    gdm = {}
    wt_init0 = {}
    wt_init = {}
    datalayer = DataLayer(
        name='d0',
        is_local=True,
        nofm=3,
        ofmshape=[32, 32],
    ),

    layers = [
        datalayer,
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
        PoolingLayer(
            name='layer8',
            op='max',
            fshape=[2, 2],
            stride=2,
        ),
        CrossMapResponseNormLayer(
            name='layer9',
            ksize=5,
            alpha=0.0001,
            beta=1.9365,
        ),
        DropOutLayer(
            name='do3',
            keep=0.42,
        ),

        FCLayer(
            name='layer10',
            nout=2048,
            lrule_init=gdm,
            weight_init=wt_init,
            activation=RectLin(),
        ),
        DropOutLayer(
            name='do4',
            keep=0.37,
        ),

        FCLayer(
            name='layer11',
            nout=1024,
            lrule_init=gdm,
            weight_init=wt_init,
            activation=RectLin(),
        ),
        DropOutLayer(
            name='do5',
            keep=0.32,
        ),
        FCLayer(
            name='output',
            lrule_init=gdm,
            weight_init=wt_init,
            nout=100,
            activation=Softmax(),
        ),
        CostLayer(
            name='cost',
            ref_layer=datalayer,
            cost=CrossEntropy(),
        ),
    ]

    return MLP(
        serialized_path='./saved_models/' + name,
        deserialize_path='./saved_models/' + name,
        num_epochs=74,
        batch_norm=True,
        batch_size=128,
        layers=layers,
    )
