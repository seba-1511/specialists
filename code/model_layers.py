#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from neon.models import MLP
from neon.backends import gen_backend
from neon.util.persist import deserialize
from neon.optimizers import GradientDescentMomentumWeightDecay
from neon.params import UniformValGen, GaussianValGen
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


def load_cifar100_train32_test50(path, experiment):
    name = str(path + '/saved_experiments/' + experiment + '/model.prm')
    gdm = GradientDescentMomentumWeightDecay(
        name='gdmwd',
        lr_params={
            'learning_rate': 0.17,
            'weight_decay': 0.0005,
            'schedule': {
                'type': 'step',
                'ratio': 0.1,
                'step_epochs': 50,
            },
            'momentum_params': {
                'type': 'linear_monotone',
                'coef': 0.5,
            },
        },
    )
    wt_init0 = GaussianValGen(scale=0.01, bias_init=0.0)
    wt_init = UniformValGen(low=-0.1, high=0.1)
    datalayer = DataLayer(
        name='d0',
        is_local=True,
        nofm=3,
        ofmshape=[32, 32],
    )

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
            nout=2048,
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

    mlp = MLP(
        num_epochs=740,
        batch_norm=True,
        batch_size=128,
        layers=layers,
    )
    #: Has to be cudanet, because of pad!=0 not implemented on CPU
    backend = gen_backend(gpu='cudanet')
    mlp.link()
    backend.par.init_model(mlp, backend)
    mlp.initialize(backend)
    name = os.path.expandvars(os.path.expanduser(name))
    params = deserialize(load_path=name)
    mlp.set_params(params)
    return mlp
