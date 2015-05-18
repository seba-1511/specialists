#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(level=20)
logger = logging.getLogger()

from neon.backends import gen_backend
from neon.layers import (
    FCLayer,
    DataLayer,
    CostLayer,
    ConvLayer,
    PoolingLayer,
)
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.datasets import CIFAR100
from neon.experiments import FitPredictErrorExperiment


def create_model(nin):
    layers = [
        DataLayer(nofm=3, ofmshape=[32, 32], is_local=True),
        ConvLayer(nofm=16, fshape=[5, 5]),
        PoolingLayer(op='max', fshape=[2, 2], stride=2),
        ConvLayer(nofm=32, fshape=[5, 5]),
        PoolingLayer(op='max', fshape=[2, 2], stride=2),
        FCLayer(nout=500, activation=RectLin()),
        FCLayer(nout=100, activation=Logistic()),
        CostLayer(cost=CrossEntropy()),
    ]
    model = MLP(num_epochs=10, batch_size=128, layers=layers)
    return model


def run():
    model = create_model(nin=3072)
    backend = gen_backend(rng_seed=0)
    dataset = CIFAR100(repo_path='~/data/')
    experiment = FitPredictErrorExperiment(model=model,
                                           backend=backend,
                                           dataset=dataset)
    experiment.run()


if __name__ == '__main__':
    run()
