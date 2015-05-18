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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_model(nin):
    layers = [
        DataLayer(nofm=3, ofmshape=[32, 32], is_local=True),
        #ConvLayer(nofm=16, fshape=[5, 5]),
        #PoolingLayer(op='max', fshape=[2, 2], stride=2),
        #ConvLayer(nofm=32, fshape=[5, 5]),
        #PoolingLayer(op='max', fshape=[2, 2], stride=2),
        FCLayer(nin=32*32*3, nout=500, activation=RectLin()),
        FCLayer(nin=500, nout=100, activation=Logistic()),
        CostLayer(cost=CrossEntropy()),
    ]
    model = MLP(num_epochs=3, batch_size=128, layers=layers)
    return model


def run():
    model = create_model(nin=3072)
    backend = gen_backend(rng_seed=0)
    dataset = CIFAR100(repo_path='~/data/')
    model.initialize(backend)
    model.fit(dataset)
    pred = model.predict_generator(dataset, 'test')
    confusion = confusion_matrix(dataset.targets['test'], pred)
    plot_confusion_matrix(cm=confusion)
    # experiment = FitPredictErrorExperiment(model=model,
                                           # backend=backend,
                                           # dataset=dataset)
    # experiment.run()


if __name__ == '__main__':
    run()
