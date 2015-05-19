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
from neon.transforms import RectLin, Logistic, CrossEntropy, Softmax
from neon.datasets import CIFAR100
from neon.experiments import FitPredictErrorExperiment
from neon.optimizers import GradientDescentMomentum

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


def create_model(nout):
    layers = [
        DataLayer(nofm=3, ofmshape=[32, 32], is_local=True),
        ConvLayer(nofm=64, fshape=[5, 5], activation=RectLin()),
        PoolingLayer(op='max', fshape=[2, 2], stride=1),
        ConvLayer(nofm=128, fshape=[5, 5], activation=RectLin()),
        PoolingLayer(op='max', fshape=[2, 2], stride=1),
        FCLayer(nout=2000, activation=RectLin()),
        FCLayer(nout=2000, activation=RectLin()),
        FCLayer(nout=nout, activation=Logistic()),
        CostLayer(cost=CrossEntropy()),
    ]
    model = MLP(num_epochs=300, batch_size=128, layers=layers)
    return model


def run():
    backend = gen_backend(rng_seed=0, gpu='cudanet')
    dataset = CIFAR100(repo_path='~/data/')
    dataset.load(backend=backend)
    model = create_model(nout=len(dataset.targets['train']))
    momentum_params={
        'type': 'constant',
        'coef': 0.8,
    }
    lr_params = {
        'learning_rate': .01,
        'momentum_params': momentum_params,
    }
    lrule = GradientDescentMomentum(lr_params=lr_params, name='gdm')
    #model.initialize(backend)
    #model.fit(dataset)
    #pred = model.predict_generator(dataset, 'test')
    #confusion = confusion_matrix(dataset.targets['test'], pred)
    #plot_confusion_matrix(cm=confusion)
    experiment = FitPredictErrorExperiment(
        model=model,
        backend=backend,
        dataset=dataset,
        lrule=lrule,
    )
    #pred = model.predict_generator(dataset, 'test')
    #import pdb; pdb.set_trace()
    #confusion = confusion_matrix(dataset.targets['test'], pred)
    #plot_confusion_matrix(cm=confusion)

    experiment.run()


if __name__ == '__main__':
    run()
