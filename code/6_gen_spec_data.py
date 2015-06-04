#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO:
#: Import cudanet backend
#: Import Datapar and try it

import numpy as np

from sklearn.metrics import confusion_matrix

from neon.utils import deserialize
from neon.datasets import CIFAR100
from neon.backends.cpu import CPU
from neon.backends.par import NoPar


def load_model(name):
    return deserialize('./saved_models' + name)


def load_data():
    data = CIFAR100()
    data.backend = CPU()
    par = NoPar()
    par.associate(data.backend)
    data.load()
    return data, par

def soft_cm(targets, preds):
    return [[] ]


if __name__ == '__main__':
    model = load_model('maxout_copy_100.prm')
    data = load_data()
    targets = data.targets['test']
    pred_probs = model.predict_proba(data.inputs['test'])
    pred_classes = np.argmax(pred_probs, axis=1)
    cm_categorical = confusion_matrix(targets, pred_classes)
    cm_soft = soft_cm(targets, pred_probs)
