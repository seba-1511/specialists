#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO:
#: Import cudanet backend
#: Import Datapar and try it

import os
import numpy as np
import cPickle as pk

from sklearn.metrics import confusion_matrix

from neon.backends.par import NoPar
from neon.datasets import CIFAR100
from neon.backends.cpu import CPU

from model_layers import (
    load_cifar100_train32_test50,
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def load_targets(name):
    f = open(CURR_DIR + '/inferences/' + name + '/train-targets.pkl', 'rb')
    train = pk.load(f)
    train = np.argmax(train, axis=1)
    f.close()
    f = open(CURR_DIR + '/inferences/' + name + '/test-targets.pkl', 'rb')
    test = pk.load(f)
    test = np.argmax(test, axis=1)
    f.close()
    return (train, test)


def load_inferences(name):
    f = open(CURR_DIR + '/inferences/' + name + '/train-inference.pkl', 'rb')
    train = pk.load(f)
    f.close()
    f = open(CURR_DIR + '/inferences/' + name + '/test-inference.pkl', 'rb')
    test = pk.load(f)
    f.close()
    return (train, test)


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
    # model = load_cifar100_train32_test50()
    # data, par = load_data()
    # targets = data.targets['test']
    # pred_probs = model.predict_proba(data.inputs['test'])
    experiment = '5_test50_train33_156epochs'
    train_pred_probs, test_pred_probs = load_inferences(name=experiment)
    train_targets, test_targets = load_targets(name=experiment)
    # train_pred_classes = np.argmax(train_pred_probs, axis=1)
    test_pred_classes = np.argmax(test_pred_probs, axis=1)
    cm_categorical = confusion_matrix(test_targets, test_pred_classes)
    cm_soft = soft_cm(test_targets, test_pred_probs)

    #import pdb
    #pdb.set_trace()
