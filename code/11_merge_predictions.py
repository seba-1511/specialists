#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pk
import numpy as np
from neon.datasets import SpecialistDataset
from sklearn.metrics import (
    accuracy_score,
    log_loss,
)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


def load_targets(path, name):
    f = open(path + '/saved_experiments/' +
             name + '/train-targets.pkl', 'rb')
    train = pk.load(f)
    train = np.argmax(train, axis=1)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/test-targets.pkl', 'rb')
    test = pk.load(f)
    test = np.argmax(test, axis=1)
    f.close()
    return (train, test)


def load_inferences(path, name):
    f = open(path + '/saved_experiments/' +
             name + '/train-inference.pkl', 'rb')
    train = pk.load(f)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/test-inference.pkl', 'rb')
    test = pk.load(f)
    f.close()
    return (train, test)


def load_spec_inferences(name, spec):
    pass


def merge_predictions():
    pass


def merge_predictions_weighted():
    pass

if __name__ == '__main__':
    experiment = '5_test45_train22_740epochs'
    train_probs, test_probs = load_inferences(name=experiment)
    train_targets, test_targets = load_targets(name=experiment)
    clusters = SpecialistDataset.cluster_classes(
        test_targets,
        test_probs,
        cm=SpecialistDataset.cm_types['greedy'],
        nb_clusters=10,
        clustering=SpecialistDataset.clustering_methods['soft_sum_pred_cm']
    )
    final_probs = np.zeros(np.shape(test_probs))
    print 'Generalist logloss: ', log_loss(test_targets, test_probs)
    print 'Generalist accuracy: ', accuracy_score(test_targets, np.argmax(test_probs, axis=1))
    for i, c in enumerate(clusters):
        spec_train_probs, spec_test_probs = load_spec_inferences(
            name=experiment, spec=i
        )
        final_probs = merge_predictions(
            generalist=test_probs,
            specialist=spec_test_probs,
            cluster=c,
            final=final_probs,
        )
        print i, ': Spec logloss: ', log_loss(test_targets, spec_test_probs)
        print i, ': Spec accuracy: ', accuracy_score(test_targets, np.argmax(spec_test_probs, axis=1))
        print i, ': Final logloss: ', log_loss(test_targets, final_probs)
        print i, ': Final accuracy:', accuracy_score(test_targets, np.argmax(final_probs, axis=1))
    print 'The final results for the whole system are: '
    print 'Logloss: ', log_loss(test_targets, final_probs)
    print 'Accuracy: ', accuracy_score(test_targets, np.argmax(final_probs, axis=1))
