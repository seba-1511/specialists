#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pk
import numpy as np
from model_layers import load_cifar100_train32_test50
from neon.backends import gen_backend
from neon.datasets import (
    SpecialistDataset,
    CIFAR10,
    CIFAR100,
)
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
    path = CURR_DIR
    f = open(path + '/saved_experiments/' +
             name + '/spec_' + str(spec) + '/train-inference.pkl', 'rb')
    train = pk.load(f)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/spec_' + str(spec) + '/test-inference.pkl', 'rb')
    test = pk.load(f)
    f.close()
    return (train, test)


def load_spec_targets(name, spec):
    path = CURR_DIR
    f = open(path + '/saved_experiments/' +
             name + '/spec_' + str(spec) + '/train-targets.pkl', 'rb')
    train = pk.load(f)
    train = np.argmax(train, axis=1)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/spec_' + str(spec) + '/test-targets.pkl', 'rb')
    test = pk.load(f)
    test = np.argmax(test, axis=1)
    f.close()
    return (train, test)


def to_one_hot(i, n_classes):
    return np.array([0 if x != i else 1 for x in xrange(n_classes)])


def merge_predictions(generalist, specialist, cluster, final):
    n_classes = len(generalist[0])
    for i, g in enumerate(generalist):
        if np.argmax(g) in cluster:
            s = np.argmax(specialist[i])
            s = to_one_hot(cluster[s], n_classes)
            final[i] += s
    return final


def merge_predictions_weighted(generalist, specialist, cluster, final):
    n_classes = len(generalist[0])
    for i, g in enumerate(generalist):
        if np.argmax(g) in cluster:
            s = np.argmax(specialist[i])
            s = to_one_hot(cluster[s], n_classes)
            final[i] += np.max(g) * s
    return final


def spec_log_loss(targets, probs, cluster):
    spec_tar = []
    spec_probs = []
    for t, p in zip(targets, probs):
        if t in cluster:
            spec_tar.append(t)
            spec_probs.append(p)
    return log_loss(spec_tar, spec_probs)


def spec_accuracy(targets, probs, cluster):
    spec_tar = []
    spec_probs = []
    for t, p in zip(targets, probs):
        if t in cluster:
            spec_tar.append(t)
            spec_probs.append(p)
    return accuracy_score(spec_tar, spec_probs)


def merge_predictions_archives():
    #: I believe that the predictions are shuffled, so you need to load them and
    #:
    nb_clusters = 10
    experiment = '9_gene45.22_10spec'
    # experiment = '5_test45_train22_740epochs'
    train_probs, test_probs = load_inferences(path=CURR_DIR, name=experiment)
    train_targets, test_targets = load_targets(path=CURR_DIR, name=experiment)
    clusters = SpecialistDataset.cluster_classes(
        test_targets,
        test_probs,
        cm=SpecialistDataset.cm_types['soft_sum_pred_cm'],
        nb_clusters=nb_clusters,
        clustering=SpecialistDataset.clustering_methods['greedy']
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
        print i, ': Spec logloss: ', spec_log_loss(test_targets, spec_test_probs, c)
        print i, ': Spec accuracy: ', spec_accuracy(test_targets, np.argmax(spec_test_probs, axis=1), c)
        print i, ': Final logloss: ', log_loss(test_targets, final_probs)
        print i, ': Final accuracy:', accuracy_score(test_targets, np.argmax(final_probs, axis=1))
    print 'The final results for the whole system are: '
    print 'Logloss: ', log_loss(test_targets, final_probs)
    print 'Accuracy: ', accuracy_score(test_targets, np.argmax(final_probs, axis=1))


def load_generalist_model(experiment):
    return load_cifar100_train32_test50(CURR_DIR, experiment)


def load_specialist_model(experiment, spec):
    pass


def get_spec_probs(spec, experiment, features):
    specialist = load_specialist_model(experiment, spec)
    specialist.set_train_mode(False)
    return specialist.predict(features)

if __name__ == '__main__':
    nb_clusters = 10
    experiment = '9_gene45.22_10spec'
    # mlp = load_generalist_model(experiment)
    backend = gen_backend()
    data = CIFAR100(repo_path='~/data', backend=backend)
    data.load()
    features = data.inputs['test']
    targets = data.targets['test']
    mlp.set_train_mode(False)
    gene_probs = mlp.predict(features)
    final_probs = np.zeros(np.shape(gene_probs))
    print 'Generalist logloss: ', log_loss(targets, gene_probs)
    print 'Generalist accuracy: ', accuracy_score(targets, np.argmax(gene_probs, axis=1))
    #: TODO: Change to some validation set !
    clusters = SpecialistDataset.cluster_classes(
        targets,
        gene_probs,
        cm=SpecialistDataset.cm_types['soft_sum_pred_cm'],
        nb_clusters=nb_clusters,
        clustering=SpecialistDataset.clustering_methods['greedy']
    )
    for i, c in enumerate(clusters):
        spec_probs = get_spec_probs(i, experiment, features)
        final_probs = merge_predictions(
            generalist=gene_probs,
            specialist=spec_probs,
            cluster=c,
            final=final_probs,
        )
        print i, ': Spec logloss: ', spec_log_loss(targets, spec_probs, c)
        print i, ': Spec accuracy: ', spec_accuracy(targets, np.argmax(spec_probs, axis=1), c)
        print i, ': Final logloss: ', log_loss(targets, final_probs)
        print i, ': Final accuracy:', accuracy_score(targets, np.argmax(final_probs, axis=1))
    print 'The final results for the whole system are: '
    print 'Logloss: ', log_loss(targets, final_probs)
    print 'Accuracy: ', accuracy_score(targets, np.argmax(final_probs, axis=1))
