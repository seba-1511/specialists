#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cPickle as pk
import numpy as np
from model_layers import load_model
from neon.backends import gen_backend
from neon.backends.par import NoPar
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
            s = g
            for j, val in enumerate(specialist[i]):
                s[cluster[j]] += val
            s /= sum(s)
            final[i] += s
    return final


def merge_predictions_weighted(generalist, specialist, cluster, final):
    n_classes = len(generalist[0])
    for i, g in enumerate(generalist):
        if np.argmax(g) in cluster:
            s = g * np.max(g)
            for j, val in enumerate(specialist[i]):
                s[cluster[j]] += val
            s /= sum(s)
            final[i] += s
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


def load_generalist_model(experiment, backend, nb_classes):
    name = str(CURR_DIR + '/saved_experiments/' + experiment + '/model.prm')
    return load_model(experiment, name, backend, nb_classes)


def load_specialist_model(experiment, spec, backend, nb_classes):
    name = str(CURR_DIR + '/saved_experiments/' +
               experiment + '/spec_' + str(spec) + '/model.prm')
    return load_model(experiment, name, backend, nb_classes)


def get_spec_probs(spec, experiment, dataset, backend, nb_classes):
    specialist = load_specialist_model(experiment, spec, backend, nb_classes)
    specialist.set_train_mode(False)
    temp = []
    targets = []
    gene_probs = specialist.predict_generator(data, 'test')
    for pred, tar in gene_probs:
        pred = pred.asnumpyarray().transpose()
        tar = np.argmax(tar.asnumpyarray().transpose(), axis=1)
        temp += pred.tolist()
        targets += tar.tolist()
    gene_probs = np.array(temp)
    targets = np.array(targets)
    return gene_probs, targets

if __name__ == '__main__':
    nb_classes = 10
    nb_clusters = 3
    experiment = '4_test22_train14_74epochs'
    #experiment = '9_gene45.22_10spec'
    par = NoPar()
    backend = gen_backend(gpu='cudanet', device_id=0)
    par.associate(backend)
    mlp = load_generalist_model(experiment, backend, nb_classes)
    backend.actual_batch_size = mlp.batch_size
    data = CIFAR10(repo_path='~/data', backend=backend)
    data.backend = backend
    data.set_batch_size(mlp.batch_size)
    data.load(backend=backend)
    mlp.set_train_mode(False)
    temp = []
    targets = []
    gene_probs = mlp.predict_generator(data, 'test')
    for pred, tar in gene_probs:
        pred = pred.asnumpyarray().transpose()
        tar = np.argmax(tar.asnumpyarray().transpose(), axis=1)
        temp += pred.tolist()
        targets += tar.tolist()
    gene_probs = np.array(temp)
    targets = np.array(targets)
    final_probs = np.zeros(np.shape(gene_probs))
    print 'Generalist logloss: ', log_loss(targets, gene_probs)
    print 'Generalist accuracy: ', accuracy_score(targets, np.argmax(gene_probs, axis=1))
    #: TODO: Change to some validation set !
    clusters = SpecialistDataset.cluster_classes(
        targets=targets,
        probs=gene_probs,
        cm=SpecialistDataset.cm_types['soft_sum_pred_cm'],
        nb_clusters=nb_clusters,
        clustering=SpecialistDataset.clustering_methods['greedy']
    )
    for i, c in enumerate(clusters):
        spec_probs, spec_targets = get_spec_probs(i, experiment, data, backend, len(c))
        final_probs = merge_predictions_weighted(
            generalist=gene_probs,
            specialist=spec_probs,
            cluster=c,
            final=final_probs,
        )
        print i, ': Spec logloss: ', spec_log_loss(spec_targets, spec_probs, c)
        print i, ': Spec accuracy: ', spec_accuracy(spec_targets, np.argmax(spec_probs, axis=1), c)
        print i, ': Final logloss: ', log_loss(targets, final_probs)
        print i, ': Final accuracy:', accuracy_score(targets, np.argmax(final_probs, axis=1))
    print 'The final results for the whole system are: '
    print 'Logloss: ', log_loss(targets, final_probs)
    print 'Accuracy: ', accuracy_score(targets, np.argmax(final_probs, axis=1))
    import pdb; pdb.set_trace()
