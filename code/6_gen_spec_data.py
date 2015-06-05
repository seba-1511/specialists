#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO:
#: Import cudanet backend
#: Import Datapar and try it

import os
import numpy as np
import cPickle as pk

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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


def clean_cm(cm):
    for i in xrange(len(cm)):
        cm[i, i] = 0
    return cm


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, xrange(len(cm)), rotation=45)
    plt.yticks(tick_marks, xrange(len(cm)))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.close()


def soft_sum_cm(targets, preds):
    """
        Return a confusion matrix, containing the sum of all probabilities for
        a given class.
   """
    N = max(targets) + 1
    nb_preds = len(preds)
    placeholder = np.zeros(nb_preds)
    cm = np.zeros((N, N))
    for label in xrange(N):
        entry = []
        for i in xrange(N):
            values = np.where(targets == label, preds[:, i], placeholder)
            entry.append(sum(values))
        cm[label] += entry
    return clean_cm(cm)


def soft_sum_pred_cm(targets, preds):
    """
        Returns a confusion matrix, where the the content of a cell is the sum
        of prediction probabilities, only when this class was predicted.
    """
    N = max(targets) + 1
    nb_preds = len(preds)
    placeholder = np.zeros(nb_preds)
    cm = np.zeros((N, N))
    for label in xrange(N):
        entry = []
        for i in xrange(N):
            values = np.where(targets == label, preds[:, i], placeholder)
            values = np.where(
                label == np.argmax(preds, axis=1), values, placeholder)
            entry.append(sum(values))
        cm[label] += entry
    return clean_cm(cm)


def soft_sum_n_pred_cm(targets, preds, n=5):
    """
        Same as soft_sum_pred_cm, but we consider the case when the class is in
        the top n predictions. (Might be usefull when a model is not so far to
        predict the right class)
    """
    pass


def unfriendliness_matrix(cm):
    """
        Returns a len(cm)**2 matrix friends, where an entry is the friendliness
        between two classes, based on a confusion matrix.
        (friends[i, j] = cm[i, j] + cm[j, i])
    """
    N = len(cm)
    return np.array(
        [[cm[i, j] + cm[j, i] for j in xrange(N)] for i in xrange(N)])


def set_score(friends_mat, cls, friends_set):
    """
        Returns the friends score for cls, with respect to a given friend set.
    """
    return sum([friends_mat[cls, f] for f in friends_set])


def greedy_clustering(friends, N):
    """
        Returns a list of N sets, so that each set is the most friendly as
        possible. Note that this does not allow for a class to be in two sets.
    """
    nb_classes = len(friends)
    clusters = [set() for _ in xrange(N)]
    seen = set()
    scores = friends.copy()
    while len(seen) < nb_classes:
        best = np.argmax(scores)
        class1, class2 = best = (best / nb_classes, best % nb_classes)
        if class1 in seen and class2 in seen:
            scores[class1, class2] = -1
            scores[class2, class1] = -1
            continue

        elif class1 in seen:
            opt_set = [i for i, s in enumerate(clusters) if class1 in s][0]
        elif class2 in seen:
            opt_set = [i for i, s in enumerate(clusters) if class2 in s][0]
        else:
            set_scores_1 = np.array(
                [set_score(friends, class1, s) for s in clusters])
            set_scores_2 = np.array(
                [set_score(friends, class2, s) for s in clusters])
            opt_set = np.argmin(set_scores_1 + set_scores_2)
        clusters[opt_set].add(class1)
        clusters[opt_set].add(class2)
        seen.add(class1)
        seen.add(class2)
        scores[class1, class2] = -1
        scores[class2, class1] = -1
    return clusters


def greedy_clustering_overlapping(friends, N):
    """
        Returns a list of N sets, so that each set is the most friendly as
        possible. Note that this allows for a class to be in two sets.
    """
    nb_classes = len(friends)
    clusters = [set() for _ in xrange(N)]
    seen = set()
    scores = friends.copy()
    while len(seen) < nb_classes:
        best = np.argmax(scores)
        class1, class2 = best = (best / nb_classes, best % nb_classes)
        if class1 in seen:
            opt_set = [i for i, s in enumerate(clusters) if class1 in s][0]
        elif class2 in seen:
            opt_set = [i for i, s in enumerate(clusters) if class2 in s][0]
        else:
            set_scores_1 = np.array(
                [set_score(friends, class1, s) for s in clusters])
            set_scores_2 = np.array(
                [set_score(friends, class2, s) for s in clusters])
            opt_set = np.argmin(set_scores_1 + set_scores_2)
        clusters[opt_set].add(class1)
        clusters[opt_set].add(class2)
        seen.add(class1)
        seen.add(class2)
        scores[class1, class2] = -1
        scores[class2, class1] = -1
    return clusters


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
    cm_categorical = clean_cm(cm_categorical)
    # cm_soft = soft_sum_cm(test_targets, test_pred_probs)
    #cm_soft = soft_sum_pred_cm(test_targets, test_pred_probs)
    friendliness = unfriendliness_matrix(cm_categorical)
    #plot_confusion_matrix(friendliness)
    clusters = greedy_clustering(friendliness, 5)

    # plot_confusion_matrix(cm_soft, title=experiment)
    # plot_confusion_matrix(cm_categorical, title=experiment)


    # To try when generating sets:
    # - overlapping vs no-overlapping
    # - greedy(+ friendliness) vs k-means vs spectral clustering
    # - Try for all the different kind of cm (cat, soft, soft_pred, soft_n_pred)
