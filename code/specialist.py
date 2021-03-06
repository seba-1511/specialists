#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import cPickle as pk
from itertools import izip

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt

# from neon.datasets.dataset import Dataset
# from neon.util.persist import deserialize
# from neon.datasets import (
    # CIFAR10,
    # CIFAR100,
    # MNIST,
# )
# from neon.backends.cpu import CPU

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
    f = open(path + '/saved_experiments/' +
             name + '/validation-targets.pkl', 'rb')
    valid = pk.load(f)
    valid = np.argmax(test, axis=1)
    f.close()
    return (train, valid, test)


def load_inferences(path, name):
    f = open(path + '/saved_experiments/' +
             name + '/train-inference.pkl', 'rb')
    train = pk.load(f)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/test-inference.pkl', 'rb')
    test = pk.load(f)
    f.close()
    f = open(path + '/saved_experiments/' +
             name + '/validation-inference.pkl', 'rb')
    valid = pk.load(f)
    f.close()
    return (train, valid, test)


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
    cm = np.zeros((N, N))
    for pred, label in zip(preds, targets):
        cm[label, :] += pred[:N]
    return cm


def confuse_cm(targets, preds):
    # targets = np.argmax(targets, axis=1)
    preds = np.argmax(preds, axis=1)
    return confusion_matrix(targets, preds)


def soft_sum_pred_cm(targets, preds):
    """
        Returns a confusion matrix, where the content of a cell is the sum
        of prediction probabilities, only when this class was predicted.
    """
    N = max(targets) + 1
    cm = np.zeros((N, N))
    for pred, label in zip(preds, targets):
        if np.argmax(pred) == label:
            cm[label, :] += pred[:N]
    return cm


def soft_sum_not_pred_cm(targets, preds):
    N = max(targets) + 1
    cm = np.zeros((N, N))
    for pred, label in zip(preds, targets):
        if np.argmax(pred) != label:
            cm[label, :] += pred[:N]
    return cm


def soft_sum_n_pred_cm(targets, preds, n=5):
    """
        Same as soft_sum_pred_cm, but we consider the case when the class is in
        the top n predictions. (Might be usefull when a model is not so far to
        predict the right class)
    """
    pass


def spectral_clustering(matrix, N):
    spectral = SpectralClustering(n_clusters=N)
    clusters = spectral.fit_predict(matrix)
    res = [[] for _ in range(N)]
    for i, c in enumerate(clusters):
        res[c].append(i)
    return res


def kmeans_clustering(matrix, N):
    km = KMeans(n_clusters=N, n_jobs=-1)
    clusters = km.fit_predict(matrix)
    res = [[] for _ in range(N) ]
    for i, c in enumerate(clusters):
        res[c].append(i)
    return res


def unfriendliness_matrix(cm):
    """
        Returns a len(cm)**2 matrix friends, where an entry is the friendliness
        between two classes, based on a confusion matrix.
        (friends[i, j] = cm[i, j] + cm[j, i])
    """
    N = len(cm)
    return np.array(
        [[cm[i, j] + cm[j, i] for j in xrange(N)] for i in xrange(N)])


def adversity(friends_mat, cls, friends_set):
    """
        Returns the friends score for cls, with respect to a given friend set.
    """
    return sum([friends_mat[cls, f] for f in friends_set])

def non_diag_argmin(matrix):
    x_min, y_min, curr_min = None, None, np.inf
    for x, r in enumerate(matrix):
        for y, val in enumerate(r):
            if val < curr_min and x != y:
                curr_min = val
                x_min = x
                y_min = y
    return (x_min, y_min)

def greedy_clustering(friends, N):
    """
        Returns a list of N sets, so that each set is the most friendly as
        possible. Note that this does not allow for a class to be in two sets.
    """
    nb_classes = len(friends)
    clusters = [set() for _ in xrange(N)]
    seen = set()
    scores = friends.copy()
    while sum([len(c) for c in clusters]) < N:
        # That loop is to avoid each class ending in the same cluster.
        clss, most_friendly = non_diag_argmin(scores)
        friend_clstr = np.argmin([adversity(friends, most_friendly, s) if clss not in s else np.inf
                                  for s in clusters])
        added = False
        for c in clusters:
            if len(c) <= 0 or most_friendly in c:
                c.add(most_friendly)
                added = True
                break
        if not added:
            clusters[friend_clstr].add(most_friendly)

        clss_clstr = np.argmin([adversity(friends, most_friendly, s) if most_friendly not in s else np.inf
                                  for s in clusters])
        added = False
        for c in clusters:
            if len(c) <= 0 or clss in c:
                c.add(clss)
                added = True
                break
        if not added:
            clusters[clss_clstr].add(clss)
        seen.add(clss)
        seen.add(most_friendly)
        scores[clss][most_friendly] = np.max(scores)
        scores[most_friendly][clss] = np.max(scores)

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
                [adversity(friends, class1, s) for s in clusters])
            set_scores_2 = np.array(
                [adversity(friends, class2, s) for s in clusters])
            opt_set = np.argmax(set_scores_1 + set_scores_2)
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


def to_one_hot(i, n_classes):
    return np.array([0 if x != i else 1 for x in xrange(n_classes)])


def single_score(clss, adv, cluster):
    score = sum([adv[clss, c] for c in cluster])
    return float(score) / len(cluster)


def greedy_singles(cm, M):
    # N classes, M clusters
    N = len(cm)
    # Create and sort pair-scores
    assert cm.shape[0] == cm.shape[1]
    adv = cm + cm.T
    adverse = []
    for i in xrange(N):
        for j in xrange(i):
            adverse.append((adv[i, j], i, j))
    adverse = sorted(adverse, key=lambda s: s[0])
    # build clusters
    clusters = []
    seen = set()
    while len(clusters) != M:
        _, a, b = adverse.pop()
        if a in seen or b in seen:
            continue
        seen.add(a)
        seen.add(b)
        clusters.append([a, b, ])
    scores = np.argmax(adv, axis=1)
    clss = 0
    for i in seen: scores[i] = 0
    while len(seen) != N:
        clss = (clss + 1) % N
        if clss not in seen and scores[clss] == np.max(scores):
            score = adv[clss]
            c = np.argmin([single_score(clss, adv, c) for c in clusters])
            clusters[c].append(clss)
            seen.add(clss)
            scores[clss] = 0
    return clusters


def pair_score(a, b, adv, cluster):
    return (single_score(a, adv, cluster) + single_score(b, adv, cluster)) / 2.0


def greedy_pairs(cm, M):
    # N classes, M clusters
    N = len(cm)
    # Create and sort pair-scores
    assert cm.shape[0] == cm.shape[1]
    adv = cm + cm.T
    adverse = []
    for i in xrange(N):
        for j in xrange(i):
            adverse.append((adv[i, j], i, j))
    adverse = sorted(adverse, key=lambda s: s[0])
    # build clusters
    clusters = []
    seen = set()
    # Init the clusters so that non-empty
    while len(clusters) != M:
        _, a, b = adverse.pop()
        if a in seen or b in seen:
            adverse.insert(0, (_, a, b))
            continue
        seen.add(a)
        seen.add(b)
        clusters.append(set([a, b, ]))
    adverse = sorted(adverse, key=lambda s: s[0])
    # Fill with remaining classes:
    while len(adverse) != 0:
        _, a, b = adverse.pop()
        c = np.argmin([pair_score(a, b, adv, c) for c in clusters])
        clusters[c].add(a)
        clusters[c].add(b)
    return [list(c) for c in clusters]


class SpecialistDataset(object):

    """
         Class allowing to create a sub-dataset, from a given one.
         Especially usefull for specialists.
    """

    clustering_methods = {
        'greedy': greedy_clustering,
        'overlap_greedy': greedy_clustering_overlapping,
        'spectral': spectral_clustering,
        'kmeans': kmeans_clustering,
        'greedy_singles': greedy_singles,
        'greedy_pairs': greedy_pairs,
    }

    cm_types = {
        'standard': confuse_cm,
        'soft_sum': soft_sum_cm,
        'soft_sum_pred': soft_sum_pred_cm,
        'soft_sum_not_pred': soft_sum_not_pred_cm,
        'soft_sum_n_pred': soft_sum_n_pred_cm,
    }

    @classmethod
    def cluster_classes(cls, targets, probs, cm=None, nb_clusters=5,
                        clustering=None, friendly=True):
        cm = cm if cm else cls.cm_types['soft_sum_pred_cm']
        clustering = clustering if clustering else cls.clustering_methods[
            'greedy']
        cm = cm(targets, probs)
        cm = clean_cm(cm)
        if friendly:
            cm = unfriendliness_matrix(cm)
        cluster = clustering(cm, nb_clusters)
        return [list(c) for c in cluster]



if __name__ == '__main__':
    size = 10
    cm = np.random.random((size, size))
    greedy_pairs(cm, 3)
