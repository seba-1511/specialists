#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import cPickle as pk
from itertools import izip

from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, SpectralClustering
#import matplotlib.pyplot as plt

from neon.backends.par import NoPar
from neon.datasets.dataset import Dataset
from neon.datasets import (
    CIFAR10,
    CIFAR100,
    MNIST,
)
from neon.backends.cpu import CPU

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


#def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    #tick_marks = np.arange(len(cm))
    #plt.xticks(tick_marks, xrange(len(cm)), rotation=45)
    #plt.yticks(tick_marks, xrange(len(cm)))
    #plt.tight_layout()
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.show()
    #plt.close()


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
    return cm


def soft_sum_pred_cm(targets, preds):
    """
        Returns a confusion matrix, where the content of a cell is the sum
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
    return cm


def soft_sum_n_pred_cm(targets, preds, n=5):
    """
        Same as soft_sum_pred_cm, but we consider the case when the class is in
        the top n predictions. (Might be usefull when a model is not so far to
        predict the right class)
    """
    pass


def spectral_clustering(matrix, N):
    spectral = SpectralClustering(nb_clusters=N, n_jobs=-1)
    clusters = spectral.fit_predict(matrix)
    res = [[], ] * N
    for i, c in enumerate(clusters):
        res[c].append(i)
    return res




def kmeans_clustering(matrix, N):
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


def to_one_hot(i, n_classes):
    return np.array([0 if x != i else 1 for x in xrange(n_classes)])


class SpecialistDataset(Dataset):

    """
         Class allowing to create a sub-dataset, from a given one.
         Especially usefull for specialists.
    """

    clustering_methods = {
        'greedy': greedy_clustering,
        'overlap_greedy': greedy_clustering_overlapping,
        'spectral': None,
        'kmeans': None,
    }
    cm_types = {
        'standard': confusion_matrix,
        'soft_sum': soft_sum_cm,
        'soft_sum_pred_cm': soft_sum_pred_cm,
        'soft_sum_n_pred': soft_sum_n_pred_cm,
    }

    def __init__(self, dataset=None, experiment='', nb_clusters=5, cluster=0,
                 confusion_matrix='soft_sum_pred_cm', clustering='greedy',
                 repo_path='~/data', inferences_path='~/inferences',
                 full_predictions=False, **kwargs):
        """
            dataset: which dataset to sub-set, should be an Dataset() instance.
            experiment: on which experiment should the clustering process be
                        based, and the inferences loaded.
            nb_clusters: total number of clusters.
            cluster: which cluster to use for this current experiment.
        """
        self.repo_path = repo_path
        self.__dict__ = dataset.__dict__
        self.dataset = dataset
        self.experiment = experiment
        self.nb_clusters = nb_clusters
        self.cluster = cluster
        self.confusion_matrix = self.cm_types[confusion_matrix]
        self.clustering = self.clustering_methods[clustering]
        self.inferences_path = inferences_path
        self.full_predictions = full_predictions

    @classmethod
    def cluster_classes(cls, targets, probs, cm=None, nb_clusters=5,
                        clustering=None):
        cm = cm if cm else cls.cm_types['soft_sum_pred_cm']
        clustering = clustering if clustering else cls.clustering_methods[
            'greedy']
        cm = cm(targets, probs)
        cm = clean_cm(cm)
        friendliness = unfriendliness_matrix(cm)
        cluster = clustering(friendliness, nb_clusters)
        return [list(c) for c in cluster]

    def load(self, backend, experiment):
        self.dataset.load(backend, experiment)
        train_probs, test_probs = load_inferences(
            path=self.inferences_path, name=self.experiment)
        train_targets, test_targets = load_targets(
            path=self.inferences_path, name=self.experiment)
        #: TODO: Change test_* to validation set. (Necessary !)
        cluster = SpecialistDataset.cluster_classes(test_targets, test_probs,
                                                    cm=self.confusion_matrix,
                                                    nb_clusters=self.nb_clusters,
                                                    clustering=self.clustering)[self.cluster]
        cluster = list(cluster)
        n_classes = len(cluster)
        print '\n\nNumber of classes: ', n_classes, '\n\n'
        new_inputs = []
        new_targets = []
        for bi, bt in izip(self.dataset.inputs['train'], self.dataset.targets['train']):
            bi = bi.asnumpyarray().transpose()
            bt = bt.asnumpyarray().transpose()
            for i, t in izip(bi, bt):
                if np.argmax(t) in cluster:
                    clss = np.argmax(t)
                    clss = cluster.index(clss)
                    new_inputs.append(i)
                    new_targets.append(to_one_hot(clss, n_classes))
        self.dataset.inputs['train'] = np.array(new_inputs)
        self.dataset.targets['train'] = np.array(new_targets)
        new_inputs = []
        new_targets = []
        for bi, bt in izip(self.dataset.inputs['test'], self.dataset.targets['test']):
            bi = bi.asnumpyarray().transpose()
            bt = bt.asnumpyarray().transpose()
            for i, t in izip(bi, bt):
                if np.argmax(t) in cluster or self.full_predictions:
                    clss = np.argmax(t)
                    clss = cluster.index(clss) if clss in cluster else 0
                    new_inputs.append(i)
                    new_targets.append(to_one_hot(clss, n_classes))
        self.dataset.inputs['test'] = np.array(new_inputs)
        self.dataset.targets['test'] = np.array(new_targets)
        self.inputs = self.dataset.inputs
        self.targets = self.dataset.targets
        self.format()


# if __name__ == '__main__':
    # model = load_cifar100_train32_test50()
    # data, par = load_data()
    # targets = data.targets['test']
    # pred_probs = model.predict_proba(data.inputs['test'])

    # experiment = '5_test50_train33_156epochs'
    # experiment = '5_test45_train22_740epochs'
    # experiment = '4_test22_train14_74epochs'
    # train_pred_probs, test_pred_probs = load_inferences(name=experiment)
    # train_targets, test_targets = load_targets(name=experiment)
    # train_pred_classes = np.argmax(train_pred_probs, axis=1)
    # test_pred_classes = np.argmax(test_pred_probs, axis=1)
    # cm_categorical = confusion_matrix(test_targets, test_pred_classes)
    # cm_categorical = clean_cm(cm_categorical)
    # cm_soft = soft_sum_cm(test_targets, test_pred_probs)
    # cm_soft = soft_sum_pred_cm(test_targets, test_pred_probs)
    # friendliness = unfriendliness_matrix(cm_soft)
    # plot_confusion_matrix(friendliness)
    # clusters = greedy_clustering(friendliness, 5)

    # plot_confusion_matrix(cm_soft, title=experiment)
    # plot_confusion_matrix(cm_categorical, title=experiment)

    # sdata = SpecialistDataset(dataset='cifar100', experiment=experiment)

    # To try when generating sets:
    # - overlapping vs no-overlapping
    # - greedy(+ friendliness) vs k-means vs spectral clustering
    # - Try for all the different kind of cm (cat, soft, soft_pred, soft_n_pred)
    # - and compare on datasets
    # - (in greedy_clustering, if classes not seen, should it be argmax or argmin ? )
    # - When making predictions, do it on the whole test set not the cluster subset.
