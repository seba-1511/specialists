#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file benchmarks saved and trained specialists and generalists on CIFARs
datasets.
"""

import numpy as np
from neon.backends import gen_backend
from neon.data import load_cifar10, DataIterator
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks

from sklearn.metrics import log_loss, accuracy_score

from a_train_generalist import split_train_set, EXPERIMENT_DIR
#from b_train_specialists import
from cifar_net import get_custom_vgg, get_allconv, get_dummy
from specialist import SpecialistDataset

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    rng_seed = 1234
    num_clusters = 4
    clustering_name = 'kmeans'
    confusion_matrix_name = 'soft_sum_pred_cm'
    spec_net = get_custom_vgg
    gene_net = get_allconv
    gene_archive = 'allconv.prm'

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    # Load and split datasets
    (X_train, y_train), (X_test, y_test), nout = load_cifar10(path=args.data_dir)
    (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
    test_set = DataIterator(X_test, y_test, nclass=nout_gene, lshape=(3, 32, 32))
    valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))

    # Get generalist predictions
    gene_path = EXPERIMENT_DIR + gene_archive
    generalist, opt, cost = gene_net(archive_path=gene_path, nout=nout)
    gene_preds = []
    gene_targets = []
    for X, y in valid_set:
        gene_preds += generalist.fprop(
            X, inference=True).asnumpyarray().transpose().tolist()
        gene_targets += np.argmax(
            y.asnumpyarray().transpose(), axis=1).tolist()
    gene_preds = np.array(gene_preds)
    gene_targets = np.array(gene_targets)

    # Compute the clusters
    # TODO: Fix clustering, it doesn't work.
    clustering = SpecialistDataset.clustering_methods[clustering_name]
    confusion_matrix = SpecialistDataset.cm_types[confusion_matrix_name]
    clusters = SpecialistDataset.cluster_classes(
        probs=gene_preds,
        targets=gene_targets,
        clustering=clustering,
        cm=confusion_matrix,
        nb_clusters=num_clusters,
    )
    del gene_preds
    del gene_targets
    del valid_set
    print 'Clusters: ', clusters

    # Load each specialist
    specialists = []
    for c in range(num_clusters):
        path = EXPERIMENT_DIR + 'specialist' + str(c) + '_' + confusion_matrix_name + '_' + clustering_name + '_' + str(num_clusters) + 'clusters.prm'
        specialists.append(spec_net(archive_path=path))

    # Merge all predictions together
    final_pred = np.empty((1, 1))
    final_targets = np.empty((1, 1))
    for X, y in test_set:
        gene_preds = generalist.fprop(X, inference=True).asnumpyarray().transpose()
        for clss in range(nout):
            for spec, cluster in zip(specialists, clusters):
                if clss in cluster:
                    mask = y.asnumpyarray() == clss
                    spec_pred = spec.fprop(X, inference=True).asnumpyarray().transpose()
                    spec_pred = np.dot(mask, spec_pred)
                    # Note: We are not weighting anything.
                    gene_pred += spec_pred
        final_pred = np.vstack((final_pred, gene_pred), axis=1)
        final_targets = np.vstack((final_targets, y), axis=1)

    # Print scores
    acc = accuracy_score(final_targets, final_pred)
    ll = log_loss(final_targets, final_pred)
    gene_acc = accuracy_score(final_targets, gene_preds)
    gene_ll = log_loss(final_targets, gene_preds)
    print '-'*40
    print 'Generalist Accuracy: ', gene_acc
    print 'Generalist LogLoss: ', gene_ll
    print '-'*40
    print 'Final Accuracy: ', acc
    print 'Final LogLoss: ', ll
