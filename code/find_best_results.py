#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file benchmarks saved and trained specialists and generalists on CIFARs
datasets.
"""

import random
import numpy as np
from neon.backends import gen_backend
from neon.data import load_cifar10, DataIterator
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks

from sklearn.metrics import log_loss, accuracy_score

from a_train_generalist import split_train_set, EXPERIMENT_DIR, DATASET_NAME, load_data
from cifar_net import get_custom_vgg, get_allconv, get_dummy
from specialist import SpecialistDataset

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()


def weighted_preds_merge(gene_preds, spec_preds, final_preds, cluster, spec_weight):
    preds = final_preds
    for i, pred in enumerate(preds):
        p = np.argmax(pred)
        if p in cluster:
            preds[i] += spec_weight * spec_preds[i]
    return preds


if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    rng_seed = 1234
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    num_clusters = 2
    clustering_name = 'spectral'
    confusion_matrix_name = 'standard'
    spec_net = get_custom_vgg
    gene_net = get_custom_vgg
    gene_archive = 'generalist_val.prm'

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    # Main Loop:
    clusterings = ['spectral', 'kmeans', 'greedy_singles', 'greedy_pairs']
    cm_names = ['standard', 'soft_sum', 'soft_sum_pred', 'soft_sum_not_pred']
    # nb_clusters_list = range(2, 5, 1)
    nb_clusters_list = range(2, 30, 4)
    main_results = {}
    for clustering_name in clusterings:
        main_results[clustering_name] = {}
        for confusion_matrix_name in cm_names:
            main_results[clustering_name][confusion_matrix_name] = {}
            for num_clusters in nb_clusters_list:
                try:
                    np.random.seed(rng_seed)
                    random.seed(rng_seed)
                    main_results[clustering_name][confusion_matrix_name][num_clusters] = {}
                    # Load and split datasets
                    (X_train, y_train), (X_test, y_test), nout = load_data(DATASET_NAME)
                    (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
                    test_set = DataIterator(X_test, y_test, nclass=nout, lshape=(3, 32, 32))
                    valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))

                    # Get generalist predictions
                    gene_path = EXPERIMENT_DIR + gene_archive
                    generalist, opt, cost = gene_net(archive_path=gene_path, nout=nout)
                    generalist.cost = cost
                    generalist.optimizer = opt
                    generalist.initialize(valid_set, cost)
                    gene_preds = generalist.get_outputs(valid_set)
                    gene_targets = y_valid
                    print 'Generalist score: ', accuracy_score(gene_targets, np.argmax(gene_preds, axis=1))

                    # Compute the clusters
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
                    # specialists = []
                    # for c in range(num_clusters):
                        # path = EXPERIMENT_DIR + confusion_matrix_name + '_' + clustering_name + '_' + str(num_clusters) + 'clusters/' + 'specialist' + '_' + str(c) + '.prm'
                        # s, opt, cost = spec_net(archive_path=path, nout=nout)
                        # s.cost = cost
                        # s.optimizer = opt
                        # s.initialize(test_set, cost)
                        # specialists.append(s)

                    # # Merge all predictions together
                    # final_targets = y_test
                    # gene_preds = generalist.get_outputs(test_set)
                    # spec_preds = [s.get_outputs(test_set) for s in specialists]
                    # final_preds = weighted_preds_merge(gene_preds, spec_preds, clusters, 0.5, 0.5)

                    # Testing
                    final_targets = y_test
                    gene_preds = generalist.get_outputs(test_set)
                    final_preds = 0.0 * gene_preds
                    for i, cluster in enumerate(clusters):
                        path = EXPERIMENT_DIR + confusion_matrix_name + '_' + clustering_name + '_' + str(num_clusters) + 'clusters/' + 'specialist' + '_' + str(i) + '.prm'
                        s, opt, cost = spec_net(archive_path=path, nout=nout)
                        s.cost = cost
                        s.optimizer = opt
                        s.initialize(test_set, cost)
                        spec_preds = s.get_outputs(test_set)
                        final_preds = weighted_preds_merge(gene_preds, spec_preds, final_preds, cluster, 1.0)

                    # End testing

                    nclasses = np.max(final_targets) + 1
                    gene_preds = gene_preds[:, :nclasses]
                    final_preds = final_preds[:, :nclasses]

                    # Print scores
                    acc = accuracy_score(final_targets, np.argmax(final_preds, axis=1))
                    ll = log_loss(final_targets, final_preds)
                    gene_acc = accuracy_score(final_targets, np.argmax(gene_preds, axis=1))
                    gene_ll = log_loss(final_targets, gene_preds)
                    print '-'*40
                    print 'Generalist Accuracy: ', gene_acc
                    print 'Generalist LogLoss: ', gene_ll
                    print '-'*40
                    print 'Final Accuracy: ', acc
                    print 'Final LogLoss: ', ll
                    print '-'*40
                    main_results[clustering_name][confusion_matrix_name][num_clusters]['accuracy'] = acc
                    main_results[clustering_name][confusion_matrix_name][num_clusters]['logloss'] = ll
                except:
                    main_results[clustering_name][confusion_matrix_name][num_clusters]['accuracy'] = 0.0
                    main_results[clustering_name][confusion_matrix_name][num_clusters]['logloss'] = 0.0

    # We explore the table and get the best values:
    # Ordered: (score, clustering, cm, nb_clusters)
    top_results = []
    table_results = {clustering: {
        name: (0.0, None) for name in cm_names
    } for clustering in clusterings}
    for clustering in clusterings:
        for name in cm_names:
            for n_cluster in nb_clusters_list:
                res = round(main_results[clustering][name][n_cluster]['accuracy'], 4)
                if res > table_results[clustering][name][0]:
                    table_results[clustering][name] = (res, n_cluster)
                top_results.append((str.format('{0:.4f}', res), clustering, name, n_cluster))

    # Nicely Printed:
    print '-' * 50
    print 'FINAL RESULTS: '
    headers = [' ' * 20, ] + [clus.ljust(20, ' ') for clus in cm_names]
    print ' '.join(headers)
    for c in table_results.keys():
        line = [str(table_results[c][b]).ljust(20, ' ') for b in cm_names]
        line = [c.ljust(20, ' '), ] + line
        print ' '.join(line)
    print ' '
    print '-' * 50
    print 'Ordered best results: '
    top_results = reversed(sorted(top_results, key=lambda r: r[0]))
    for t in top_results:
        print t
