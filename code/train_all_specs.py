#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file creates the specialist dataset and trains a specialist network on it.
"""

import numpy as np
import random

from a_train_generalist import split_train_set, EXPERIMENT_DIR, DATASET_NAME, load_data
from cifar_net import get_custom_vgg, get_allconv, get_dummy, get_mini_custom
from neon.backends import gen_backend
from neon.callbacks.callbacks import Callbacks
from neon.data import load_cifar10, DataIterator
from neon.transforms.cost import Misclassification
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from sklearn.metrics import accuracy_score
from specialist import SpecialistDataset

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()


def early_stop(params, valid):
    if params is None:
        prev_min, time_length = np.inf, 0
    else:
        prev_min, time_length = params
    if prev_min < valid:
        time_length += 1
    else:
        time_length = 0
    prev_min = min(prev_min, valid)
    return ((prev_min, time_length), time_length >= 3)


def filter_dataset(X, y, cluster):
    new_X = []
    new_y = []
    for x, label in zip(X, y):
        if label in cluster:
            new_X.append(x)
            new_y.append(label)
    new_X = np.array(new_X)
    new_y = np.array(new_y)
    return (new_X, new_y, len(cluster))

if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    num_epochs = args.epochs
    num_epochs = 20
    rng_seed = 1234
    friendliness = False
    gene_net = get_custom_vgg
    gene_archive = 'generalist_val.prm'
    spec_net = get_custom_vgg

    num_clusters = 3
    clustering_name = 'spectral'
    confusion_matrix_name = 'standard'
    for confusion_matrix_name in ['standard', ]:
        for clustering_name in ['spectral', 'kmeans', ]:
            # for num_clusters in xrange(2, 30, 2):
            for num_clusters in xrange(5, 6, 2):
                np.random.seed(rng_seed)
                random.seed(rng_seed)

                # setup backend
                be = gen_backend(
                    backend=args.backend,
                    batch_size=batch_size,
                    rng_seed=rng_seed,
                    device_id=args.device_id,
                    default_dtype=args.datatype,
                )

                # Load and split datasets
                (X_train, y_train), (X_test, y_test), nout = load_data(DATASET_NAME)
                (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
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
                    friendly=friendliness,
                )
                del gene_preds
                del gene_targets
                del valid_set
                print clusters

                specialists = []
                X_train = np.vstack((X_train, X_valid))
                y_train = np.vstack((y_train, y_valid))

                # Train each specialist
                for i, cluster in enumerate(clusters):
                    print 'Training specialist: ', i
                    path = EXPERIMENT_DIR + confusion_matrix_name + '_' + clustering_name + '_' + str(num_clusters) + 'clusters/' + 'specialist' + '_' + str(i) + '.prm'

                    # Create datasets
                    X_spec, y_spec, spec_out = filter_dataset(X_train, y_train, cluster)
                    X_spec_test, y_spec_test, spec_out = filter_dataset(
                        X_test, y_test, cluster)
                    spec_out = nout
                    spec_set = DataIterator(
                        X_spec, y_spec, nclass=spec_out, lshape=(3, 32, 32))
                    spec_test = DataIterator(
                        X_spec_test, y_spec_test, nclass=spec_out, lshape=(3, 32, 32))

                    # Train the specialist
                    specialist, opt, cost = spec_net(nout=spec_out, archive_path=gene_path)
                    callbacks = Callbacks(specialist, spec_set, args, eval_set=spec_test)
                    callbacks.add_early_stop_callback(early_stop)
                    callbacks.add_save_best_state_callback(path)
                    specialist.fit(spec_set, optimizer=opt,
                                num_epochs=specialist.epoch_index + num_epochs, cost=cost, callbacks=callbacks)

                    # Print results
                    print 'Specialist Train misclassification error: ', specialist.eval(spec_set, metric=Misclassification())
                    print 'Specialist Test misclassification error: ', specialist.eval(spec_test, metric=Misclassification())
                    print 'Generalist Train misclassification error: ', generalist.eval(spec_set, metric=Misclassification())
                    print 'Generalist Test misclassification error: ', generalist.eval(spec_test, metric=Misclassification())
                    specialists.append(specialist)
                    save_obj(specialist.serialize(), path)

