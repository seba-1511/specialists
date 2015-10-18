#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file creates the specialist dataset and trains a specialist network on it.
TODO: It could actually train all the specialists, and save all of the archives.
"""

import numpy as np
import random

from neon.backends import gen_backend
from neon.data import load_cifar10, DataIterator
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks

from a_train_generalist import split_train_set, EXPERIMENT_DIR, DATASET_NAME
from cifar_net import get_custom_vgg, get_allconv, get_dummy, get_mini_custom
from specialist import SpecialistDataset

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()


def filter_dataset(X, y, cluster):
    new_X = np.empty((0, X.shape[1]))
    new_y = np.empty((0, y.shape[1]))
    for c in cluster:
        idx = y == c
        idx = idx[:, 0]
        new_X = np.vstack((new_X, X[idx]))
        new_y = np.vstack((new_y, y[idx]))
    return (new_X, new_y, len(cluster))

if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    num_epochs = args.epochs
    num_epochs = 74 if num_epochs == 10 else num_epochs
    rng_seed = 1234
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    num_clusters = 3
    clustering_name = 'kmeans'
    confusion_matrix_name = 'standard'
    friendliness = False
    gene_net = get_custom_vgg
    gene_archive = 'generalist_val.prm'
    spec_net = get_mini_custom

    # setup backend
    be = gen_backend(
        backend=args.backend,
        batch_size=batch_size,
        rng_seed=rng_seed,
        device_id=args.device_id,
        default_dtype=args.datatype,
    )

    # Load and split datasets
    (X_train, y_train), (X_test,
                         y_test), nout = load_cifar10(path=args.data_dir)
    (X_train, y_train), (X_valid, y_valid) = split_train_set(X_train, y_train)
    nout = 16
    valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))

    # Get generalist predictions
    gene_path = EXPERIMENT_DIR + gene_archive
    generalist, opt, cost = gene_net(archive_path=gene_path, nout=nout)
    generalist.initialize(valid_set)
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

        # Create datasets
        X_spec, y_spec, spec_out = filter_dataset(X_train, y_train, cluster)
        spec_out = nout
        X_spec_test, y_spec_test, spec_out = filter_dataset(
            X_test, y_test, cluster)
        spec_set = DataIterator(
            X_spec, y_spec, nclass=spec_out, lshape=(3, 32, 32))
        spec_test = DataIterator(
            X_spec_test, y_spec_test, nclass=spec_out, lshape=(3, 32, 32))

        # Train the specialist
        # TODO: Load from archive so that the net is pre-trained
        # specialist, opt, cost = spec_net(nout=spec_out, archive_path=gene_path)
        specialist, opt, cost = spec_net(nout=spec_out)
        # specialist.epoch_index = 0
        opt.optimizer_mapping['default'].learning_rate *= 0.01
        opt.optimizer_mapping['Bias'].learning_rate *= 0.01
        callbacks = Callbacks(
            specialist, spec_set, output_file=args.output_file,
                             valid_set=spec_test, valid_freq=args.validation_freq,
                             progress_bar=True)
        # TODO: Add early stopping callback to make sure no overfitting.
        specialist.fit(spec_set, optimizer=opt,
                       num_epochs=specialist.epoch_index + num_epochs, cost=cost, callbacks=callbacks)

        # Print results
        print 'Train misclassification error: ', specialist.eval(spec_set, metric=Misclassification())
        print 'Test misclassification error: ', specialist.eval(spec_test, metric=Misclassification())
        specialists.append(specialist)
        save_obj(specialist.serialize(), EXPERIMENT_DIR + 'specialist' + str(i) + '_' +
                 confusion_matrix_name + '_' + clustering_name + '_' + str(num_clusters) + 'clusters.prm')