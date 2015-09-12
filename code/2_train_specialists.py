#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file creates the specialist dataset and trains a specialist network on it.
TODO: It could actually train all the specialists, and save all of the archives.
"""

import numpy as np

from neon.backends import gen_backend
from neon.data import load_cifar10, DataIterator
from neon.util.argparser import NeonArgparser
from neon.util.persist import save_obj
from neon.transforms.cost import Misclassification
from neon.callbacks.callbacks import Callbacks

from cifar_net import get_custom_vgg, get_allconv, get_dummy
from specialist import SpecialistDataset

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

# TODO: Import from
# from 1_train_generalist import split_train_set
def split_train_set(X_train, y_train):
    return (X_train[:-5000], y_train[:-5000]), (X_train[-5000:], y_train[-5000:])

def filter_dataset(X, y, clusters):
    return (X, y, len(clusters))


if __name__ == '__main__':
    # hyperparameters
    batch_size = 64
    num_epochs = args.epochs
    num_epochs = 74 if num_epochs == 10 else num_epochs
    rng_seed = 1234
    num_clusters = 2
    clustering = SpecialistDataset.clustering_methods['overlap_greedy']
    confusion_matrix = SpecialistDataset.cm_types['soft_sum_pred_cm']
    spec_net = get_dummy
    gene_net = get_dummy
    gene_archive = 'dummy.prm'

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
    # nout = 16
    valid_set = DataIterator(X_valid, y_valid, nclass=nout, lshape=(3, 32, 32))

    # Get generalist predictions
    generalist, opt, cost = gene_net(archive_path=gene_archive, nout=nout)
    gene_preds = []
    gene_targets = []
    # TODO: Check that the following works as expected
    for X, y in valid_set:
       gene_preds += generalist.fprop(X, inference=True).transpose().asnumpyarray().tolist()
       gene_targets += np.argmax(y.transpose().asnumpyarray(), axis=1).tolist()
    gene_preds = np.array(gene_preds)
    gene_targets = np.array(gene_targets)

    # Compute the clusters
    # TODO: Fix clustering, it doesn't work.
    clusters = SpecialistDataset.cluster_classes(
        probs=gene_preds,
        targets=gene_targets,
        clustering=clustering,
        cm=confusion_matrix,
        nb_clusters=num_clusters,
    )

    specialists = []
    X_train = np.vstack((X_train, X_valid))
    y_train = np.vstack((y_train, y_valid))

    # Train each specialist
    for cluster in clusters:

        # Create datasets
        X_spec, y_spec, spec_out = filter_dataset(X_train, y_train, cluster)
        X_spec_test, y_spec_test, spec_out = filter_dataset(X_test, y_test, cluster)
        spec_set = DataIterator(X_spec, y_spec, nclass=spec_out, lshape=(3, 32, 32))
        spec_test = DataIterator(X_spec_test, y_spec_test, nclass=spec_out, lshape=(3, 32, 32))

        # Train the specialist
        specialist, opt, cost = spec_net(nout=spec_out)
        callbacks = Callbacks(specialist, spec_set, output_file=args.output_file,
                          valid_set=spec_test, valid_freq=args.validation_freq,
                          progress_bar=args.progress_bar)
        specialist.fit(spec_set, optimizer=opt, num_epochs=num_epochs, cost=cost, callbacks=callbacks)

        # Print results
        print 'Train misclassification error: ', specialist.eval(spec_set, metric=Misclassification())
        print 'Test misclassification error: ', specialist.eval(spec_test, metric=Misclassification())
        specialists.append(specialist)

# TODO: Save specialists weights in some way.
