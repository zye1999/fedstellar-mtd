# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import MNIST

#######################################
#    MNISTDatasetScikit for MNIST     #
#######################################


class MNISTDatasetScikit:
    """
    Dataset for MNIST using torchvision, but provides data in a format for scikit-learn.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        val_percent: The percentage of the validation set.
        iid: If True, data distribution is iid, otherwise non-iid.
    """

    mnist_train = None
    mnist_val = None

    def __init__(self, sub_id=0, number_sub=1, val_percent=0.1, iid=True):
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.val_percent = val_percent
        self.iid = iid

        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

        if MNISTDatasetScikit.mnist_train is None:
            MNISTDatasetScikit.mnist_train = MNIST(
                f"{sys.path[0]}/data", train=True, download=True, transform=transforms.ToTensor()
            )
            if not iid:
                sorted_indexes = MNISTDatasetScikit.mnist_train.targets.sort()[1]
                MNISTDatasetScikit.mnist_train.targets = (
                    MNISTDatasetScikit.mnist_train.targets[sorted_indexes]
                )
                MNISTDatasetScikit.mnist_train.data = MNISTDatasetScikit.mnist_train.data[
                    sorted_indexes
                ]

        if MNISTDatasetScikit.mnist_val is None:
            MNISTDatasetScikit.mnist_val = MNIST(
                f"{sys.path[0]}/data", train=False, download=True, transform=transforms.ToTensor()
            )
            if not iid:
                sorted_indexes = MNISTDatasetScikit.mnist_val.targets.sort()[1]
                MNISTDatasetScikit.mnist_val.targets = MNISTDatasetScikit.mnist_val.targets[
                    sorted_indexes
                ]
                MNISTDatasetScikit.mnist_val.data = MNISTDatasetScikit.mnist_val.data[
                    sorted_indexes
                ]

        self.train_set = MNISTDatasetScikit.mnist_train
        self.test_set = MNISTDatasetScikit.mnist_val

    def train_dataloader(self):
        X_train = self.train_set.data.numpy().reshape(-1, 28 * 28)
        y_train = self.train_set.targets.numpy()
        return X_train, y_train

    def test_dataloader(self):
        X_test = self.test_set.data.numpy().reshape(-1, 28 * 28)
        y_test = self.test_set.targets.numpy()
        return X_test, y_test
