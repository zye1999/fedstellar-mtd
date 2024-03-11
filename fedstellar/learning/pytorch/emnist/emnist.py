# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
from math import floor

import lightning as pl
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import EMNIST


class EMNISTDataset(Dataset):
    def __init__(self, loading="torchvision", iid=True, root_dir="./data"):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.iid = iid
        self.root_dir = root_dir
        self.loading = loading

        self.train_set = self.get_dataset(
            train=True,
            transform=T.ToTensor()
        )

        self.test_set = self.get_dataset(
            train=False,
            transform=T.ToTensor()
        )

        if not self.iid:
            # if non-iid, sort the dataset
            self.train_set = self.sort_dataset(self.train_set)
            self.test_set = self.sort_dataset(self.test_set)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset

    def get_dataset(self, train, transform, download=True):
        if self.loading == "torchvision":
            dataset = EMNIST(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
                split="digits"
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
