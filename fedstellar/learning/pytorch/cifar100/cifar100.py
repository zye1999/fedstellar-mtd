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
from torchvision.datasets import CIFAR100


class CIFAR100Dataset(Dataset):
    def __init__(self, normalization="cifar100", loading="torchvision", sub_id=0, number_sub=1, num_workers=4, batch_size=32, iid=True, root_dir="./data"):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.iid = iid
        self.root_dir = root_dir
        self.loading = loading
        self.normalization = normalization

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
            dataset = CIFAR100(
                root=self.root_dir,
                train=train,
                transform=transform,
                download=download,
            )
        elif self.loading == "custom":
            raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset
