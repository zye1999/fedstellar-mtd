#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#
from math import floor
import os
import sys
import torch

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")
from fedstellar.learning.pytorch.fedstellardataset import FedstellarDataset
from torchvision import transforms
from torchvision.datasets import MNIST


#######################################
#    FederatedDataModule for MNIST    #
#######################################


class MNISTDataset(FedstellarDataset):
    """
    LightningDataModule of partitioned MNIST.

    Args:
        sub_id: Subset id of partition. (0 <= sub_id < number_sub)
        number_sub: Number of subsets.
        batch_size: The batch size of the data.
        num_workers: The number of workers of the data.
        val_percent: The percentage of the validation set.
    """

    def __init__(
        self,
        num_classes=10,
        sub_id=0,
        number_sub=1,
        batch_size=32,
        num_workers=4,
        val_percent=0.1,
        iid=True,
        partition="dirichlet",
        seed=42,
        config=None,
    ):
        super().__init__(
            num_classes=num_classes,
            sub_id=sub_id,
            number_sub=number_sub,
            batch_size=batch_size,
            num_workers=num_workers,
            val_percent=val_percent,
            iid=iid,
            partition=partition,
            seed=seed,
            config=config,
        )
        if sub_id < 0 or sub_id >= number_sub:
            raise ValueError(
                f"sub_id {sub_id} is out of range for number_sub {number_sub}"
            )

        # Create data directory in fedstellar folder (if not exists)
        if not os.path.exists(f"{sys.path[0]}/data"):
            os.makedirs(f"{sys.path[0]}/data")

    def initialize_dataset(self):
        # Load MNIST train dataset
        if self.train_set is None:
            self.train_set = self.load_mnist_dataset(train=True)
        if self.test_set is None:
            self.test_set = self.load_mnist_dataset(train=False)

        # All nodes have the same test set (indices are the same for all nodes)
        self.test_indices_map = list(range(len(self.test_set)))

        # Depending on the iid flag, generate a non-iid or iid map of the train set
        if self.iid:
            self.train_indices_map = self.generate_iid_map(self.train_set)
        else:
            self.train_indices_map = self.generate_non_iid_map(
                self.train_set, self.partition
            )

        print(f"Length of train indices map: {len(self.train_indices_map)}")
        print(f"Lenght of test indices map: {len(self.test_indices_map)}")

    def load_mnist_dataset(self, train=True):
        apply_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        return MNIST(
            f"{sys.path[0]}/data",
            train=train,
            download=True,
            transform=apply_transforms,
        )

    def generate_non_iid_map(self, dataset, partition="dirichlet"):
        if partition == "dirichlet":
            partitions_map = self.dirichlet_partition(dataset, alpha=0.5)
        elif partition == "percent":
            # At now, percentage is fixed to 0.2
            partitions_map = self.percentage_partition(dataset, percentage=0.2)
        else:
            raise ValueError(f"Partition {partition} is not supported for Non-IID map")

        if self.sub_id == 0:
            self.plot_data_distribution(dataset, partitions_map)

        return partitions_map[self.sub_id]

    def generate_iid_map(self, dataset):
        # rows_by_sub = floor(len(dataset) / self.number_sub)
        # partitions_map = [
        #     range(i * rows_by_sub, (i + 1) * rows_by_sub)
        #     for i in range(self.number_sub)
        # ]
        # if self.sub_id == 0:
        #     self.plot_data_distribution(dataset, partitions_map)
        partitions_map = self.homo_partition(dataset)
        if self.sub_id == 0:
            self.plot_data_distribution(dataset, partitions_map)
        return partitions_map[self.sub_id]
