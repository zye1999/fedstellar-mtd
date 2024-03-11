from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset


class FedstellarDataset(Dataset, ABC):
    """
    Abstract class for a partitioned dataset.

    Classes inheriting from this class need to implement specific methods
    for loading and partitioning the dataset.
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
        super().__init__()
        self.num_classes = num_classes
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.iid = iid
        self.partition = partition
        self.seed = seed
        self.config = config

        self.train_set = None
        self.train_indices_map = None
        self.test_set = None
        self.test_indices_map = None

        self.initialize_dataset()

    @abstractmethod
    def initialize_dataset(self):
        """
        Initialize the dataset. This should load or create the dataset.
        """
        pass

    @abstractmethod
    def generate_non_iid_map(self, dataset, partition="dirichlet"):
        """
        Create a non-iid map of the dataset.
        """
        pass

    @abstractmethod
    def generate_iid_map(self, dataset):
        """
        Create an iid map of the dataset.
        """
        pass

    def plot_data_distribution(self, dataset, partitions_map):
        """
        Plot the data distribution of the dataset.

        Plot the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        # Plot the data distribution of the dataset, one graph per partition
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        for i in range(self.number_sub):
            indices = partitions_map[i]
            class_counts = [0] * self.num_classes
            for idx in indices:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Participant {i+1} class distribution: {class_counts}")
            plt.figure()
            plt.bar(range(self.num_classes), class_counts)
            plt.xlabel("Class")
            plt.ylabel("Number of samples")
            plt.xticks(range(self.num_classes))
            plt.title(f"Partition {i+1} class distribution {'(IID)' if self.iid else '(Non-IID)'}{' - ' + self.partition if not self.iid else ''}")
            plt.tight_layout()
            path_to_save = f"{self.config.participant['tracking_args']['log_dir']}/{self.config.participant['scenario_args']['name']}/participant_{i+1}_class_distribution_{'iid' if self.iid else 'non_iid'}{'_' + self.partition if not self.iid else ''}.png"
            plt.savefig(
                path_to_save, dpi=300, bbox_inches="tight"
            )
            plt.close()

    def dirichlet_partition(self, dataset, alpha=0.5):
        """
        Partition the dataset into multiple subsets using a Dirichlet distribution.

        This function divides a dataset into a specified number of subsets (federated clients),
        where each subset has a different class distribution. The class distribution in each
        subset is determined by a Dirichlet distribution, making the partition suitable for 
        simulating non-IID (non-Independently and Identically Distributed) data scenarios in 
        federated learning.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.
            alpha (float): The concentration parameter of the Dirichlet distribution. A lower 
                        alpha value leads to more imbalanced partitions.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function ensures that each class is represented in each subset but with varying 
        proportions. The partitioning process involves iterating over each class, shuffling 
        the indices of that class, and then splitting them according to the Dirichlet 
        distribution. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = dirichlet_partition(my_dataset, alpha=0.5)
            # This creates federated data subsets with varying class distributions based on
            # a Dirichlet distribution with alpha = 0.5.
        """
        np.random.seed(self.seed)
        X_train, y_train = dataset.data.numpy(), dataset.targets.numpy()
        min_size = 0
        K = 10
        N = y_train.shape[0]
        n_nets = self.number_sub
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        # partitioned_datasets = []
        for i in range(self.number_sub):
            #    subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            #    partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            # print(f"Partition {i+1} class distribution: {class_counts}")

        return net_dataidx_map

    def homo_partition(self, dataset):
        """
        Homogeneously partition the dataset into multiple subsets.

        This function divides a dataset into a specified number of subsets, where each subset
        is intended to have a roughly equal number of samples. This method aims to ensure a
        homogeneous distribution of data across all subsets. It's particularly useful in 
        scenarios where a uniform distribution of data is desired among all federated learning 
        clients.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function randomly shuffles the entire dataset and then splits it into the number 
        of subsets specified by `number_sub`. It ensures that each subset has a similar number
        of samples. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = homo_partition(my_dataset)
            # This creates federated data subsets with homogeneous distribution.
        """
        n_nets = self.number_sub

        n_train = dataset.data.shape[0]
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        # partitioned_datasets = []
        for i in range(self.number_sub):
            # subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            # partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Partition {i+1} class distribution: {class_counts}")

        return net_dataidx_map

    def percentage_partition(self, dataset, percentage=0.1):
        """
        Partition a dataset into multiple subsets with a specified level of non-IID-ness.

        This function divides a dataset into several subsets where each subset has a 
        different class distribution, controlled by the 'percentage' parameter. The 
        'percentage' parameter determines the degree of non-IID-ness in the label distribution
        among the federated data subsets.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have 
                                                'data' and 'targets' attributes.
            percentage (float): A value between 0 and 100 that specifies the desired 
                                level of non-IID-ness for the labels of the federated data. 
                                This percentage controls the imbalance in the class distribution 
                                across different subsets.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to number_sub-1) 
                and values are lists of indices corresponding to the samples in each subset.

        The function uses a Dirichlet distribution to create imbalanced proportions of classes
        in each subset. A higher 'percentage' value leads to a more pronounced imbalance in 
        class distribution across subsets. The function ensures that each subset is shuffled 
        and has a unique distribution of classes.

        Example usage:
            federated_data = percentage_partition(my_dataset, 20)
            # This creates federated data subsets with a 20% level of non-IID-ness.
        """
        np.random.seed(self.seed)
        
        if isinstance(dataset.data, np.ndarray):
            X_train = dataset.data
        elif hasattr(dataset.data, 'numpy'):  # Check if it's a tensor with .numpy() method
            X_train = dataset.data.numpy()
        else:  # If it's a list
            X_train = np.asarray(dataset.data)

        if isinstance(dataset.targets, np.ndarray):
            y_train = dataset.targets
        elif hasattr(dataset.targets, 'numpy'):  # Check if it's a tensor with .numpy() method
            y_train = dataset.targets.numpy()
        else:  # If it's a list
            y_train = np.asarray(dataset.targets)
        
        num_classes = self.num_classes
        num_subsets = self.number_sub
        class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}

        imbalance_factor = percentage / 100

        subset_indices = [[] for _ in range(num_subsets)]

        for class_idx in range(num_classes):
            indices = class_indices[class_idx]
            np.random.shuffle(indices)
            num_samples_class = len(indices)

            proportions = np.random.dirichlet(np.repeat(1.0 + imbalance_factor, num_subsets))
            proportions = (np.cumsum(proportions) * num_samples_class).astype(int)[:-1]
            split_indices = np.split(indices, proportions)

            for i, subset_idx in enumerate(split_indices):
                subset_indices[i].extend(subset_idx)

        for i in range(num_subsets):
            np.random.shuffle(subset_indices[i])

            class_counts = [0] * num_classes
            for idx in subset_indices[i]:
                _, label = dataset[idx]
                class_counts[label] += 1
            print(f"Partition {i+1} class distribution: {class_counts}")

        partitioned_datasets = {i: subset_indices[i] for i in range(num_subsets)}

        return partitioned_datasets
