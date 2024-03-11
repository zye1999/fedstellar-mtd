#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Chao Feng.
#
import os
import zipfile
import ast
from math import floor

import pandas as pd
# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, random_split, Dataset
from torchvision.datasets import MNIST, utils

torch.multiprocessing.set_sharing_strategy("file_system")


class SYSCALL(MNIST):
    def __init__(self, sub_id, number_sub, root_dir, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root_dir, transform=None, target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.download = download
        self.download_link = 'https://files.ifi.uzh.ch/CSG/research/fl/data/syscall.zip'
        self.train = train
        self.root = root_dir
        self.training_file = f'{self.root}/syscall/processed/syscall_train.pt'
        self.test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        if not os.path.exists(f'{self.root}/syscall/processed/syscall_test.pt') or not os.path.exists(f'{self.root}/syscall/processed/syscall_train.pt'):
            if self.download:
                self.dataset_download()
                self.process()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')
        else:
            print('SYSCALL dataset already downloaded and processed.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        # Whole dataset
        data_and_targets = torch.load(data_file)
        self.data, self.targets = data_and_targets[0], data_and_targets[1]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = img
        if self.target_transform is not None:
            target = target
        return img, target

    def dataset_download(self):
        paths = [f'{self.root}/syscall/raw/', f'{self.root}/syscall/processed/']
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

        # download files only if they do not exist
        print('Downloading SYSCALL dataset...')
        filename = self.download_link.split('/')[-1]
        utils.download_and_extract_archive(self.download_link, download_root=f'{self.root}/syscall/raw/', filename=filename)

        with zipfile.ZipFile(f'{self.root}/syscall/raw/{filename}', 'r') as zip_ref:
            zip_ref.extractall(f'{self.root}/syscall/raw/')

    def process(self):
        print('Processing SYSCALL dataset...')
        df = pd.DataFrame()
        files = os.listdir(f'{self.root}/syscall/raw/')
        feature_name = 'system calls frequency_1gram-scaled'
        for f in files:
            if '.csv' in f:
                fi_path = f'{self.root}/syscall/raw/{f}'
                csv_df = pd.read_csv(fi_path, sep='\t')
                feature = [ast.literal_eval(i) for i in csv_df[feature_name]]
                csv_df[feature_name] = feature
                df = pd.concat([df, csv_df])
        df['maltype'] = df['maltype'].replace(to_replace='normalv2', value='normal')
        classes_to_targets = {}
        t = 0
        for i in set(df['maltype']):
            classes_to_targets[i] = t
            t += 1
        classes = list(classes_to_targets.keys())

        for c in classes_to_targets:
            df['maltype'] = df['maltype'].replace(to_replace=c, value=classes_to_targets[c])

        all_targes = torch.tensor(df['maltype'].tolist())
        all_data = torch.tensor(df[feature_name].tolist())

        x_train, x_test, y_train, y_test = train_test_split(all_data, all_targes, test_size=0.15, random_state=42)
        train = [x_train, y_train, classes_to_targets, classes]
        test = [x_test, y_test, classes_to_targets, classes]

        # save to files
        train_file = f'{self.root}/syscall/processed/syscall_train.pt'
        test_file = f'{self.root}/syscall/processed/syscall_test.pt'

        # save to processed dir
        if not os.path.exists(train_file):
            torch.save(train, train_file)
        if not os.path.exists(test_file):
            torch.save(test, test_file)


#######################################
#    SYSCALLDataset for SYSCALL    #
#######################################


class SYSCALLDataset(Dataset):
    """
    LightningDataModule of partitioned SYSCALL.

    Args:

    """

    def __init__(
            self,
            sub_id=0,
            number_sub=1,
            batch_size=32,
            num_workers=4,
            val_percent=0.01,
            root_dir=None,
            iid=True,
    ):
        super().__init__()
        self.train_set = None
        self.test_set = None
        self.sub_id = sub_id
        self.number_sub = number_sub
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_percent = val_percent
        self.root_dir = root_dir
        self.iid = iid

        self.train_set = SYSCALL(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=True, download=True)
        self.test_set = SYSCALL(sub_id=self.sub_id, number_sub=self.number_sub, root_dir=root_dir, train=False, download=True)

        if not self.iid:
            # if non-iid, sort the dataset
            self.train_set = self.sort_dataset(self.train_set)
            self.test_set = self.sort_dataset(self.test_set)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset