# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Chao Feng.
#
import os
import sys

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing
from datasets import load_dataset

torch.multiprocessing.set_sharing_strategy("file_system")
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from torchtext import vocab
import pandas as pd
from torch.nn.functional import pad
from string import punctuation
import random


#######################################
#  FederatedDataModule for Sentiment140  #
#######################################

class SENTIMENT140(MNIST):
    def __init__(self, train=True):
        self.root = f"{sys.path[0]}/data"
        self.download = True
        self.train = train
        super(MNIST, self).__init__(self.root)
        self.training_file = f'{self.root}/sentiment140/processed/sentiment140_train.pt'
        self.test_file = f'{self.root}/sentiment140/processed/sentiment140_test.pt'

        if not os.path.exists(f'{self.root}/sentiment140/processed/sentiment140_test.pt') or not os.path.exists(f'{self.root}/sentiment140/processed/sentiment140_train.pt'):
            if self.download:
                self.dataset_download()
            else:
                raise RuntimeError('Dataset not found, set parameter download=True to download')

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
        saved_path = f'{self.root}/sentiment140/processed/'
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        # download data
        dataset = load_dataset("sentiment140")
        indices = range(0, len(dataset['train']))
        num_samp = 80000
        random_index = random.sample(indices, num_samp)
        dataset1 = dataset['train'][random_index]
        # from datasets to DataFrame
        data_df = pd.DataFrame(dataset1)
        data_df['sentiment'] = data_df['sentiment'].replace(to_replace=4, value=1)

        # from text to tensor
        vec = vocab.FastText()
        # tokenizing
        tokenlized_text_data = data_df['text'].apply(str.lower).apply(str.split)
        # remove punctuation
        table = str.maketrans('', '', punctuation)
        tokenlized_text_data = tokenlized_text_data.apply(lambda x: [w.translate(table) for w in x])
        tokenlized_text_data = tokenlized_text_data.apply(vec.get_vecs_by_tokens).tolist()

        # padding to 64*300
        # maxLen = max([i.shape[0] for i in tokenlized_text_data])
        tokenlized_text_data = [pad(i, [0, 0, 0, 64 - i.shape[0]], "constant", 0) for i in tokenlized_text_data]
        tokenlized_text_data = torch.stack(tokenlized_text_data)
        text_label = torch.Tensor(data_df['sentiment'].tolist())

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(tokenlized_text_data, text_label, test_size=0.15, random_state=False)
        train = [X_train, y_train]
        test = [X_test, y_test]

        # save to files
        train_file = f'{self.root}/sentiment140/processed/sentiment140_train.pt'
        test_file = f'{self.root}/sentiment140/processed/sentiment140_test.pt'

        # save to processed dir            
        if not os.path.exists(train_file):
            torch.save(train, train_file)
        if not os.path.exists(test_file):
            torch.save(test, test_file)


class Sent140DATASET():
    """
    Down the Sentiment140 datasets from torchversion.

    Args:
    iid: iid or non-iid data seperate
    """

    def __init__(self, iid=True):
        self.train_set = None
        self.test_set = None
        self.iid = iid

        data_path = f"{sys.path[0]}/data/sentiment140/"

        self.train_set = SENTIMENT140(train=True)
        self.test_set = SENTIMENT140(train=False)

        if not self.iid:
            # if non-iid, sort the dataset
            self.train_set = self.sort_dataset(self.train_set)
            self.test_set = self.sort_dataset(self.test_set)

    def sort_dataset(self, dataset):
        sorted_indexes = dataset.targets.sort()[1]
        dataset.targets = (dataset.targets[sorted_indexes])
        dataset.data = dataset.data[sorted_indexes]
        return dataset
