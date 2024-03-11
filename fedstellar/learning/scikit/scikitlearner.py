# 
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
# 

import logging
import pickle
from collections import OrderedDict
import traceback

import numpy as np
from sklearn.metrics import accuracy_score
import copy

from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner


###########################
#       ScikitLearner     #
###########################


class ScikitLearner(NodeLearner):
    """
    Learner using scikit-learn.

    Attributes:
        model: scikit-learn model to train.
        data: Data to train the model.
    """

    def __init__(self, model, data, config=None, logger=None):
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.round = 0
        self.epochs = 1
        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)

    def set_model(self, model):
        self.model = model

    def get_round(self):
        return self.round

    def set_data(self, data):
        self.data = data

    def encode_parameters(self, params=None, contributors=None, weight=None):
        if params is None:
            params = self.model.get_params()
        return pickle.dumps(params)

    def decode_parameters(self, data):
        try:
            params = pickle.loads(data)
            return params
        except:
            raise DecodingParamsError("Error decoding parameters")

    def check_parameters(self, params):
        # This is a basic check. Depending on the model, you might need more specific checks.
        if not params:
            return False
        return True

    def set_parameters(self, params):
        self.model.set_params(**params)

    def get_parameters(self):
        return self.model.get_params()

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        try:
            X_train, y_train = self.data.train_dataloader()
            # X_train = X_train.view(X_train.size(0), -1).numpy()
            # y_train = y_train.numpy()
            self.model.fit(X_train, y_train)
        except Exception as e:
            logging.error("Error with scikit-learn fit. {}".format(e))
            logging.error(traceback.format_exc())

    def interrupt_fit(self):
        pass

    def evaluate(self):
        try:
            X_test, y_test = self.data.test_dataloader()
            # X_test = X_test.view(X_test.size(0), -1).numpy()
            # y_test = y_test.numpy()
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Accuracy: {accuracy}")
        except Exception as e:
            logging.error("Error with scikit-learn evaluate. {}".format(e))
            logging.error(traceback.format_exc())
            return None

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        if self.logger:
            self.logger.log_metrics({"Test/Accuracy": metric})

    def get_num_samples(self):
        return (
            len(self.data.train_dataloader()),
            len(self.data.test_dataloader()),
        )

    def finalize_round(self):
        self.round += 1
        if self.logger:
            self.logger.log_metrics({"Round": self.round})
