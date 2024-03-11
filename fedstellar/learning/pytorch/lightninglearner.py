# 
# This file an adaptation and extension of the p2pfl library (https://pypi.org/project/p2pfl/).
# Refer to the LICENSE file for licensing information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import logging
import os
import pickle
from collections import OrderedDict
import random
import traceback
import hashlib
import numpy as np
import io
import gzip

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
import copy

from fedstellar.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from fedstellar.learning.learner import NodeLearner
from torch.nn import functional as F

###########################
#    LightningLearner     #
###########################


class LightningLearner(NodeLearner):
    """
    Learner with PyTorch Lightning.

    Atributes:
        model: Model to train.
        data: Data to train the model.
        epochs: Number of epochs to train.
        logger: Logger.
    """

    def __init__(self, model, data, config=None, logger=None):
        # logging.info("[Learner] Compiling model... (BETA)")
        # self.model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.data = data
        self.config = config
        self.logger = logger
        self.__trainer = None
        self.epochs = 1
        logging.getLogger("lightning.pytorch").setLevel(logging.INFO)

        # FL information
        self.round = 0
        
        self.fix_randomness()
        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)

    def fix_randomness(self):
        seed = self.config.participant["scenario_args"]["random_seed"]
        logging.info("[Learner] Fixing randomness with seed {}".format(seed))
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def get_round(self):
        return self.round

    def set_model(self, model):
        self.model = model

    def set_data(self, data):
        self.data = data

    ####
    # Model weights
    # Encode/decode parameters: https://pytorch.org/docs/stable/notes/serialization.html
    # There are other ways to encode/decode parameters: protobuf, msgpack, etc.
    ####
    def encode_parameters(self, params=None):
        if params is None:
            params = self.model.state_dict()
        buffer = io.BytesIO()
        #with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        #    torch.save(params, f)
        torch.save(params, buffer)
        return buffer.getvalue()

    def decode_parameters(self, data):
        try:
            buffer = io.BytesIO(data)
            #with gzip.GzipFile(fileobj=buffer, mode='rb') as f:
            #    params_dict = torch.load(f, map_location='cpu')
            params_dict = torch.load(buffer, map_location='cpu')
            return OrderedDict(params_dict)
        except Exception as e:
            raise DecodingParamsError("Error decoding parameters: {}".format(e))

    def check_parameters(self, params):
        # Check ordered dict keys
        if set(params.keys()) != set(self.model.state_dict().keys()):
            return False
        # Check tensor shapes
        for key, value in params.items():
            if value.shape != self.model.state_dict()[key].shape:
                return False
        return True

    def set_parameters(self, params):
        try:
            self.model.load_state_dict(params)
        except:
            raise ModelNotMatchingError("Not matching models")

    def get_parameters(self):
        return self.model.state_dict()
    
    def get_hash_model(self):
        '''
        Returns:
            str: SHA256 hash of model parameters
        '''
        return hashlib.sha256(self.encode_parameters()).hexdigest()
        

    def set_epochs(self, epochs):
        self.epochs = epochs

    def fit(self):
        try:
            if self.epochs > 0:
                self.create_trainer()
                # torch.autograd.set_detect_anomaly(True)
                self.__trainer.fit(self.model, self.data)
                self.__trainer = None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            # Log full traceback
            logging.error(traceback.format_exc())

    def interrupt_fit(self):
        if self.__trainer is not None:
            self.__trainer.should_stop = True
            self.__trainer = None

    def evaluate(self):
        try:
            if self.epochs > 0:
                # self.create_trainer()
                # self.__trainer.test(self.model, self.data, verbose=True)
                # self.__trainer = None

                # results = self.__trainer.test(self.model, self.data, verbose=True)
                # loss = results[0]["Test/Loss"]
                # metric = results[0]["Test/Accuracy"]
                # self.__trainer = None
                # self.log_validation_metrics(loss, metric, self.round)
                # return loss, metric

                # Zi Ye
                # Begin
                self.create_trainer()
                results = self.__trainer.test(self.model, self.data, verbose=True)
                self.__trainer = None
                return results
                # End

            else:
                return None
        except Exception as e:
            logging.error("Something went wrong with pytorch lightning. {}".format(e))
            # Log full traceback
            logging.error(traceback.format_exc())
            return None

    def log_validation_metrics(self, loss, metric, round=None, name=None):
        self.logger.log_metrics({"Test/Loss": loss, "Test/Accuracy": metric}, step=self.logger.global_step)
        pass

    def get_num_samples(self):
        return (
            len(self.data.train_dataloader().dataset),
            len(self.data.test_dataloader().dataset),
        )

    def finalize_round(self):
        self.logger.global_step = self.logger.global_step + self.logger.local_step
        self.logger.local_step = 0
        self.round += 1
        self.logger.log_metrics({"Round": self.round}, step=self.logger.global_step)
        pass

    def create_trainer(self):
        logging.info("[Learner] Creating trainer with accelerator: {}".format(self.config.participant["device_args"]["accelerator"]))
        progress_bar = RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
            leave=True,
        )
        self.__trainer = Trainer(
            callbacks=[RichModelSummary(max_depth=1), progress_bar],
            max_epochs=self.epochs,
            accelerator=self.config.participant["device_args"]["accelerator"],
            devices="auto" if self.config.participant["device_args"]["accelerator"] == "cpu" else "1",  # TODO: only one GPU for now
            # strategy="ddp" if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            # strategy=self.config.participant["device_args"]["strategy"] if self.config.participant["device_args"]["accelerator"] != "auto" else None,
            logger=self.logger,
            log_every_n_steps=50,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True
        )

    def validate_neighbour_model(self, neighbour_model_param):
        avg_loss = 0
        running_loss = 0
        bootstrap_dataloader = self.data.bootstrap_dataloader()
        num_samples = 0
        neighbour_model = copy.deepcopy(self.model)
        neighbour_model.load_state_dict(neighbour_model_param)

        # enable evaluation mode, prevent memory leaks.
        # no need to switch back to training since model is not further used.
        if torch.cuda.is_available():
            neighbour_model = neighbour_model.to('cuda')
        neighbour_model.eval()

        # bootstrap_dataloader = bootstrap_dataloader.to('cuda')

        with torch.no_grad():
            for inputs, labels in bootstrap_dataloader:
                if torch.cuda.is_available():
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')
                outputs = neighbour_model(inputs)
                loss = F.cross_entropy(outputs, labels)
                running_loss += loss.item()
                num_samples += inputs.size(0)

        avg_loss = running_loss / len(bootstrap_dataloader)
        logging.info("[Learner.validate_neighbour]: Computed neighbor loss over {} data samples".format(num_samples))
        return avg_loss