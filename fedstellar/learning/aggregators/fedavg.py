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

import torch

from fedstellar.learning.aggregators.aggregator import Aggregator


class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FedAvg] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Weighted average of the models.

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models) == 0:
            logging.error("[FedAvg] Trying to aggregate models when there are no models")
            return None

        models = list(models.values())

        # Total Samples
        total_samples = sum(w for _, w in models)

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1][0].items()}

        # Add weighted models
        logging.info(f"[FedAvg.aggregate] Aggregating models: num={len(models)}")
        for model, weight in models:
            for layer in accum:
                accum[layer] += model[layer] * weight

        # Normalize Accum
        for layer in accum:
            accum[layer] /= total_samples
            
        # self.print_model_size(accum)

        return accum
