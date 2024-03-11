import logging

import torch
import numpy as np
from fedstellar.learning.aggregators.aggregator import Aggregator


class TrimmedMean(Aggregator):
    """
    TrimmedMean [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self, node_name="unknown", config=None, beta=0):
        super().__init__(node_name, config)
        self.beta = beta
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[TrimmedMean] My config is {}".format(self.config))

    def get_trimmedmean(self, weights):
        """
        For weight list [w1j,w2j,··· ,wmj], removes the largest and
        smallest β of them, and computes the mean of the remaining
        m-2β parameters

        Args:
            weights: weights list, 2D tensor
        """

        # check if the weight tensor has enough space
        weight_len = len(weights)
        if weight_len == 0:
            logging.error(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None

        if weight_len <= 2 * self.beta:
            # logging.error(
            #     "[TrimmedMean] Number of model should larger than 2 * beta"
            # )
            remaining_wrights = weights
            res = torch.mean(remaining_wrights, 0)

        else:
            # remove the largest and smallest β items
            arr_weights = np.asarray(weights)
            nobs = arr_weights.shape[0]
            start = self.beta
            end = nobs - self.beta
            atmp = np.partition(arr_weights, (start, end - 1), 0)
            sl = [slice(None)] * atmp.ndim
            sl[0] = slice(start, end)
            print(atmp[tuple(sl)])
            arr_median = np.mean(atmp[tuple(sl)], axis=0)
            res = torch.tensor(arr_median)

        # get the mean of the remaining weights

        return res

    def aggregate(self, models):
        """
        For each jth model parameter, the master device sorts the jth parameters
        of the m local models, i.e., w1j,w2j,··· ,wmj, where wij is the
        jth parameter of the ith local model, removes the largest and
        smallest β of them, and computes the mean of the remaining
        m-2β parameters as the jth parameter of the global model.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None

        models = list(models.values())
        models_params = [m for m, _ in models]

        # Total Samples
        total_samples = sum([y for _, y in models])
        total_models = len(models)

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighted models
        logging.info("[TrimmedMean.aggregate] Aggregating models: num={}".format(len(models)))

        # Calculate the trimmedmean for each parameter
        for layer in accum:
            weight_layer = accum[layer]
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([models_params[j][layer] for j in range(0, total_models)])
                weights = weights.double()
                w = self.get_trimmedmean(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model
                models_layer_weight_flatten = torch.stack([models_params[j][layer].view(number_layer_weights) for j in range(0, total_models)], 0)

                # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
                trimmedmean = self.get_trimmedmean(models_layer_weight_flatten)
                accum[layer] = trimmedmean.view(l_shape)

        return accum