import logging

import torch
import numpy
from fedstellar.learning.aggregators.aggregator import Aggregator


class Krum(Aggregator):
    """
    Krum [Peva Blanchard et al., 2017]
    Paper: https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[Krum] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Krum selects one of the m local models that is similar to other models
        as the global model, the euclidean distance between two local models is used.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[Krum] Trying to aggregate models when there is no models"
            )
            return None

        models = list(models.values())

        # Total Samples
        total_samples = sum([y for _, y in models])

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighteds models
        logging.info("[Krum.aggregate] Aggregating models: num={}".format(len(models)))

        # Create model distance list
        total_models = len(models)
        distance_list = [0 for i in range(0, total_models)]

        # Calculate the L2 Norm between xi and xj
        min_index = 0
        min_distance_sum = float('inf')

        for i in range(0, total_models):
            m1, _ = models[i]
            for j in range(0, total_models):
                m2, _ = models[j]
                distance = 0
                if i == j:
                    distance = 0
                else:
                    for layer in m1:
                        l1 = m1[layer]
                        # l1 = l1.view(len(l1), 1)

                        l2 = m2[layer]
                        # l2 = l2.view(len(l2), 1)
                        distance += numpy.linalg.norm(l1 - l2)
                distance_list[i] += distance

            if min_distance_sum > distance_list[i]:
                min_distance_sum = distance_list[i]
                min_index = i

        # Assign the model with min distance with others as the aggregated model
        m, _ = models[min_index]
        for layer in m:
            accum[layer] = accum[layer] + m[layer]

        return accum