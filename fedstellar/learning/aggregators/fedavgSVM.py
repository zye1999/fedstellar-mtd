import numpy as np
import logging
from fedstellar.learning.aggregators.aggregator import Aggregator

from sklearn.svm import LinearSVC


class FedAvgSVM(Aggregator):
    """
    Federated Averaging (FedAvg) for Scikit-learn Linear SVMs.
    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        logging.info("[FedAvgSVM] My config is {}".format(self.config))

    def aggregate(self, models):
        """
        Ponderated average of the SVMs' coefficients and intercepts.

        Args:
            models: Dictionary with the SVMs (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error("[FedAvgSVM] Trying to aggregate models when there is no models")
            return None

        models = list(models.values())
        logging.info(f"Models: {models}")

        # Total Samples
        total_samples = sum([y for _, y in models])

        # Initialize accumulators
        coeff_accum = np.zeros_like(models[-1][0].coef_)
        intercept_accum = 0.0

        # Add weighted models' parameters
        logging.info("[FedAvgSVM.aggregate] Aggregating models: num={}".format(len(models)))
        for model, w in models:
            if not isinstance(model, LinearSVC):
                logging.error("[FedAvgSVM] Model is not a LinearSVC. Aggregation skipped.")
                return None
            coeff_accum += model.coef_ * w
            intercept_accum += model.intercept_ * w

        # Normalize Accum
        coeff_accum /= total_samples
        intercept_accum /= total_samples

        # Create a new SVM with the averaged parameters
        aggregated_svm = LinearSVC()
        aggregated_svm.coef_ = coeff_accum
        aggregated_svm.intercept_ = intercept_accum

        return aggregated_svm
