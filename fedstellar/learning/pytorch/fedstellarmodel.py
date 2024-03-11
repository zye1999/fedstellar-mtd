#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

from abc import ABC, abstractmethod
import torch
import lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics import MetricCollection
import seaborn as sns
import matplotlib.pyplot as plt


class FedstellarModel(pl.LightningModule, ABC):
    """
    Abstract class for the Fedstellar model.
    
    This class is an abstract class that defines the interface for the Fedstellar model.
    """
    
    def process_metrics(self, phase, y_pred, y, loss=None):
        """
        Calculate and log metrics for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            y_pred (torch.Tensor): Model predictions
            y (torch.Tensor): Ground truth labels
            loss (torch.Tensor, optional): Loss value
        """
        if loss is not None:
            self.log(f"{phase}/Loss", loss, prog_bar=True, logger=True)

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            output = self.train_metrics(y_pred_classes, y)
        elif phase == "Validation":
            output = self.val_metrics(y_pred_classes, y)
        elif phase == "Test":
            output = self.test_metrics(y_pred_classes, y)
        else:
            raise NotImplementedError
        # print(f"y_pred shape: {y_pred.shape}, y_pred_classes shape: {y_pred_classes.shape}, y shape: {y.shape}")  # Debug print
        output = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}
        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            self.cm.update(y_pred_classes, y)

    def log_metrics_by_epoch(self, phase, print_cm=False, plot_cm=False):
        """
        Log all metrics at the end of an epoch for the given phase.
        Args:
            phase (str): One of 'Train', 'Validation', or 'Test'
            :param phase:
            :param plot_cm:
        """
        print(f"Epoch end: {phase}, epoch number: {self.epoch_global_number[phase]}")
        if phase == "Train":
            output = self.train_metrics.compute()
            self.train_metrics.reset()
        elif phase == "Validation":
            output = self.val_metrics.compute()
            self.val_metrics.reset()
        elif phase == "Test":
            output = self.test_metrics.compute()
            self.test_metrics.reset()
        else:
            raise NotImplementedError

        output = {f"{phase}Epoch/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            cm = self.cm.compute().cpu()
            print(f"{phase}Epoch/CM\n", cm) if print_cm else None
            if plot_cm:
                plt.figure(figsize=(10, 7))
                ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title("Confusion Matrix")
                ax.set_xticks(range(self.out_channels))
                ax.set_yticks(range(self.out_channels))
                ax.xaxis.set_ticklabels([i for i in range(self.out_channels)])
                ax.yaxis.set_ticklabels([i for i in range(self.out_channels)])
                self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                plt.close()

        self.epoch_global_number[phase] += 1

    def __init__(
        self,
        in_channels=1,
        out_channels=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        
        if metrics is None:
            metrics = MetricCollection([
                MulticlassAccuracy(num_classes=out_channels),
                MulticlassPrecision(num_classes=out_channels),
                MulticlassRecall(num_classes=out_channels),
                MulticlassF1Score(num_classes=out_channels)
            ])

        # Define metrics
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=out_channels)
        
        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}
    
    @abstractmethod
    def forward(self, x):
        """Forward pass of the model."""
        pass
    
    @abstractmethod
    def configure_optimizers(self):
        """Optimizer configuration."""
        pass
    
    @abstractmethod
    def step(self, batch, batch_idx, phase):
        """Training/validation/test step."""
        pass
    
    def training_step(self, batch, batch_id):
        """
        Training step for the model.
        Args:
            batch:
            batch_id:

        Returns:
        """
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_metrics_by_epoch("Train", print_cm=True, plot_cm=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_metrics_by_epoch("Validation", print_cm=True, plot_cm=False)

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        Args:
            batch:
            batch_idx:

        Returns:
        """
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_metrics_by_epoch("Test", print_cm=True, plot_cm=True)