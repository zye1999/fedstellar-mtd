#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from torch import nn
from torchmetrics import MetricCollection

import lightning as pl
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, MulticlassF1Score, MulticlassConfusionMatrix
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 32

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CIFAR10ModelResNet(pl.LightningModule):
    """
    LightningModule for CIFAR10.
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
                import seaborn as sns
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 7))
                ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title("Confusion Matrix")
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.xaxis.set_ticklabels([i for i in range(10)])
                ax.yaxis.set_ticklabels([i for i in range(10)])
                self.logger.experiment.add_figure(f"{phase}Epoch/CM", ax.get_figure(), global_step=self.epoch_global_number[phase])
                plt.close()

        self.epoch_global_number[phase] += 1

    def __init__(
            self,
            in_channels=3,
            out_channels=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            implementation="scratch",
            classifier="resnet9",
    ):
        super().__init__()
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

        self.implementation = implementation
        self.classifier = classifier

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = self._build_model(in_channels, out_channels)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def _build_model(self, in_channels, out_channels):
        """
        Build the model
        Args:
            in_channels:
            out_channels:

        Returns:

        """

        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                '''
                ResNet9 implementation
                '''

                def conv_block(in_channels, out_channels, pool=False):
                    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True)]
                    if pool: layers.append(nn.MaxPool2d(2))
                    return nn.Sequential(*layers)

                conv1 = conv_block(in_channels, 64)
                conv2 = conv_block(64, 128, pool=True)
                res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

                conv3 = conv_block(128, 256, pool=True)
                conv4 = conv_block(256, 512, pool=True)
                res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

                self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                                nn.Flatten(),
                                                nn.Linear(512, out_channels))

                return dict(conv1=conv1, conv2=conv2, res1=res1, conv3=conv3, conv4=conv4, res2=res2, classifier=self.classifier)

            elif self.implementation in classifiers.keys():
                model = classifiers[self.classifier]
                # ResNet models in torchvision are trained on ImageNet, which has 1000 classes, and that is why they have 1000 output neurons.
                # To adapt a pre-trained ResNet model to classify images in the CIFAR-10 dataset, you need to replace the last layer (FC layer) with a new layer that has 10 output neurons.
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                return model
            else:
                raise NotImplementedError()

        elif self.implementation == "timm":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def forward(self, x):
        """ """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(x)}")

        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                out = self.model["conv1"](x)
                out = self.model["conv2"](out)
                out = self.model["res1"](out) + out
                out = self.model["conv3"](out)
                out = self.model["conv4"](out)
                out = self.model["res2"](out) + out
                out = self.model["classifier"](out)
                return out
            else:
                return self.model(x)
        elif self.implementation == "timm":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer

    def step(self, batch, phase):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        y_pred = self.forward(images)
        loss = self.criterion(y_pred, labels)

        # Get metrics for each batch and log them
        self.process_metrics(phase, y_pred, labels, loss)

        return loss

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
        self.log_metrics_by_epoch("Validation", print_cm=True, plot_cm=True)

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
