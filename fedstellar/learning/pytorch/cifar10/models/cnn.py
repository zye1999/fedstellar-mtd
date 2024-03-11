#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")

import torch
from fedstellar.learning.pytorch.fedstellarmodel import FedstellarModel


class CIFAR10ModelCNN(FedstellarModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
            self,
            in_channels=3,
            out_channels=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None
    ):
        super().__init__(in_channels, out_channels, learning_rate, metrics, confusion_matrix, seed)

        self.config = {
            'beta1': 0.851436,
            'beta2': 0.999689,
            'amsgrad': True
        }
        
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define layers of the model
        self.conv1 = torch.nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, out_channels)

    def forward(self, x):
        """Forward pass of the model."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.config['beta1'], self.config['beta2']), amsgrad=self.config['amsgrad'])
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