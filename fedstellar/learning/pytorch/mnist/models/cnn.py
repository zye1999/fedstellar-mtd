#
# This file is part of the Fedstellar platform (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2023 Enrique Tomás Martínez Beltrán.
#

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy("file_system")

import torch
from fedstellar.learning.pytorch.fedstellarmodel import FedstellarModel


class MNISTModelCNN(FedstellarModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
            self,
            in_channels=1,
            out_channels=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None
    ):
        super().__init__(in_channels, out_channels, learning_rate, metrics, confusion_matrix, seed)

        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define layers of the model
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=(5, 5), padding="same"
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 64, 2048)
        self.l2 = torch.nn.Linear(2048, out_channels)

    def forward(self, x):
        """Forward pass of the model."""
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)
        
        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)
        
        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)
        
        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)
        
        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)
        
        return logits

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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