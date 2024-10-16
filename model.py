# model.py

import torch
import torch.nn as nn

class CREPEModel(nn.Module):
    def __init__(self, capacity='full'):
        super(CREPEModel, self).__init__()
        capacity_multiplier = {'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32}[capacity]

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32 * capacity_multiplier, kernel_size=(512, 1), stride=(4, 1), padding=(512 // 2, 0))
        self.conv2 = nn.Conv2d(32 * capacity_multiplier, 4 * capacity_multiplier, kernel_size=(64, 1), stride=(1, 1), padding=(64 // 2, 0))
        self.conv3 = nn.Conv2d(4 * capacity_multiplier, 4 * capacity_multiplier, kernel_size=(64, 1), stride=(1, 1), padding=(64 // 2, 0))
        self.conv4 = nn.Conv2d(4 * capacity_multiplier, 4 * capacity_multiplier, kernel_size=(64, 1), stride=(1, 1), padding=(64 // 2, 0))
        self.conv5 = nn.Conv2d(4 * capacity_multiplier, 8 * capacity_multiplier, kernel_size=(64, 1), stride=(1, 1), padding=(64 // 2, 0))
        self.conv6 = nn.Conv2d(8 * capacity_multiplier, 16 * capacity_multiplier, kernel_size=(64, 1), stride=(1, 1), padding=(64 // 2, 0))

        # MaxPooling and Dropout
        self.maxpool = nn.MaxPool2d((2, 1))
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(2048, 360)  # Adjust the input size accordingly
        self.sigmoid = nn.Sigmoid()

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(32 * capacity_multiplier)
        self.batchnorm2 = nn.BatchNorm2d(4 * capacity_multiplier)
        self.batchnorm3 = nn.BatchNorm2d(4 * capacity_multiplier)
        self.batchnorm4 = nn.BatchNorm2d(4 * capacity_multiplier)
        self.batchnorm5 = nn.BatchNorm2d(8 * capacity_multiplier)
        self.batchnorm6 = nn.BatchNorm2d(16 * capacity_multiplier)
    def forward(self, x):
        # Input x shape: (batch_size, 1024)
        x = x.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, 1024, 1)
        # No need to permute x

        # Apply convolutions, batch normalization, activations, pooling, and dropout
        x = torch.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.relu(self.batchnorm2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.relu(self.batchnorm3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.relu(self.batchnorm4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.relu(self.batchnorm5(self.conv5(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.relu(self.batchnorm6(self.conv6(x)))
        x = self.maxpool(x)
        x = self.dropout(x)

        # Flatten and pass through fully connected layer with sigmoid
        x = self.flatten(x)
        x = self.sigmoid(self.fc(x))
        return x  # Output shape: (batch_size, 360)
