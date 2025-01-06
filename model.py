import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


class CREPEModel(nn.Module):
    """CREPE model definition, based on TorchCrepe architecture"""

    def __init__(self, capacity='full'):
        super(CREPEModel, self).__init__()

        if capacity == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif capacity == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        else:
            raise ValueError(f"Invalid model capacity: {capacity}")

        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]

        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.0)

        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])
        self.conv1_dropout = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])
        self.conv2_dropout = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])
        self.conv3_dropout = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])
        self.conv4_dropout = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(in_channels[4], out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])
        self.conv5_dropout = nn.Dropout(0.25)

        self.conv6 = nn.Conv2d(in_channels[5], out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])
        self.conv6_dropout = nn.Dropout(0.25)

        self.classifier = nn.Linear(self.in_features, 360)  # Output 360 pitch bins

    def forward(self, x, embed=False):
        x = x[:, None, :, None]

        x = self.layer(x, self.conv1, self.conv1_BN, self.conv1_dropout, (0, 0, 254, 254))
        x = self.layer(x, self.conv2, self.conv2_BN, self.conv2_dropout)
        x = self.layer(x, self.conv3, self.conv3_BN, self.conv3_dropout)
        x = self.layer(x, self.conv4, self.conv4_BN, self.conv4_dropout)
        x = self.layer(x, self.conv5, self.conv5_BN, self.conv5_dropout)

        if embed:
            return x

        x = self.layer(x, self.conv6, self.conv6_BN, self.conv6_dropout)

        x = x.permute(0, 2, 1, 3).reshape(-1, self.in_features)

        return self.classifier(x)

    def layer(self, x, conv, batch_norm, dropout, padding=(0, 0, 31, 32)):
        """Pass through one layer with custom padding, convolution, ReLU, batch norm, dropout, and max pooling"""
        x = F.pad(x, padding)
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        x = F.max_pool2d(x, (2, 1), (2, 1))
        return dropout(x)