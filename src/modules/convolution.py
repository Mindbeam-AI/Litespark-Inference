# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConvolution(nn.Module):
    """Short convolution layer for sequence modeling"""

    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = 'silu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        # 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size  # Depthwise convolution
        )

        # Activation function
        if activation == 'silu':
            self.activation = F.silu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = lambda x: x

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # Conv1d expects: (batch_size, hidden_size, seq_len)

        batch_size, seq_len, hidden_size = x.shape

        # Transpose for conv1d
        x = x.transpose(1, 2)  # (batch_size, hidden_size, seq_len)

        # Apply convolution
        x = self.conv1d(x)

        # Remove extra padding (keep original sequence length)
        x = x[:, :, :seq_len]

        # Apply activation
        x = self.activation(x)

        # Transpose back
        x = x.transpose(1, 2)  # (batch_size, seq_len, hidden_size)

        return x