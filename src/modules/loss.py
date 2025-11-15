# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedCrossEntropyLoss(nn.Module):
    """Fused Cross Entropy Loss for language modeling"""

    def __init__(self, ignore_index: int = -100, reduction: str = 'mean', inplace_backward: bool = False):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.inplace_backward = inplace_backward

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size) or (batch_size * seq_len, vocab_size)
            labels: (batch_size, seq_len) or (batch_size * seq_len,)

        Returns:
            Cross entropy loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)

        return F.cross_entropy(
            logits,
            labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction
        )