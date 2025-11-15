# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size)) if bias else None
        self.eps = eps

    def forward(self, hidden_states, residual=None, prenorm=False):
        """
        Args:
            hidden_states: input tensor
            residual: residual tensor (optional)
            prenorm: whether to return residual connection (optional)
        Returns:
            If prenorm=True and residual is provided: (normalized_states, residual)
            Otherwise: normalized_states
        """
        if residual is not None and prenorm:
            # Pre-norm with residual: norm(x) and return both norm(x) and residual
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = self.weight * hidden_states
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
            return hidden_states.to(input_dtype), residual
        else:
            # Standard normalization
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = self.weight * hidden_states
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
            return hidden_states.to(input_dtype)


class FusedRMSNormSwishGate(nn.Module):
    """Fused RMS Norm with Swish Gate activation"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps)

    def forward(self, gate, x):
        # Apply RMS norm to x
        x_norm = self.norm(x)
        # Apply swish gate: gate * silu(x_norm)
        return gate * F.silu(x_norm)


class RMSNormLinear(nn.Module):
    """RMS Norm followed by Linear layer"""

    def __init__(self, in_features: int, out_features: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.norm = RMSNorm(in_features, eps)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(self.norm(x))