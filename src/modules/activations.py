# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from typing import Optional


def swiglu(gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU activation function: gate * silu(x)

    Args:
        gate: Gate tensor
        x: Input tensor

    Returns:
        SwiGLU activated tensor
    """
    return gate * F.silu(x)


def swiglu_linear(gate: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    SwiGLU activation followed by linear transformation

    Args:
        gate: Gate tensor
        x: Input tensor
        weight: Linear layer weight
        bias: Linear layer bias (optional)

    Returns:
        Linear transformation of SwiGLU activated tensor
    """
    activated = swiglu(gate, x)
    return F.linear(activated, weight, bias)