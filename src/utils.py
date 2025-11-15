
from __future__ import annotations

from functools import wraps

import torch


def contiguous(fn):
    """
    A decorator that ensures all tensor arguments are contiguous.

    Args:
        fn: The function to be decorated.

    Returns:
        The decorated function.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args),
                  **{k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()})
    return wrapper
