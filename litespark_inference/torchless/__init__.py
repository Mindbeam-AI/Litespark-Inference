"""
Torchless runtime for Litespark-Inference.

A parallel stack that does not import torch or transformers. Weights are read
directly from safetensors into numpy buffers, quantized to ternary, and packed
4-per-byte (same format as kernels/x86_64/matmul_lut_standalone.cpp). Kernel
calls and tokenizer/generation are TODO.

Motivation: the torch-hosted runtime carries a ~200 MB baseline (Python + torch
+ transformers) and is subject to PyTorch caching-allocator stickiness, which
prevents reproducing the README's sub-1GB memory numbers as live RSS. A torchless
runtime measured ~600 MB live RSS for a fully-loaded BitNet-2B in a numpy-only
POC.
"""

from .loader import load_bitnet_2b
from .model import PackedBitNetModel
from .tokenizer import BitNetTokenizer, load_tokenizer

__all__ = [
    "load_bitnet_2b",
    "PackedBitNetModel",
    "BitNetTokenizer",
    "load_tokenizer",
]
