"""
BitNet CPU - Efficient CPU inference for BitNet 1.58-bit models

Supports:
- Apple Silicon (M1/M2/M3/M4) with NEON SDOT
- Intel/AMD x86_64 with AVX-512 VNNI
- Intel Core Ultra with AVX-VNNI (256-bit)

Usage:
    from bitnet_cpu import load_model

    model, tokenizer = load_model("bitnet-2b")
    output = model.generate(tokenizer.encode("Hello", return_tensors='pt'), max_new_tokens=50)
    print(tokenizer.decode(output[0]))
"""

__version__ = "0.1.0"

from .models import (
    load_ternary_model as load_model,
    get_arch_info,
    get_kernel_type,
)

__all__ = [
    "__version__",
    "load_model",
    "get_arch_info",
    "get_kernel_type",
]
