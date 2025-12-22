"""
BitNet CPU - Efficient CPU inference for BitNet 1.58-bit models

Supports:
- Apple Silicon (M1/M2/M3/M4) with NEON SDOT
- Intel/AMD x86_64 with AVX-512 VNNI
- Intel Core Ultra with AVX-VNNI (256-bit)

Usage:
    from bitnet_cpu import load_model, generate

    model, tokenizer = load_model("bitnet-2b")
    output = generate(model, tokenizer, "The capital of France is")
    print(output)
"""

__version__ = "0.1.0"

from .models import (
    load_model,
    generate,
    get_model_info,
    list_available_models,
)

from .arch import (
    get_arch_info,
    get_kernel_type,
)

__all__ = [
    "__version__",
    "load_model",
    "generate",
    "get_model_info",
    "list_available_models",
    "get_arch_info",
    "get_kernel_type",
]
