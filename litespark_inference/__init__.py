"""
Litespark-Inference - Efficient CPU inference for ternary language models

Supports:
- Apple Silicon (M1/M2/M3/M4) with NEON SDOT
- Intel/AMD x86_64 with AVX-512 VNNI
- Intel Core Ultra with AVX-VNNI (256-bit)
- BitNet and Falcon Edge model families

Usage:
    from litespark_inference import load_model

    model, tokenizer = load_model("bitnet-2b")
    output = model.generate(tokenizer.encode("Hello", return_tensors='pt'), max_new_tokens=50)
    print(tokenizer.decode(output[0]))
"""

__version__ = "0.1.5"

# Lazy top-level imports. `.models` pulls torch; we don't want to drag torch
# into sibling subpackages like `.torchless` that must stay torch-free.
_TORCH_BACKED = {
    "load_model": ("litespark_inference.models", "load_ternary_model"),
    "get_arch_info": ("litespark_inference.models", "get_arch_info"),
    "get_kernel_type": ("litespark_inference.models", "get_kernel_type"),
}


def __getattr__(name):
    if name in _TORCH_BACKED:
        import importlib
        mod_name, attr = _TORCH_BACKED[name]
        value = getattr(importlib.import_module(mod_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'litespark_inference' has no attribute {name!r}")


__all__ = [
    "__version__",
    "load_model",
    "get_arch_info",
    "get_kernel_type",
]
