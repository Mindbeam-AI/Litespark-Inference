# MatMul-Free CPU Kernels

CPU-optimized kernels for ternary weight matrix multiplication, based on Microsoft's BitNet/T-MAC approach.

## What's this?

This branch has optimized x86_64 kernels for MatMul-free inference with ternary weights (-1, 0, +1). The main implementations are:

- T-MAC int8: Uses PSHUFB for 16 parallel lookups with int8 quantized activations
- TL2 int8: Element-wise LUT approach (groups of 3 weights)
- AVX-512 packed: Direct computation baseline

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- x86_64 CPU with AVX2 (int8 kernels) or AVX-512 (packed kernel)
- OpenMP

## Setup

```bash
pip install -r requirements.txt
```

For PyTorch CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running tests

```bash
python test_tmac_x86.py
```

This compiles the kernels and runs benchmarks on different matrix sizes.

## Files

- src/cpu_ops/kernels/x86_64/ - C++ kernel implementations
- test_tmac_x86.py - Main benchmark script
- test_int8_kernels.py - Int8 kernel tests
