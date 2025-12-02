# MatMul-Free CPU Kernels

CPU-optimized kernels for ternary weight matrix multiplication, implementing the MatMul-free approach with Microsoft's T-MAC/BitNet optimizations.

## Background

### MatMul-Free LLMs
Traditional neural networks rely heavily on matrix multiplication (MatMul), which dominates compute and memory bandwidth. The [MatMul-free approach](https://arxiv.org/abs/2406.02528) replaces dense MatMul with ternary weights (-1, 0, +1), enabling:
- **8x memory reduction**: Ternary weights need only 2 bits vs 16/32 bits for float
- **Simpler computation**: Multiply becomes conditional add/subtract/skip

### Microsoft's T-MAC Approach
[Microsoft's T-MAC](https://arxiv.org/abs/2407.00088) achieves high performance on CPUs by:
1. **Quantizing activations to int8** with per-row scaling
2. **Packing ternary weights** into bit-planes (sign + value)
3. **Using PSHUFB** for parallel table lookups (32 lookups/instruction on AVX2)
4. **VNNI instructions** for direct int8 dot products (64 multiply-accumulates/instruction)

## Kernel Implementations

### T-MAC Int8 PSHUFB (`matmul_free_tmac_int8.cpp`)
Microsoft's LUT-based approach using AVX2:
- Groups 4 weights → 16-entry lookup table
- PSHUFB performs 32 parallel lookups per instruction
- Int8 activations with per-row quantization
- ~190-220 GFLOPS

### T-MAC Int8 AVX-512 v2 (`matmul_free_tmac_int8_avx512_v2.cpp`)
Extended to 512-bit vectors:
- 64 parallel lookups per instruction
- Optimized lane handling for cross-lane operations
- ~250-285 GFLOPS

### VNNI Kernels (`matmul_free_tmac_vnni.cpp`)
Direct dot-product computation using AVX-512 VNNI:

**VNNI Simple**: Basic implementation
- Uses `_mm512_dpbusd_epi32` for 64 int8 multiply-accumulates per instruction
- Weights stored as int8 (-1, 0, +1)
- ~340-590 GFLOPS

**VNNI v2 (Register Blocking)**: Optimized for small/medium matrices
- Processes 4 outputs simultaneously (register blocking)
- Vectorized int8→uint8 activation conversion
- Per-thread activation buffers
- **~720-790 GFLOPS** (best for small/medium)

**VNNI v3 (Tiled)**: Cache-optimized for large matrices
- M-tiling (32 rows) + N-tiling (64 outputs)
- Precomputes uint8 activations per M-tile
- Weight tiles fit in L2 cache (~128KB per tile)
- **~550-600 GFLOPS** (best for large matrices)

### Other Kernels
- **T-MAC float32** (`matmul_free_tmac_avx2.cpp`): Reference implementation with float LUTs (~50 GFLOPS)
- **TL2** (`matmul_free_tl2_*.cpp`): Alternative 3-weight grouping scheme
- **AVX-512 packed** (`matmul_free_avx512.cpp`): Direct computation baseline (~145 GFLOPS)

## Benchmark Results

Tested on Intel Xeon Platinum 8375C (Ice Lake) with 8 threads:

| Kernel | Small (128×1024×1024) | Medium (256×1024×1024) | Large (512×1024×1024) | XLarge (1024×2048×2048) |
|--------|----------------------|------------------------|----------------------|-------------------------|
| T-MAC int8 PSHUFB | 194 GFLOPS | 187 GFLOPS | 186 GFLOPS | 223 GFLOPS |
| T-MAC int8 AVX-512 v2 | 252 GFLOPS | 250 GFLOPS | 216 GFLOPS | 284 GFLOPS |
| VNNI simple | 587 GFLOPS | 557 GFLOPS | 515 GFLOPS | 338 GFLOPS |
| **VNNI v2 (reg block)** | **766 GFLOPS** | **789 GFLOPS** | **723 GFLOPS** | 348 GFLOPS |
| **VNNI v3 (tiled)** | 589 GFLOPS | 597 GFLOPS | 597 GFLOPS | **549 GFLOPS** |
| PyTorch (MKL) | 250 GFLOPS | 351 GFLOPS | 562 GFLOPS | 575 GFLOPS |

### Speedup vs PyTorch MKL

| Kernel | Small | Medium | Large | XLarge |
|--------|-------|--------|-------|--------|
| VNNI v2 | **3.06x** | **2.25x** | **1.29x** | 0.61x |
| VNNI v3 | 2.36x | 1.70x | 1.06x | **0.95x** |

**Key insight**: VNNI v2 is best for small/medium matrices (up to 3x faster than MKL), while VNNI v3's cache-aware tiling helps on larger matrices.

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- x86_64 CPU with:
  - AVX2 (minimum for int8 kernels)
  - AVX-512 + VNNI (for best performance)
- OpenMP
- matplotlib (for benchmark plots)

## Setup

```bash
# Create virtual environment
python -m venv cpu-matmul-env
source cpu-matmul-env/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy psutil matplotlib
```

## Running Benchmarks

```bash
python test_tmac_x86.py
```

This will:
1. Compile all kernels using PyTorch's JIT
2. Run benchmarks on 4 matrix sizes (Small, Medium, Large, XLarge)
3. Verify correctness against PyTorch
4. Print summary tables
5. Generate plots in `benchmark_plots/`

### Output Files
- `benchmark_plots/gflops_comparison.png` - GFLOPS by matrix size
- `benchmark_plots/speedup_vs_pytorch.png` - Speedup vs MKL baseline
- `benchmark_plots/time_comparison.png` - Execution time comparison
- `benchmark_plots/gflops_scaling.png` - Performance scaling across sizes

## Project Structure

```
├── src/cpu_ops/kernels/x86_64/
│   ├── matmul_free_tmac_avx2.cpp      # T-MAC float32 (reference)
│   ├── matmul_free_tmac_int8.cpp      # T-MAC int8 PSHUFB (AVX2)
│   ├── matmul_free_tmac_int8_avx512.cpp    # T-MAC int8 (AVX-512)
│   ├── matmul_free_tmac_int8_avx512_v2.cpp # T-MAC int8 optimized
│   ├── matmul_free_tmac_vnni.cpp      # VNNI kernels (v2, v3)
│   ├── matmul_free_tl2_avx2.cpp       # TL2 float32
│   ├── matmul_free_tl2_int8.cpp       # TL2 int8
│   └── matmul_free_avx512.cpp         # AVX-512 packed baseline
├── test_tmac_x86.py                   # Main benchmark script
├── test_int8_kernels.py               # Int8 kernel unit tests
└── benchmark_plots/                   # Generated plots
```

## References

- [MatMul-free Language Models](https://arxiv.org/abs/2406.02528)
- [T-MAC: CPU Renaissance via Table Lookup](https://arxiv.org/abs/2407.00088)
- [Microsoft BitNet](https://github.com/microsoft/BitNet)
- [Microsoft T-MAC](https://github.com/microsoft/T-MAC)
