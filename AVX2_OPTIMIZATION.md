# x86_64 AVX2 Optimization Guide

## Overview

This document describes the AVX2-optimized MatMul-free kernel for x86_64 systems (Intel/AMD CPUs).

**Status:** ✅ Optimized kernel implemented, awaiting x86_64 testing

**Expected Performance:** 200-250 GFLOPS (2x faster than ARM64 NEON)

**Target Platforms:**
- Linux servers (Intel Xeon, AMD EPYC, Ryzen)
- Windows workstations
- Cloud instances (AWS EC2, Azure, GCP)

---

## Key Optimizations Applied

### 1. Output Blocking (N_BLOCK=16)

Based on successful ARM64 optimizations (N_BLOCK=12), the AVX2 kernel processes **16 outputs simultaneously**:

```cpp
const int N_BLOCK = 16;  // Process 16 outputs together

for (int n = 0; n <= N - N_BLOCK; n += N_BLOCK) {
    __m256 sum_pos[N_BLOCK], sum_neg[N_BLOCK];
    // ... process all 16 outputs with shared input
}
```

**Why 16?**
- AVX2 has 8-wide SIMD (2x NEON's 4-wide)
- Following ARM64's ratio: 12-way for 4-wide → 16-way for 8-wide
- Balances instruction-level parallelism vs register pressure

### 2. Hardware Predication with `_mm256_blendv_ps`

Replaced bitwise AND (`_mm256_and_ps`) with proper hardware predication:

```cpp
// OLD: Bitwise AND (incorrect semantic)
sum_pos = _mm256_add_ps(sum_pos, _mm256_and_ps(mask_pos, x_vals));

// NEW: Hardware predication (correct)
sum_pos = _mm256_add_ps(sum_pos, _mm256_blendv_ps(zero, x_vals, mask_pos));
```

**Benefit:** `_mm256_blendv_ps` is the x86_64 equivalent of ARM's `vbslq_f32` - selects values based on mask without bit manipulation.

### 3. Prefetching for Memory Latency Hiding

Added software prefetching to hide memory access latency:

```cpp
const int K_PREFETCH = 64;

if (k + K_PREFETCH < K) {
    _mm_prefetch((const char*)(x_row + k + K_PREFETCH), _MM_HINT_T0);
    _mm_prefetch((const char*)(w_row + k + K_PREFETCH), _MM_HINT_T0);
}
```

**Benefit:** Prefetches data 64 elements ahead while processing current data.

### 4. Dynamic Scheduling with OpenMP

```cpp
#pragma omp parallel for schedule(dynamic, 8)
for (int m = 0; m < M; m++) {
    // ... process row m
}
```

**Benefit:** Better load balancing across threads, especially for irregular workloads.

---

## Performance Expectations

### Theoretical Analysis

```
SIMD Width:       8 floats (vs NEON's 4)
Output Blocking:  16-way (vs NEON's 12-way)
Expected Ratio:   ~2x ARM64 performance
```

**Target Performance:**
- **Small matrices (128×1024×1024):** 180-220 GFLOPS
- **Medium matrices (256×1024×1024):** 200-240 GFLOPS
- **Large matrices (512×1024×1024):** 220-260 GFLOPS

**vs Baseline:**
- **PyTorch (MKL/OneDNN):** 0.15-0.20x (unavoidable, optimized libraries)
- **ARM64 NEON:** ~2x faster (wider SIMD)
- **Generic fallback:** ~20x faster (SIMD + blocking)

### Hardware Comparison

| Platform | SIMD Width | Blocking | Expected GFLOPS |
|----------|------------|----------|-----------------|
| ARM64 NEON | 4 floats | 12-way | 126 (Measured) |
| x86_64 AVX2 | 8 floats | 16-way | 200-250 (Expected) |
| x86_64 AVX-512 | 16 floats | 24-32 way | 400-500 (Future) |

---

## Implementation Details

### Core Kernel Structure

**File:** `src/cpu_ops/kernels/x86_64/matmul_free_avx2.cpp`

```cpp
void matmul_free_avx2(
    torch::Tensor x_tensor,      // [M, K]
    torch::Tensor w_tensor,      // [N, K] - ternary {-1, 0, 1}
    torch::Tensor y_tensor,      // [M, N]
    torch::Tensor bias_tensor,   // [N]
    int M, int N, int K,
    int num_threads
) {
    const int K_SIMD = 8;
    const int N_BLOCK = 16;

    #pragma omp parallel for schedule(dynamic, 8)
    for (int m = 0; m < M; m++) {
        // Process N_BLOCK outputs together
        for (int n = 0; n <= N - N_BLOCK; n += N_BLOCK) {
            __m256 sum_pos[N_BLOCK], sum_neg[N_BLOCK];

            for (int k = 0; k < K; k += 8) {
                __m256 x_vals = _mm256_loadu_ps(x_row + k);

                for (int i = 0; i < N_BLOCK; i++) {
                    __m256 w_vals = _mm256_loadu_ps(w + (n+i)*K + k);
                    __m256 mask_pos = _mm256_cmp_ps(w_vals, zero, _CMP_GT_OQ);
                    __m256 mask_neg = _mm256_cmp_ps(w_vals, zero, _CMP_LT_OQ);

                    sum_pos[i] = _mm256_add_ps(sum_pos[i],
                        _mm256_blendv_ps(zero, x_vals, mask_pos));
                    sum_neg[i] = _mm256_add_ps(sum_neg[i],
                        _mm256_blendv_ps(zero, x_vals, mask_neg));
                }
            }

            // Reduce and store results...
        }
    }
}
```

### AVX2 Intrinsics Used

| Intrinsic | Purpose | Equivalent |
|-----------|---------|------------|
| `_mm256_loadu_ps` | Load 8 floats | ARM: `vld1q_f32` |
| `_mm256_cmp_ps` | Compare floats, create mask | ARM: `vcgtq_f32`/`vcltq_f32` |
| `_mm256_blendv_ps` | Conditional select | ARM: `vbslq_f32` |
| `_mm256_add_ps` | Add 8 floats | ARM: `vaddq_f32` |
| `_mm_prefetch` | Prefetch data | ARM: `__builtin_prefetch` |

### Horizontal Sum Reduction

```cpp
inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);          // Lower 128 bits
    __m128 hi = _mm256_extractf128_ps(v, 1);        // Upper 128 bits
    __m128 sum = _mm_add_ps(lo, hi);                // Add halves
    __m128 shuf = _mm_movehdup_ps(sum);             // Shuffle
    __m128 sums = _mm_add_ps(sum, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);                      // Extract scalar
}
```

---

## Compilation and Testing

### Compilation Flags

**Linux:**
```python
extra_cflags = [
    '-O3',                  # Maximum optimization
    '-mavx2',              # Enable AVX2 instructions
    '-mfma',               # Enable FMA (fused multiply-add)
    '-DHAS_AVX2_SUPPORT',  # Define feature flag
    '-fopenmp',            # OpenMP support
    '-std=c++17',          # C++17 standard
    '-march=native'        # Optimize for current CPU
]

extra_ldflags = [
    '-fopenmp',
    '-lgomp'               # Link GNU OpenMP
]
```

**Windows (MSVC):**
```python
extra_cflags = [
    '/O2',                 # Maximum optimization
    '/arch:AVX2',          # Enable AVX2
    '/openmp',             # OpenMP support
    '/std:c++17'
]
```

### Running Benchmarks

**On x86_64 Linux/Windows:**

```bash
# Test AVX2 optimization
python test_avx2_optimization.py

# Expected output:
# Small (128×1024×1024):  180-220 GFLOPS
# Medium (256×1024×1024): 200-240 GFLOPS
# Large (512×1024×1024):  220-260 GFLOPS
```

### Verification Steps

1. **Check CPU support:**
   ```python
   import platform
   print(platform.machine())  # Should show 'x86_64' or 'AMD64'
   ```

2. **Verify AVX2 availability:**
   ```bash
   # Linux
   lscpu | grep avx2

   # Or check /proc/cpuinfo
   grep avx2 /proc/cpuinfo
   ```

3. **Run correctness test:**
   ```python
   # Compares AVX2 output vs PyTorch
   # Max difference should be < 1e-3
   ```

---

## Platform-Specific Notes

### Intel CPUs

**Haswell and newer** (2013+):
- Full AVX2 support
- Good `_mm256_blendv_ps` performance
- Expected: 200-250 GFLOPS

**Skylake and newer** (2015+):
- Improved AVX2 throughput
- May support AVX-512 (check with `lscpu`)
- Expected: 220-260 GFLOPS

### AMD CPUs

**Zen 2 and newer** (2019+):
- Full AVX2 support
- Larger L3 cache (good for this workload)
- Expected: 200-240 GFLOPS

**Zen 3/4/5:**
- Excellent AVX2 performance
- May benefit from different prefetch distance
- Expected: 220-260 GFLOPS

**Note:** AMD CPUs may benefit from tuning `K_PREFETCH` (try 32, 64, 128).

---

## Comparison with ARM64 NEON

| Aspect | ARM64 NEON | x86_64 AVX2 |
|--------|------------|-------------|
| SIMD Width | 4 floats | 8 floats (2x) |
| Blocking | 12-way | 16-way |
| Measured GFLOPS | 126 | ~200-250 (expected) |
| Hardware Predication | `vbslq_f32` | `_mm256_blendv_ps` |
| Horizontal Sum | `vaddvq_f32` | Custom reduction |
| OpenMP Flags | `-Xpreprocessor -fopenmp` (macOS) | `-fopenmp` (standard) |

**Key Differences:**
1. **Wider SIMD** → 2x theoretical speedup
2. **More outputs** → Better instruction-level parallelism
3. **Standard OpenMP** → Easier compilation on Linux/Windows

---

## Future Optimizations

### Near-term (can add now)

1. **FMA Instructions:**
   ```cpp
   // Instead of separate multiply + add
   sum = _mm256_fmadd_ps(a, b, sum);  // sum += a * b
   ```
   **Note:** MatMul-free doesn't have multiplies, but FMA might help in reductions.

2. **Cache Blocking:**
   ```cpp
   // Block M dimension for L1/L2 cache
   for (int mb = 0; mb < M; mb += M_BLOCK) {
       for (int nb = 0; nb < N; nb += N_BLOCK) {
           // Process M_BLOCK × N_BLOCK tile
       }
   }
   ```

3. **Tune N_BLOCK:**
   - Test 8, 12, 16, 20, 24
   - May differ from ARM64 optimal

### Long-term (requires new kernel)

4. **AVX-512 Implementation:**
   - 16-wide SIMD (2x AVX2)
   - Mask registers for predication
   - Expected: 400-500 GFLOPS

---

## Troubleshooting

### "AVX2 not supported"

**Solution:** Check CPU with `lscpu` or use generic kernel.

### Compilation fails with "unknown instruction"

**Solution:** Update compiler or remove `-march=native`:
```python
extra_cflags = ['-O3', '-mavx2', '-fopenmp', '-std=c++17']
```

### Performance lower than expected

**Checklist:**
1. ✓ Compiled with `-O3`?
2. ✓ AVX2 enabled (`-mavx2`)?
3. ✓ OpenMP working? (check `num_threads`)
4. ✓ Using ternary weights {-1, 0, 1}?
5. ✓ Large enough matrices? (try 512×1024×1024)

### OpenMP not working on Linux

**Solution:**
```bash
# Install GNU OpenMP
sudo apt-get install libomp-dev

# Or use Intel OpenMP
sudo apt-get install intel-mkl
```

---

## Files Reference

### Core Implementation
- **`src/cpu_ops/kernels/x86_64/matmul_free_avx2.cpp`** - AVX2 kernel (main file)
- **`src/cpu_ops/setup_extensions.py`** - Compilation configuration
- **`test_avx2_optimization.py`** - Benchmark script

### Testing
```bash
# Run AVX2 benchmark (x86_64 only)
python test_avx2_optimization.py

# Expected output:
# ✓ AVX2 kernel compiled
# Small:  ~200 GFLOPS
# Medium: ~220 GFLOPS
# Large:  ~240 GFLOPS
```

---

## Summary

**Completed:**
- ✅ 16-way output blocking
- ✅ Hardware predication with `_mm256_blendv_ps`
- ✅ Prefetching
- ✅ OpenMP parallelization
- ✅ TRUE matmul-free (conditional add/subtract only)

**Expected Performance:**
- **200-250 GFLOPS** on modern x86_64 CPUs
- **2x faster** than ARM64 NEON
- **0.15-0.20x** vs PyTorch (hardware limitation)

**Next Steps:**
1. Test on Intel/AMD servers
2. Benchmark against PyTorch
3. Tune N_BLOCK if needed
4. Consider AVX-512 implementation

**Testing Required:**
Run `test_avx2_optimization.py` on x86_64 system to validate these optimizations!
