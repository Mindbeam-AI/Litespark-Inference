# CPU-Dev Branch - MatMul-Free CPU Optimization

## Branch Purpose
This branch focuses exclusively on CPU-optimized implementations of MatMul-Free operations, including support for:
- **Intel/AMD x86_64**: AVX2, AVX-512 instruction sets
- **Apple Silicon (M1/M2/M3)**: ARM64 with NEON vectorization
- **ARM64 Linux**: General ARM64 support for servers/edge devices

## Key Differences from Main Branch

### Removed Components
- ❌ All Triton kernels and CUDA dependencies
- ❌ GPU-specific memory management
- ❌ DDP training scripts (focus on inference)
- ❌ Mixed precision training utilities

### New Components
- ✅ CPU-optimized MatMul-free kernels (C++/Assembly)
- ✅ Cross-platform SIMD implementations
- ✅ Apple Silicon specific optimizations
- ✅ CPU-friendly memory layouts
- ✅ Threading strategies for CPU cores

## Directory Structure

```
src/
├── cpu_ops/                    # CPU-optimized operations
│   ├── __init__.py
│   ├── matmul_free_cpu.py     # Python interface
│   ├── kernels/               # C++ kernel implementations
│   │   ├── x86_64/           # Intel/AMD optimizations
│   │   ├── arm64/            # ARM64/Apple Silicon
│   │   └── generic/          # Fallback implementations
│   ├── simd/                 # SIMD utilities
│   └── threading/            # CPU threading strategies
├── models/                   # CPU-optimized model implementations
├── modules/                  # CPU-friendly modules
└── utils/                    # CPU-specific utilities

configs/
├── cpu_models.yaml          # CPU-optimized model configurations
└── inference_configs.yaml  # CPU inference settings

scripts/
├── benchmark_cpu.py         # CPU performance benchmarking
├── test_cpu_ops.py         # CPU operation testing
└── inference_cpu.py        # CPU-only inference script
```

## Development Phases

### Phase 1: Core CPU Kernels ⚡
- [x] Create branch structure
- [ ] Implement basic MatMul-free CPU kernel
- [ ] Add x86_64 AVX2/AVX-512 optimizations
- [ ] Add ARM64/NEON optimizations for Apple Silicon
- [ ] Create Python bindings

### Phase 2: Model Integration 🔧
- [ ] CPU-optimized model classes
- [ ] Remove GPU dependencies
- [ ] Implement CPU-friendly attention mechanisms
- [ ] Add automatic hardware detection

### Phase 3: Performance & Testing 📊
- [ ] Comprehensive benchmarking suite
- [ ] Cross-platform testing
- [ ] Memory usage optimization
- [ ] Threading performance tuning

## Target Performance Goals

| Metric | Target Improvement |
|--------|-------------------|
| Linear Layer Speed | 2-3x faster than PyTorch CPU |
| Memory Usage | 4-8x reduction (ternary weights) |
| Model Loading | 3-5x faster (compressed weights) |
| Apple Silicon | Native ARM64 optimization |
| Power Efficiency | 2-3x better for inference |

## Hardware Support Matrix

| Platform | ISA | Status | Optimizations |
|----------|-----|--------|---------------|
| Intel x86_64 | AVX2 | 🟡 Planned | Vectorized add/subtract |
| Intel x86_64 | AVX-512 | 🟡 Planned | 16-wide SIMD operations |
| AMD x86_64 | AVX2 | 🟡 Planned | Same as Intel AVX2 |
| Apple M1/M2/M3 | ARM64+NEON | 🟡 Planned | ARM64 native + Metal? |
| ARM64 Linux | NEON | 🟡 Planned | Server/edge deployment |
| Generic | Fallback | 🟡 Planned | Pure C++ implementation |

## Next Steps
1. Implement core CPU MatMul-free kernel
2. Add SIMD optimizations for x86_64 and ARM64
3. Create PyTorch C++ extensions
4. Benchmark against PyTorch CPU backend
