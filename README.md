# Litespark-Inference

**Fast CPU inference for ternary neural networks**

Litespark-Inference is a pip-installable Python library that enables efficient inference of ternary models on consumer CPUs. By exploiting the ternary weight structure ({-1, 0, +1}) with custom SIMD kernels, we eliminate floating-point multiplication entirely and achieve dramatic speedups over standard PyTorch inference.

## Key Results

### Apple Silicon (M1–M4)

![Performance on Apple Silicon](docs/figures/apple_silicon_summary.png)

*Performance comparison on Apple Silicon M4. Litespark-Inference achieves ~14× memory reduction, 9.2× faster TTFT, and 52× higher throughput compared to PyTorch.*

<div align="center">

| Metric | PyTorch | NEON | Accelerate |
|--------|---------|------|------------|
| Memory (MB) | 7,673 | 556 | 6,949 |
| TTFT (ms) | 2,632 | 288 | 373 |
| Throughput (tok/s) | 0.39 | 20.4 | 5.52 |

</div>

### Intel Ice Lake / AMD Zen4 (AVX-512 VNNI)

![Performance on AVX-512 VNNI](docs/figures/avx512_vnni_summary.png)

*Performance comparison on Intel Ice Lake / AMD Zen4 using AVX-512 VNNI kernels.*

<div align="center">

| Metric | PyTorch | AVX-512 VNNI | Speedup |
|--------|---------|--------------|---------|
| Memory (MB) | 7,800 | 556 | 14.0× |
| TTFT (ms) | 2,450 | 195 | 12.6× |
| Throughput (tok/s) | 0.42 | 11.2 | 26.7× |

</div>

### Intel Core Ultra (AVX-VNNI)

![Performance on AVX-VNNI](docs/figures/avx_vnni_summary.png)

*Performance comparison on Intel Core Ultra using AVX-VNNI kernels.*

<div align="center">

| Metric | PyTorch | AVX-VNNI | Speedup |
|--------|---------|----------|---------|
| Memory (MB) | 7,750 | 556 | 13.9× |
| TTFT (ms) | 2,580 | 310 | 8.3× |
| Throughput (tok/s) | 0.40 | 8.5 | 21.3× |

</div>

### Cross-Platform Comparison

![Cross-Platform Comparison](docs/figures/cross_platform_comparison.png)

*Cross-platform performance comparison showing consistent speedups across Apple Silicon, Intel, and AMD processors.*

## Comparison with BitNet.cpp v2

We benchmarked Litespark-Inference against Microsoft's BitNet.cpp v2 using their pp128+tg128 methodology (128-token prompt processing + 128-token generation).

### AMD EPYC 9R14 (AWS c7a.2xlarge)

![AMD EPYC Comparison](docs/figures/performance_comparison_amd_epyc_9r14_user.png)

*Scaling behavior on AMD EPYC 9R14. BitNet.cpp V2 shows strong prefill scaling, while all implementations converge on similar token generation performance at higher thread counts.*

<div align="center">

| Threads | Prefill (Original) | Prefill (V2) | Prefill (Litespark) | Gen (Original) | Gen (V2) | Gen (Litespark) |
|---------|-------------------|--------------|---------------------|----------------|----------|-----------------|
| 1 | 35.0 | 43.4 | 38.2 | 10.0 | 15.6 | **15.9** |
| 2 | 70.0 | 81.2 | 74.7 | 18.0 | 28.7 | 28.1 |
| 4 | 140.0 | 156.8 | 140.7 | 30.0 | 49.2 | 48.2 |
| 8 | 210.0 | **291.8** | 230.7 | 42.0 | 66.2 | **67.5** |

</div>

### Intel Xeon Platinum 8488C (AWS c7i.2xlarge)

![Intel Xeon Comparison](docs/figures/performance_comparison_intel_xeon_8488c_user.png)

*Scaling behavior on Intel Xeon Platinum 8488C. Litespark-Inference maintains a consistent lead in prefill throughput across all thread configurations.*

<div align="center">

| Threads | Prefill (Original) | Prefill (V2) | Prefill (Litespark) | Gen (Original) | Gen (V2) | Gen (Litespark) |
|---------|-------------------|--------------|---------------------|----------------|----------|-----------------|
| 1 | 27.0 | 43.4 | **59.7** | 10.0 | 13.3 | **13.6** |
| 2 | 40.0 | 65.8 | **85.9** | 13.0 | 19.1 | **19.5** |
| 4 | 55.0 | 77.9 | **110.2** | 16.0 | 24.3 | **25.0** |
| 6 | 79.0 | 101.3 | **120.7** | 20.0 | **29.5** | 28.0 |

</div>

### Apple M4 (MacBook Pro)

![Apple M4 Scaling](docs/figures/performance_comparison_apple_m4_user.png)

*Litespark-Inference scaling on Apple M4. Prefill throughput scales nearly linearly up to 4 threads, while token generation benefits from using all 10 CPU cores.*

<div align="center">

| Threads | Prefill pp128 (tok/s) | Generation tg128 (tok/s) |
|---------|----------------------|--------------------------|
| 1 | 26.1 | 6.5 |
| 2 | 43.1 | 11.0 |
| 4 | 81.9 | 15.4 |
| 8 | 101.2 | 14.0 |
| 10 | 108.8 | 19.6 |

</div>

## Supported Platforms

- **Apple Silicon** (M1/M2/M3/M4) — NEON SDOT instructions
- **Intel Ice Lake+** — AVX-512 VNNI instructions
- **AMD Zen4+** — AVX-512 VNNI instructions
- **Intel Core Ultra** — AVX-VNNI (256-bit) instructions

## Installation

```bash
pip install litespark-inference
```

**Requirements:**
- Python 3.10+
- PyTorch 2.4+

**macOS (recommended):**
```bash
brew install libomp
```
OpenMP enables multi-threaded kernel execution. Without it, inference will run single-threaded.

## Usage

### Supported Models

- `bitnet-2b`
- `falcon-edge-1b`
- `falcon-edge-1b-instruct`
- `falcon-edge-3b`
- `falcon-edge-3b-instruct`

### Command Line

```bash
# Generate text with the default BitNet model
litespark-inference generate "The meaning of life is"

# Generate text with Falcon Edge instruct
litespark-inference generate "What is the capital of France?" --model falcon-edge-1b-instruct

# Interactive chat with Falcon Edge instruct
litespark-inference chat --model falcon-edge-1b-instruct

# Run benchmark on your hardware
litespark-inference benchmark

# Show system info and detected SIMD capabilities
litespark-inference info
```

### Python API

```python
from litespark_inference import load_model

# Load the default BitNet 2B model (auto-downloads from Hugging Face)
model, tokenizer = load_model("bitnet-2b")

# Generate text
input_ids = tokenizer.encode("Hello, world!", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))

# Load a Falcon Edge instruct model
falcon_model, falcon_tokenizer = load_model("falcon-edge-1b-instruct")
```

### Kernel Modes (Apple Silicon)

Two inference modes are available on Apple Silicon:

```bash
# NEON mode (default) — fast int8 quantized inference, ~556 MB
litespark-inference generate "Hello" --mode neon

# Accelerate mode — float32 with Apple AMX, bit-exact accuracy, ~2.5 GB
litespark-inference generate "Hello" --mode accelerate
```

```python
# In Python
model, tokenizer = load_model("bitnet-2b", mode="neon")       # default, fast
model, tokenizer = load_model("bitnet-2b", mode="accelerate") # accurate
```

## How It Works

Ternary models use weights constrained to {-1, 0, +1}. This means matrix multiplication reduces to simple addition and subtraction:

```
y = Σ x_j · w_j  →  y = Σ(w=+1) x_j - Σ(w=-1) x_j
```

Litespark-Inference exploits this structure with custom SIMD kernels that:

1. **Store weights as int8** — enabling direct use of hardware dot product instructions
2. **Quantize activations per-row** — converting float32 inputs to int8 with scale factors
3. **Use hardware SIMD instructions** — NEON SDOT (ARM) or AVX-512 VPDPBUSD (x86)
4. **Apply zero-point correction** — maintaining numerical accuracy

The library automatically detects your CPU's SIMD capabilities and dispatches to the optimal kernel.

## Benchmarking

Run the built-in benchmark to measure performance on your hardware:

```bash
litespark-inference benchmark
```

Or use the benchmark scripts for detailed profiling:

```bash
python benchmark_kernel.py      # Kernel-level benchmarks
python benchmark_synthetic.py   # Synthetic workload benchmarks
```

## Citation

If you use Litespark-Inference in your research, please cite:

```bibtex
@article{litespark2024,
  title={Litespark Inference on Consumer CPUs: Custom SIMD Kernels for Ternary Neural Networks},
  author={Dade, Nii Osae Osae and Morri, Maurizio and Rahat, Moinul Hossain},
  year={2024}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
