# Litespark-Inf

Fast CPU inference for Microsoft BitNet 1.58-bit models.

## Supported Platforms

- **Apple Silicon** (M1/M2/M3/M4) - NEON SDOT
- **Intel/AMD x86_64** - AVX-512 VNNI (Ice Lake+, Zen4+)
- **Intel Core Ultra** - AVX-VNNI (256-bit)

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python 3.9+
- PyTorch 2.0+
- macOS: `brew install libomp`

## Usage

### Command Line

```bash
# Generate text
litespark-inf generate "The capital of France is"

# Interactive chat
litespark-inf chat

# Run benchmark
litespark-inf benchmark

# System info
litespark-inf info
```

### Kernel Modes

Two modes available on Apple Silicon:

```bash
# NEON (default) - fast, int8 quantized, ~556 MB
litespark-inf generate "Hello" --mode neon

# Accelerate - accurate, float32, ~2.5 GB
litespark-inf generate "Hello" --mode accelerate
```

### Python API

```python
from litespark_inf import load_model

model, tokenizer = load_model("bitnet-2b")
# or: load_model("bitnet-2b", mode="accelerate")

input_ids = tokenizer.encode("Hello", return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
```

## Performance

On Apple M4:
- NEON: ~556 MB memory, ~20 tokens/sec
- Accelerate: ~2.5 GB memory, ~10-15 tokens/sec

## License

MIT
