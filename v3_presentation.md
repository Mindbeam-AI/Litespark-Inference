# MatMul-Free Language Model - V3 Training Report

## Slide 1: What We Built

**Model:** MatMul-Free Language Model (v3)
- **Size:** ~257M parameters
- **Architecture:**
  - 12 layers, hidden_dim=1024, 8 attention heads
  - Ternary weights (1.58-bit: {-1, 0, 1})
  - 8-bit activation quantization
  - HGRN attention (replaces softmax)
  - Fused Triton kernels (autotuning disabled)

**Training Setup:**
- Dataset: SlimPajama-6B-nanotron
- Sequence length: 128 tokens
- Batch size: 2 per GPU
- Hardware: 4x NVIDIA A10G (24GB)
- **Issue:** Only 2 of 4 GPUs actively computing

---

## Slide 2: Training Progress

**Training Run:** 10,000 steps over ~X hours

**Loss Trajectory:**
- Initial (Step 0): 10.92
- Best (Step 9800): 4.44 (59% improvement)
- Final (Step 9900): 6.62
- Checkpoint (Step 9000): 5.55

**Notable Event:**
- Step 1600: Anomalous spike to 10.11
- Recovered within 100 steps (gradient clipping saved us)

**Takeaway:** Model learned successfully, loss decreased as expected

---

## Slide 3: Benchmark Setup

**Compared Against:** Pythia-410M (EleutherAI baseline)
- Standard Transformer architecture
- 405M parameters (similar to our 257M)
- Heavily optimized by HuggingFace/EleutherAI

**Tests Performed:**
1. **Memory Usage** - Peak GPU memory across sequence lengths [512, 1024, 2048]
2. **Throughput** - Tokens/second (batch=4, seq=1024)
3. **WikiText-2 Perplexity** - Language modeling quality

**Hardware Assignment:**
- MatMul-Free: GPU 0
- Pythia-410M: GPU 1

---

## Slide 4: Results - Not What We Expected

| Metric | MatMul-Free (v3) | Pythia-410M | Difference |
|--------|------------------|-------------|------------|
| **Model Size** | 257M params | 405M params | 0.63x |
| **Memory (seq=2048)** | 2.93 GB | 1.43 GB | **+104% (worse)** |
| **Throughput** | 28,713 tok/s | 61,726 tok/s | **-53% (worse)** |

**What the Paper Claimed:**
- Lower memory usage
- Higher throughput
- Better efficiency

**What We Got:**
- HIGHER memory (despite ternary weights + 8-bit activations)
- LOWER throughput (despite "MatMul-free")
- Opposite of paper's claims

---

## Slide 5: Why Did This Happen?

**Implementation Issues Found:**

1. **Triton Autotuning DISABLED**
   - Original repo: ENABLED with up to 32 warps
   - Our implementation: Fixed config (BLOCK_SIZE=64, num_warps=4)
   - Impact: Kernels running with non-optimal parameters

2. **GPU Utilization Problem**
   - Available: 4x A10G GPUs
   - Actually working: Only 2 GPUs
   - Reason: batch_size=2 < num_GPUs=4 (DataParallel can't split)

3. **Batch Size Too Small**
   - Paper used: batch_size=256
   - We used: effective batch=4
   - Fused kernels need large batches to amortize overhead

4. **Hardware Mismatch**
   - Paper used: A100 (2 TB/s memory bandwidth)
   - We used: A10G (600 GB/s)
   - MatMul-free is memory-bandwidth bound

5. **Sequential Attention**
   - HGRN processes tokens sequentially (one-by-one)
   - Standard attention parallelizes across sequence
   - GPUs prefer parallel work

**Conclusion:** The architecture might work with proper optimization, but our implementation has broken optimizations competing against a highly-polished baseline (Pythia).

---

## Next Steps (V4)

**What We're Fixing:**
1. Scale to 370M params (match paper's model size)
2. Increase sequence length to 1024 (match paper)
3. Increase batch size to 4 minimum (use all GPUs)
4. Re-enable Triton autotuning
5. Switch to DDP for better multi-GPU efficiency

**Goal:** Determine if MatMul-free approach actually works when properly implemented, or if the paper's claims don't hold on commodity hardware.
