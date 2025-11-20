# Session 2 Notes - MatMul-Free LM v3 Analysis & v4 Setup

## Date: 2025-11-14

## Summary
Completed analysis of v3 training results, created presentation, and set up v4 experiment with proper 370M parameter model to match paper's configuration.

---

## V3 Results Analysis

### Training Completed
- **Model:** 257M parameters (12 layers, hidden_dim=1024)
- **Steps:** 10,000 completed
- **Loss trajectory:** 10.92 → 4.44 (best) → 6.62 (final)
- **Checkpoint used:** Step 9000 (loss 5.55)

### Benchmark Results vs Pythia-410M Baseline

| Metric | MatMul-Free v3 | Pythia-410M | Difference |
|--------|----------------|-------------|------------|
| Model Size | 257M | 405M | 0.63x |
| Memory (seq=512) | 2.51 GB | 0.92 GB | **+171%** |
| Memory (seq=1024) | 2.61 GB | 1.13 GB | **+131%** |
| Memory (seq=2048) | 2.93 GB | 1.43 GB | **+104%** |
| Throughput (batch=4, seq=1024) | 28,713 tok/s | 61,726 tok/s | **-53.5%** |

**Conclusion:** MatMul-Free v3 performed WORSE than baseline - opposite of paper's claims.

---

## Why V3 Failed: Root Cause Analysis

### 1. **Triton Autotuning Disabled**
- Original repo: Has autotuning ENABLED with up to 32 warps
- Our implementation: Autotuning COMMENTED OUT (due to Triton 2.2+ compatibility issues)
- Impact: Kernels run with fixed default config (BLOCK_SIZE=64, num_warps=4)
- Expected loss: ~10-20% performance

### 2. **GPU Utilization Problem**
- Available: 4x NVIDIA A10G GPUs
- Actually working: Only 2 GPUs actively computing
- Root cause: batch_size=2 < num_GPUs=4 (DataParallel can't split properly)
- Impact: 50% compute wasted

### 3. **Batch Size Too Small**
- Paper used: batch_size=256
- V3 used: effective batch=4
- Fused kernels need large batches to amortize overhead
- Small batches = overhead dominates, lose all benefits

### 4. **Hardware Mismatch**
- Paper used: A100 (2 TB/s memory bandwidth) for benchmarks
- We used: A10G (600 GB/s memory bandwidth)
- MatMul-free operations are memory-bandwidth bound
- Lower bandwidth = slower execution

### 5. **HGRN Attention is Sequential**
- HGRN processes tokens one-by-one (recurrent, sequential)
- Standard softmax attention parallelizes across sequence
- GPUs strongly prefer parallel work
- Sequential work = GPU underutilization

### 6. **Pythia is Heavily Optimized**
- HuggingFace/EleutherAI spent years optimizing Transformers
- FlashAttention, FP16 mixed precision, optimized CUDA kernels
- Our MatMul-free has broken optimizations competing against polished baseline

---

## Paper Hardware Investigation

### What We Found
**From paper (arxiv.org/html/2406.02528):**
- Benchmarks explicitly used: **NVIDIA A100 80GB GPU**
- Batch size: 256
- Sequence length: 1024 during training
- Model sizes tested: 370M, 1.3B, 2.7B parameters

**What paper DOESN'T say:**
- Which GPUs were used for main pre-training (370M/1.3B/2.7B models)
- How many GPUs for training
- Full training infrastructure specs

**Note:** The paper is vague about training hardware. Only benchmarks are explicitly on A100.

---

## V4 Experiment Setup

### Goal
Train a proper 370M parameter model matching paper's configuration and benchmark against Pythia-410M.

### V4 Configuration

**Model:**
- Size: ~370M parameters (matching paper's 370M model)
- Architecture: 24 layers, hidden_dim=1024, 8 heads
- Vocabulary: 50,000 tokens
- Max sequence: 2,048

**Training:**
- Dataset: SlimPajama-6B-nanotron
- Sequence length: **1024** (vs v3's 128) - matching paper
- Batch size per GPU: **4** (vs v3's 2)
- Effective batch: **16** (4 GPUs × 4 per GPU)
- Total steps: 10,000
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
- Scheduler: CosineAnnealingLR (T_max=10,000)
- Gradient clipping: 1.0

**Hardware:**
- 4x NVIDIA A10G (24GB each)
- Parallelization: DataParallel (for now)

**Key Improvements over V3:**
1. ✅ Model size: 370M vs 257M (closer to paper)
2. ✅ Sequence length: 1024 vs 128 (matching paper)
3. ✅ Batch size: 16 vs 4 (all 4 GPUs will work)
4. ⚠️ Autotuning: Still disabled (need to fix separately)

**Expected Memory:**
- V3 used ~4.23 GB per GPU for 257M model with seq_len=128
- V4 estimate: ~6-8 GB per GPU for 370M model with seq_len=1024
- Should fit in 24GB A10Gs with some headroom

---

## V4 Training Status

### Initial Attempt
- Started v4 training
- **Got OOM error** on GPU 0
- Error: GPU 0 tried to allocate 10.4GB + existing 12.2GB = exceeded 23GB limit

### Memory Distribution (DataParallel Issue)
```
GPU 0: 12.2 GB (holds model + gradient gathering)
GPU 1: 4.7 GB
GPU 2: 4.7 GB
GPU 3: 4.6 GB
```
DataParallel puts more load on GPU 0 (it's the "master" GPU).

### Actions Taken
1. Killed old processes (PIDs 530802, 938077) - freed ~12GB on GPU 0
2. Killed OOM'd process (PID 1015713)
3. Ready to restart training with clean memory

### Current Status
**READY TO TRAIN** - User needs to:
1. Restart Jupyter kernel
2. Re-run v4.ipynb from beginning
3. Training should fit now with freed memory

---

## Files Created This Session

### 1. `v4.ipynb`
- New notebook for 370M parameter training
- 17 cells covering:
  - Environment setup
  - Model creation (24 layers, 370M params)
  - Dataset loading (seq_len=1024)
  - Training loop (batch_size=4, all 4 GPUs)
  - Evaluation vs Pythia-410M
  - Memory, throughput, perplexity benchmarks

### 2. `v3_presentation.html`
- 5-slide HTML presentation summarizing v3 results
- Navigation: Arrow keys or buttons
- Content:
  - Slide 1: Model overview (257M params, 4x A10G)
  - Slide 2: Training progress (loss 10.92 → 4.44)
  - Slide 3: Architecture (ternary weights, HGRN, fused kernels)
  - Slide 4: Benchmark results (vs Pythia-410M)
  - Slide 5: Analysis (why it failed)

### 3. `v3_presentation.md`
- Markdown version of presentation (created first, then converted to HTML)

### 4. `matmulFree_v1-report.pdf`
- Generated by export_results_to_pdf.py (from previous session)
- 2-page report with training metrics and comparisons

---

## Paper Comparison: Expected vs Actual

### Paper's Claims (for 370M model)
- Memory reduction: Up to 61% vs baseline
- Throughput improvement: Higher than baseline
- Training efficiency: Faster with fused kernels

### Our V3 Results
- Memory: +104% vs baseline (WORSE)
- Throughput: -53.5% vs baseline (WORSE)
- Complete opposite of paper's claims

### Why the Discrepancy?
1. **Batch size:** Paper used 256, we used 4 (64x smaller)
2. **Autotuning:** Paper has it working, we disabled it
3. **GPU utilization:** Paper likely used all GPUs properly, we only used 2
4. **Hardware:** Paper used A100 (2 TB/s), we used A10G (600 GB/s)
5. **Implementation quality:** Paper's code is optimized, ours has issues

---

## Next Steps (Priority Order)

### Immediate (In Progress)
1. **Complete v4 training** (370M params, seq_len=1024, batch_size=16)
   - Monitor memory usage
   - Check all 4 GPUs are working
   - Train for 10,000 steps
   - Benchmark vs Pythia-410M

### After V4 Training
2. **Analyze v4 results:**
   - Does larger model + longer sequences improve results?
   - Are all 4 GPUs utilized properly?
   - How do metrics compare to v3?

3. **Re-enable Triton autotuning** (Critical)
   - Compare original repo's fusedbitnet.py with ours
   - Copy autotuning decorators from original
   - Fix Triton 2.2+ compatibility issues
   - Benchmark performance improvement

4. **Switch to DDP** (Optional but recommended)
   - More efficient than DataParallel
   - Expected: ~40-50% efficiency gain
   - Requires multi-process setup (torchrun)
   - Reference: train.py already has DDP implementation

5. **Scale batch size** (If memory allows)
   - Try batch_size=8 per GPU (effective=32)
   - Or even higher to approach paper's batch_size=256
   - Larger batches = fused kernels shine

### Long Term
6. **Full training run**
   - Train to convergence (paper used 15B tokens for 370M)
   - Current v4 will do ~0.16B tokens (10K steps × 16 batch × 1024 seq)
   - Need ~94x more training

7. **Compare with paper's reported metrics**
   - Perplexity on standard benchmarks
   - Memory efficiency across sequence lengths
   - Throughput comparison

---

## Key Insights from This Session

### 1. Implementation Matters More Than Architecture
A broken/unoptimized implementation of a theoretically-better architecture will lose to a heavily-optimized standard architecture every time. The "MatMul-free" concept might work, but only with proper implementation.

### 2. Batch Size is Critical
Fused kernels and quantized operations have overhead. Small batches = overhead dominates. Need batch_size ≥ 64-256 to see benefits.

### 3. Hardware Bandwidth Matters
MatMul-free doesn't eliminate memory access, just changes compute pattern. On memory-bandwidth-limited hardware (A10G), this doesn't help. Need high-bandwidth GPUs (A100/H100).

### 4. Sequential Attention is Problematic
HGRN's recurrent nature fundamentally limits parallelism. This is an architectural limitation, not an implementation bug. May need different attention mechanism.

### 5. Papers Often Underspecify Training Setup
The paper doesn't fully document:
- Exact training hardware for main models
- Number of GPUs used
- Full hyperparameter details
- Makes reproduction difficult

---

## Debugging Notes

### OOM Issues
**Problem:** 370M model with seq_len=1024 doesn't fit on 4x A10G
**Cause:** DataParallel loads GPU 0 more heavily (12GB vs 4-5GB on others)
**Solutions:**
1. Kill old processes to free memory ✅
2. Reduce batch_size to 2 per GPU (effective=8)
3. Reduce sequence length to 512
4. Use gradient accumulation
5. Switch to DDP (more balanced memory)

### GPU Utilization
**V3 issue:** Only 2 of 4 GPUs working
**Cause:** batch_size=2 < num_GPUs=4, DataParallel can't split
**V4 fix:** batch_size=4 ≥ num_GPUs=4, all GPUs should work

### Autotuning Disabled
**Problem:** Triton 2.2+ changed parameter binding requirements
**Workaround:** Commented out @triton.autotune decorators
**Impact:** ~10-20% performance loss, maybe more
**Fix needed:** Compare with original repo, copy their working solution

---

## Commands Reference

### Kill Training Process
```bash
# Find PID
nvidia-smi

# Kill specific process
kill -9 <PID>

# Verify
nvidia-smi
```

### Monitor GPU Memory
```bash
# One-time check
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

### Jupyter Kernel Management
1. Kernel → Restart & Clear Output
2. Re-run cells from beginning

---

## Important File Locations

### Training Checkpoints
- `checkpoint_step_9000.pt` - v3 final checkpoint (257M params)
- `checkpoint_v4_step_*.pt` - v4 checkpoints (370M params, will be created)

### Notebooks
- `v3.ipynb` - Previous training (257M, seq_len=128) ✅ COMPLETED
- `v4.ipynb` - Current training (370M, seq_len=1024) 🔄 IN PROGRESS

### Results & Reports
- `comparison_results.json` - v3 benchmark data
- `comparison_results_v4.json` - v4 benchmark data (will be created)
- `matmulFree_v1-report.pdf` - v3 visual report (2 pages)
- `v3_presentation.html` - v3 presentation (5 slides)

### Source Code
- `src/ops/fusedbitnet.py` - Fused kernels (autotuning disabled)
- `src/ops/hgrn/recurrent_fuse.py` - HGRN attention kernels
- `src/models/hgrn_bit/modeling_hgrn_bit.py` - Model implementation

### Original Repo (for comparison)
- Location: `/Users/mauriziomorri/Desktop/matmulfreellm/`
- Key file: `mmfreelm/ops/fusedbitnet.py` (has autotuning ENABLED)

---

## Session End Status

**Current Task:** V4 training about to restart
**Blocking Issue:** None (memory cleared)
**Ready to Continue:** YES

**When you return:**
1. Check if v4 training completed
2. Analyze v4 results vs v3
3. Work on re-enabling autotuning
4. Consider DDP migration

---

## Questions Answered This Session

**Q: "What hardware did the paper use?"**
A: A100 80GB for benchmarks (explicitly stated). Main training hardware NOT specified in paper.

**Q: "Why is MatMul-free slower than baseline?"**
A: Multiple issues - autotuning disabled, only 2 GPUs working, batch too small, sequential HGRN attention, hardware bandwidth limited, competing against heavily optimized Pythia.

**Q: "Is it possible to train 410M parameter model?"**
A: Yes - v4 targets 370M (close enough). Paper's 370M model is comparable to Pythia-410M.

**Q: "How do I kill processes and restart training?"**
A: `kill -9 <PID>`, then restart Jupyter kernel and re-run cells.

---

## Notes for Next Session

1. **Check v4 training progress** - should be running or completed
2. **Verify all 4 GPUs working** - nvidia-smi should show ~6-8GB on all 4
3. **Compare v4 vs v3 results** - did larger model + longer sequences help?
4. **Priority: Re-enable autotuning** - this is the biggest bottleneck
5. **Consider:** If v4 still shows poor results, might need to question if the architecture works at all on commodity hardware

---

## Contact Information / References

- **Paper:** "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)
- **Original Repo:** github.com/ridgerchu/matmulfreellm
- **Our Repo:** github.com/tonymindbeam/matmulMM
- **Hardware:** AWS SageMaker ml.g5.12xlarge (4x A10G, 24GB each)
- **Dataset:** SlimPajama-6B-nanotron
- **Baseline:** EleutherAI/pythia-410m-deduped (405M params)
