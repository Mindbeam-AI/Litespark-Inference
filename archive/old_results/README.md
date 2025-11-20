# Old Training Results

This directory contains results and reports from v3 training (Sessions 1-2).

## Files

### `comparison_results.json` (Nov 14, 290B)
- v3 benchmark results comparing against Pythia-410M baseline
- Results: MatMul-Free v3 performed worse than baseline (opposite of paper claims)

### `matmulFree_v1-report.pdf` (Nov 13, 49KB)
- Visual report from v3 training
- 2-page PDF with training metrics and comparisons

### `training_results_step9000_20251114_001031.pdf` (Nov 14, 49KB)
- Training report from v3 at step 9000
- Model: 257M parameters, loss ~5.55

### `v3_presentation.html` (Nov 13, 16KB)
- 5-slide HTML presentation summarizing v3 results
- Covers model overview, training progress, architecture, benchmarks, and analysis

## v3 Training Summary

**Configuration**:
- Model: 257M parameters (12 layers, hidden_size=1024)
- Steps: 10,000 completed
- Loss trajectory: 10.92 → 4.44 (best) → 6.62 (final)

**Key Findings** (documented in SESSION2_NOTES.md):
- Performance worse than baseline due to:
  - Triton autotuning disabled (10-40% performance loss)
  - Only 2 of 4 GPUs working (batch_size=2 < num_GPUs=4)
  - Batch size too small (effective batch=4 vs paper's 256)
  - Hardware mismatch (A10G vs paper's A100)

**Current Results**: See root directory for latest benchmark results from Session 3 training with autotuning enabled.
