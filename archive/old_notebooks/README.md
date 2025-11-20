# Old Training Notebooks

This directory contains Jupyter notebooks from Sessions 1-2 training experiments.

## Files

### `v3.ipynb` (Nov 14, 226KB)
- **Session**: Session 2
- **Model**: 257M parameters (12 layers, hidden_size=1024)
- **Status**: Training completed, results documented in SESSION2_NOTES.md
- **Results**: Training loss 10.92 → 4.44, final loss 6.62
- **Issues**: Only 2 of 4 GPUs working, batch size too small, autotuning disabled

### `v4.ipynb` (Nov 14, 34KB)
- **Session**: Session 2
- **Model**: 370M parameters (24 layers, hidden_size=1024)
- **Status**: Training attempted with seq_len=1024
- **Issues**: OOM errors with DataParallel on 4x A10G GPUs

### `Mm370vsP410.ipynb` (Nov 14, 34KB)
- **Session**: Session 2
- **Description**: Comparison notebook between 370M MatMul-free and Pythia-410M
- **Status**: Functionality now in benchmark scripts

## Why Archived

These notebooks were part of the development process but have been superseded by:
- Production training script: `train_ddp.py`
- Comprehensive benchmarks: `paper_training_efficiency_benchmark.py`, `evaluate_paper_benchmarks.py`
- Detailed session notes documenting findings

**For new training, use `train_ddp.py`. For benchmarks, use the scripts in the root directory.**
