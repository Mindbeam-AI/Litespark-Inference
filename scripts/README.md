# Paper Replication Scripts

Scripts to replicate experiments from "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)

## Training Scripts

### 370M Model (Paper's MMfreeLM-370M)
```bash
bash scripts/train_370M_paper.sh
```
- **Parameters:** ~370M
- **Hardware:** 8x A100 80GB
- **Effective Batch:** 256 (8 GPUs × 16 × 2)
- **Training Time:** ~4-6 hours for 10K steps

### 1.3B Model (Paper's MMfreeLM-1.3B)
```bash
bash scripts/train_1.3B_paper.sh
```
- **Parameters:** ~1.3B
- **Hardware:** 8x A100 80GB
- **Effective Batch:** 256 (8 GPUs × 8 × 4)
- **Training Time:** ~12-16 hours for 10K steps

### 2.7B Model (Paper's MMfreeLM-2.7B)
```bash
bash scripts/train_2.7B_paper.sh
```
- **Parameters:** ~2.7B
- **Hardware:** 8x A100 80GB (requires 80GB VRAM)
- **Effective Batch:** 256 (8 GPUs × 4 × 8)
- **Training Time:** ~24-30 hours for 10K steps

## Evaluation Scripts

### Perplexity Evaluation (Paper's Table 3)
```bash
python scripts/evaluate_paper.py \
  --checkpoint checkpoints/370M_paper_match/checkpoint_step_10000.pt \
  --datasets wikitext-2 wikitext-103 ptb \
  --seq_length 1024 \
  --output results/eval_370M.json
```

**Evaluates on:**
- WikiText-2 (main benchmark)
- WikiText-103
- Penn TreeBank (PTB)
- LAMBADA (optional)

### Benchmarking (Paper's Table 2)
```bash
python scripts/benchmark_paper.py \
  --checkpoint checkpoints/370M_paper_match/checkpoint_step_10000.pt \
  --baseline EleutherAI/pythia-410m-deduped \
  --seq_lengths 512 1024 2048 4096 \
  --batch_size 4 \
  --output results/benchmark_370M.json
```

**Benchmarks:**
- Memory efficiency across sequence lengths
- Throughput (tokens/sec)
- Latency (ms/batch)
- Comparison with Pythia baseline

### Skip Baseline Comparison
```bash
python scripts/benchmark_paper.py \
  --checkpoint checkpoints/370M_paper_match/checkpoint_step_10000.pt \
  --skip_baseline \
  --output results/benchmark_370M_only.json
```

## Running in tmux

For long training runs, use tmux:

```bash
# Start tmux session
tmux new -s training_370M

# Run training script
bash scripts/train_370M_paper.sh

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training_370M
```

## Expected Results (from Paper)

### 370M Model
- **WikiText-2 Perplexity:** ~25-30
- **Memory (seq=1024):** ~1.5-2.0 GB
- **Throughput:** ~50K-100K tokens/sec (on A100)
- **vs Pythia-410M:** -40% memory, similar perplexity

### 1.3B Model
- **WikiText-2 Perplexity:** ~18-22
- **Memory (seq=1024):** ~4-6 GB
- **vs Pythia-1.4B:** -50% memory, similar perplexity

### 2.7B Model
- **WikiText-2 Perplexity:** ~15-18
- **Memory (seq=1024):** ~8-12 GB
- **vs Pythia-2.8B:** -55% memory, similar perplexity

## Model Comparison Table

| Model | Parameters | Layers | Hidden Size | Heads | Baseline |
|-------|-----------|---------|-------------|-------|----------|
| 370M  | 370M      | 24      | 1024        | 8     | Pythia-410M |
| 1.3B  | 1.3B      | 24      | 2048        | 16    | Pythia-1.4B |
| 2.7B  | 2.7B      | 48      | 2048        | 16    | Pythia-2.8B |

## Hardware Requirements

| Model | Min GPUs | Min VRAM/GPU | Recommended |
|-------|----------|--------------|-------------|
| 370M  | 4        | 24 GB        | 8x A100 80GB |
| 1.3B  | 8        | 40 GB        | 8x A100 80GB |
| 2.7B  | 8        | 80 GB        | 8x A100 80GB |

## Directory Structure

```
checkpoints/
├── 370M_paper_match/
│   ├── checkpoint_step_1000.pt
│   ├── checkpoint_step_2000.pt
│   └── ...
├── 1.3B_paper_match/
└── 2.7B_paper_match/

results/
├── eval_370M.json
├── eval_1.3B.json
├── benchmark_370M.json
└── ...
```

## Notes

1. **Batch Size:** All scripts use effective batch size of 256 to match the paper
2. **Autotuning:** First training run will be slower (Triton autotuning warmup)
3. **Checkpointing:** Models saved every 1000 steps
4. **Logging:** Progress logged every 100 steps
5. **Hardware:** Scripts optimized for 8x A100 80GB GPUs

## Troubleshooting

### OOM Errors
- Reduce `--batch_size` (will increase `gradient_accumulation_steps` automatically)
- Enable mixed precision: add `--use_amp` (if implemented in train_ddp.py)
- Reduce `--seq_length` to 512

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Verify all GPUs are being used
- Ensure autotuning has completed (first few steps are slower)

### Different Hardware
For 4x A100 GPUs, adjust in training scripts:
```bash
BATCH_SIZE_PER_GPU=16
GRAD_ACCUM_STEPS=4  # Keeps effective batch at 256
```

## Citation

If using these scripts, please cite the original paper:

```bibtex
@article{zhu2024scalable,
  title={Scalable MatMul-free Language Modeling},
  author={Zhu, Rui-Jie and Yu, Zhao and Wang, Zhenqiang and others},
  journal={arXiv preprint arXiv:2406.02528},
  year={2024}
}
```
