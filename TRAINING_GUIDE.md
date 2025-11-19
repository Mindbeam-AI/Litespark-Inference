# Training Guide

## Quick Start

### 1. Check System Readiness
```bash
python check_training_readiness.py
```

This will verify:
- 8x GPUs available
- Dataset present
- All dependencies installed
- Training scripts ready

### 2. Choose Model Size

We have 3 model configurations matching the paper:

| Model | Parameters | Hidden Size | Layers | Batch Size | GPU Memory | Training Time |
|-------|-----------|-------------|---------|------------|-----------|---------------|
| 370M  | ~370M     | 1024        | 24      | 4/GPU      | ~30GB     | ~6 hours      |
| 1.3B  | ~1.3B     | 2048        | 24      | 2/GPU      | ~50GB     | ~12 hours     |
| 2.7B  | ~2.7B     | 2048        | 48      | 4/GPU      | ~80GB     | ~24 hours     |

**Note**: Training times are estimates for 10K steps on 8x A100 GPUs.

### 3. Start Training

#### 370M Model (Safe choice)
```bash
bash scripts/train_370M_paper.sh
```

#### 1.3B Model (Medium)
```bash
bash scripts/train_1.3B_paper.sh
```

#### 2.7B Model (Requires 80GB GPUs)
```bash
bash scripts/train_2.7B_paper.sh
```

### 4. Monitor Training

Training logs will show:
```
Step 100/10000 | Loss: 6.234 | LR: 0.000500 | Tokens/sec: 42,123 | Time: 1.23s
```

Checkpoints are saved every 1000 steps to `checkpoints/<model_name>/`.

### 5. Resume Training (if interrupted)

```bash
# Edit the training script and add:
--resume_from checkpoints/<model_name>/checkpoint_step_<N>.pt
```

## Training Modes

### Default: Training Mode (Fast)
```bash
bash scripts/train_370M_paper.sh
```
- Uses `F.linear` (standard matmul)
- 100-1000x faster
- Matches original paper repository
- **Recommended for training**

### Eval Mode: True MatMul-Free (Slow)
```bash
MATMUL_FREE_MODE=eval bash scripts/train_370M_paper.sh
```
- Uses add/subtract-only operations
- Much slower (200x slower)
- Proves the paper's core innovation
- **Not recommended for training, only for validation**

## What Each Script Does

All training scripts:
1. Set model architecture parameters
2. Configure training hyperparameters
3. Launch `train_ddp.py` with DistributedDataParallel
4. Save checkpoints every 1000 steps
5. Log metrics every 100 steps

## Configuration Details

### 370M Model
```bash
HIDDEN_SIZE=1024
NUM_LAYERS=24
NUM_HEADS=8
SEQ_LENGTH=1024
BATCH_SIZE_PER_GPU=4
GRAD_ACCUM_STEPS=8
EFFECTIVE_BATCH=256  # 8 GPUs * 4 batch * 8 accum
```

### 1.3B Model
```bash
HIDDEN_SIZE=2048
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=1024
BATCH_SIZE_PER_GPU=2
GRAD_ACCUM_STEPS=16
EFFECTIVE_BATCH=256  # 8 GPUs * 2 batch * 16 accum
```

### 2.7B Model
```bash
HIDDEN_SIZE=2048
NUM_LAYERS=48
NUM_HEADS=16
SEQ_LENGTH=1024
BATCH_SIZE_PER_GPU=4
GRAD_ACCUM_STEPS=8
EFFECTIVE_BATCH=256  # 8 GPUs * 4 batch * 8 accum
```

All models use:
- Effective batch size: 256
- Learning rate: 3e-4 to 5e-4
- Weight decay: 0.1
- Gradient clipping: 1.0
- 10,000 training steps

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in the script
BATCH_SIZE_PER_GPU=2  # was 4

# Increase gradient accumulation to maintain effective batch
GRAD_ACCUM_STEPS=16  # was 8
```

### Dataset Not Found
```bash
# Check dataset location
ls SlimPajama-6B-nanotron/train/*.ds

# If not found, update DATA_DIR in the training script
DATA_DIR="path/to/your/dataset"
```

### Slow Training
```bash
# Make sure you're in training mode (not eval)
echo $MATMUL_FREE_MODE  # Should be empty or "train"

# If it says "eval", unset it:
unset MATMUL_FREE_MODE
```

### Port Already in Use
```bash
# Edit train_ddp.py and change MASTER_PORT
os.environ['MASTER_PORT'] = '12356'  # was '12355'
```

## Expected Performance

### Training Speed (Training Mode)
- **370M**: ~45,000 tokens/sec on 8x A100
- **1.3B**: ~25,000 tokens/sec on 8x A100
- **2.7B**: ~12,000 tokens/sec on 8x A100 80GB

### Loss Curves
- Initial loss: ~7.0-8.0 (random initialization)
- After 1K steps: ~4.5-5.5
- After 5K steps: ~3.5-4.5
- After 10K steps: ~3.0-4.0

## Checkpoints

Checkpoints are saved to:
```
checkpoints/<model_name>/
├── checkpoint_step_1000.pt
├── checkpoint_step_2000.pt
├── ...
├── final_model/
│   ├── config.json
│   └── pytorch_model.bin
└── training_config.json
```

Each checkpoint contains:
- Model state dict
- Optimizer state
- Scheduler state
- Training step
- Current loss

## Next Steps After Training

1. **Evaluate perplexity**:
   ```bash
   python scripts/evaluate_paper.py --checkpoint checkpoints/<model_name>/final_model
   ```

2. **Benchmark speed**:
   ```bash
   python scripts/benchmark_paper.py --checkpoint checkpoints/<model_name>/final_model
   ```

3. **Test with eval mode**:
   ```bash
   MATMUL_FREE_MODE=eval python scripts/evaluate_paper.py --checkpoint checkpoints/<model_name>/final_model
   ```

## Important Notes

1. **Always use training mode for training** - It's 200x faster
2. **8 GPUs required** - Scripts are configured for 8-GPU DDP
3. **Dataset must be preprocessed** - Needs .ds files in SlimPajama format
4. **Checkpoints are large** - 370M: ~1.5GB, 1.3B: ~5GB, 2.7B: ~11GB per checkpoint
5. **Monitor GPU memory** - Use `nvidia-smi` to watch for OOM errors

## Getting Help

If training fails:
1. Run `python check_training_readiness.py`
2. Check the error message in the terminal
3. Look at the last few lines of output
4. Verify dataset and GPU memory
