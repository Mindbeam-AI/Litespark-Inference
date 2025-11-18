#!/bin/bash
# Train 1.3B parameter model for 8x A100 40GB GPUs
# Optimized memory configuration

set -e

# Configuration
MODEL_NAME="1.3B_40GB_optimized"
DATA_DIR="SlimPajama-6B-nanotron/train"
SAVE_DIR="checkpoints/${MODEL_NAME}"
NUM_GPUS=8

# Model architecture (matches paper's 1.3B model)
HIDDEN_SIZE=2048
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=256   # Further reduced for 40GB GPUs (paper uses 1024, will note in analysis)

# Training hyperparameters - extremely conservative for 40GB GPUs
BATCH_SIZE_PER_GPU=1   # Minimum to fit in memory
GRAD_ACCUM_STEPS=32    # Effective batch = 8 * 1 * 32 = 256 ✓
TOTAL_STEPS=10000
LEARNING_RATE=4e-4     # Slightly lower for larger model
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

# Logging
LOG_INTERVAL=100
SAVE_INTERVAL=1000

echo "================================================================"
echo "Training MatMul-Free LM 1.3B - 40GB GPU Configuration"
echo "================================================================"
echo "Model: ${MODEL_NAME}"
echo "Parameters: ~1.3B"
echo "GPUs: ${NUM_GPUS}x A100 40GB"
echo "Sequence Length: ${SEQ_LENGTH}"
echo "Batch per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Gradient Accumulation: ${GRAD_ACCUM_STEPS}"
echo "Effective Batch Size: $(($NUM_GPUS * $BATCH_SIZE_PER_GPU * $GRAD_ACCUM_STEPS))"
echo "Total Steps: ${TOTAL_STEPS}"
echo "Save Directory: ${SAVE_DIR}"
echo "================================================================"
echo ""

# Launch DDP training
python train_ddp.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --hidden_size ${HIDDEN_SIZE} \
  --num_layers ${NUM_LAYERS} \
  --num_heads ${NUM_HEADS} \
  --seq_length ${SEQ_LENGTH} \
  --batch_size ${BATCH_SIZE_PER_GPU} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --total_steps ${TOTAL_STEPS} \
  --learning_rate ${LEARNING_RATE} \
  --weight_decay ${WEIGHT_DECAY} \
  --max_grad_norm ${MAX_GRAD_NORM} \
  --log_interval ${LOG_INTERVAL} \
  --save_interval ${SAVE_INTERVAL}

echo ""
echo "================================================================"
echo "Training completed! Checkpoints saved to: ${SAVE_DIR}"
echo "================================================================"
