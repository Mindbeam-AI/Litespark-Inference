#!/bin/bash
# Train 2.7B parameter model matching paper configuration
# Paper: "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)
# Model: MMfreeLM-2.7B

set -e

# Configuration
MODEL_NAME="2.7B_paper_match"
DATA_DIR="SlimPajama-6B-nanotron/train"
SAVE_DIR="checkpoints/${MODEL_NAME}"
NUM_GPUS=8

# Model architecture (matches paper Table 1)
HIDDEN_SIZE=2048
NUM_LAYERS=48  # Double the layers of 1.3B model
NUM_HEADS=16
SEQ_LENGTH=1024

# Training hyperparameters (matches paper Section 4.2)
BATCH_SIZE_PER_GPU=4   # Further reduced for largest model
GRAD_ACCUM_STEPS=8     # Effective batch = 8 * 4 * 8 = 256
TOTAL_STEPS=22888      # Match dataset: 6B tokens / (256 batch × 1024 seq) = ~22,888 steps
LEARNING_RATE=3e-4     # Lower for largest model
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

# Logging
LOG_INTERVAL=100
SAVE_INTERVAL=1000

echo "================================================================"
echo "Training MatMul-Free LM 2.7B - Paper Configuration"
echo "================================================================"
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NUM_GPUS}x A100"
echo "Effective Batch Size: $(($NUM_GPUS * $BATCH_SIZE_PER_GPU * $GRAD_ACCUM_STEPS))"
echo "Total Steps: ${TOTAL_STEPS}"
echo "Save Directory: ${SAVE_DIR}"
echo "WARNING: This model requires ~80GB VRAM per GPU"
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
