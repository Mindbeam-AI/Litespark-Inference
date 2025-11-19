#!/bin/bash
# Train 1.3B parameter model matching paper configuration
# Paper: "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)
# Model: MMfreeLM-1.3B

set -e

# Configuration
MODEL_NAME="1.3B_paper_match"
DATA_DIR="SlimPajama-6B-nanotron/train"
SAVE_DIR="checkpoints/${MODEL_NAME}"
NUM_GPUS=8

# Model architecture (matches paper Table 1)
HIDDEN_SIZE=2048
NUM_LAYERS=24
NUM_HEADS=16
SEQ_LENGTH=1024

# Training hyperparameters (matches paper Section 4.2)
BATCH_SIZE_PER_GPU=2   # Reduced for larger model
GRAD_ACCUM_STEPS=16     # Effective batch = 8 * 2 * 16 = 256
TOTAL_STEPS=114440      # Paper likely: ~30B tokens = 114,440 steps
LEARNING_RATE=4e-4     # Slightly lower for larger model
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0

# Logging
LOG_INTERVAL=100
SAVE_INTERVAL=1000

echo "================================================================"
echo "Training MatMul-Free LM 1.3B - Paper Configuration"
echo "================================================================"
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${NUM_GPUS}x A100"
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
