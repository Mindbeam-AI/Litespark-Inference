# Old Training Scripts

This directory contains legacy training scripts that have been superseded by newer implementations.

## Files

### `train.py` (Nov 7)
- **Status**: OBSOLETE - Replaced by `train_ddp.py` (Nov 19)
- **Description**: Original basic training script using WikiText dataset
- **Why replaced**:
  - Used basic DDP setup without gradient accumulation
  - Limited to WikiText dataset
  - Lacked production features (proper checkpointing, mixed precision, etc.)
  - New `train_ddp.py` provides comprehensive DDP training with SlimPajama support

**Use the root-level `train_ddp.py` for all new training runs.**
