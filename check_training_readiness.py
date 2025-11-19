#!/usr/bin/env python3
"""
Check Training Readiness
Verifies all requirements for running the 3 model training scripts
"""

import os
import sys
from pathlib import Path
import torch

def check_item(name, status, details=""):
    symbol = "✓" if status else "✗"
    print(f"{symbol} {name}")
    if details:
        print(f"  {details}")
    return status

print("=" * 80)
print("TRAINING READINESS CHECK")
print("=" * 80)
print()

all_checks = []

# 1. Check GPU availability
print("[1] GPU Check")
print("-" * 80)
has_gpu = torch.cuda.is_available()
all_checks.append(check_item("CUDA available", has_gpu))
if has_gpu:
    num_gpus = torch.cuda.device_count()
    all_checks.append(check_item(f"Number of GPUs: {num_gpus}", num_gpus >= 8,
                                  f"Found {num_gpus}, need 8 for paper configs"))
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
print()

# 2. Check dataset
print("[2] Dataset Check")
print("-" * 80)
data_dir = Path("SlimPajama-6B-nanotron/train")
has_dataset = data_dir.exists()
all_checks.append(check_item(f"Dataset directory exists: {data_dir}", has_dataset))
if has_dataset:
    ds_files = list(data_dir.glob("*.ds"))
    all_checks.append(check_item(f"Found {len(ds_files)} .ds files", len(ds_files) > 0))
    if ds_files:
        total_size = sum(f.stat().st_size for f in ds_files) / 1e9
        print(f"  Total dataset size: {total_size:.2f} GB")
else:
    print("  ⚠️  Dataset not found. Expected location: SlimPajama-6B-nanotron/train")
print()

# 3. Check training scripts
print("[3] Training Scripts Check")
print("-" * 80)
scripts = [
    "scripts/train_370M_paper.sh",
    "scripts/train_1.3B_paper.sh",
    "scripts/train_2.7B_paper.sh",
]
for script in scripts:
    script_path = Path(script)
    exists = script_path.exists()
    executable = os.access(script_path, os.X_OK) if exists else False
    all_checks.append(check_item(f"{script}", exists and executable,
                                  "Not executable" if exists and not executable else ""))
print()

# 4. Check training code
print("[4] Training Code Check")
print("-" * 80)
train_script = Path("train_ddp.py")
all_checks.append(check_item(f"Main training script: {train_script}", train_script.exists()))

# Check model imports
try:
    from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
    from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
    all_checks.append(check_item("Model imports successful", True))
except ImportError as e:
    all_checks.append(check_item("Model imports successful", False, str(e)))

# Check ops imports
try:
    from src.ops.fusedbitnet import FusedBitLinear
    from src.ops.matmul_free_linear import matmul_free_linear
    all_checks.append(check_item("Ops imports successful", True))
except ImportError as e:
    all_checks.append(check_item("Ops imports successful", False, str(e)))
print()

# 5. Check dependencies
print("[5] Dependencies Check")
print("-" * 80)
try:
    import triton
    all_checks.append(check_item(f"Triton installed: {triton.__version__}", True))
except ImportError:
    all_checks.append(check_item("Triton installed", False))

try:
    import torch.distributed as dist
    all_checks.append(check_item("PyTorch DDP available", True))
except ImportError:
    all_checks.append(check_item("PyTorch DDP available", False))
print()

# 6. Check hybrid mode
print("[6] Hybrid Mode Check")
print("-" * 80)
mode = os.environ.get('MATMUL_FREE_MODE', 'train')
all_checks.append(check_item(f"Current mode: {mode}", True,
                              "Training mode (F.linear - fast)" if mode == 'train' else "Eval mode (MatMul-Free)"))
if mode != 'train':
    print("  ⚠️  Warning: Training with eval mode will be very slow!")
    print("  Recommendation: Unset MATMUL_FREE_MODE for training")
print()

# 7. Model size estimates
print("[7] Model Size Estimates")
print("-" * 80)
model_configs = {
    "370M": {"hidden": 1024, "layers": 24, "heads": 8},
    "1.3B": {"hidden": 2048, "layers": 24, "heads": 16},
    "2.7B": {"hidden": 2048, "layers": 48, "heads": 16},
}

for name, config in model_configs.items():
    # Rough estimate: embed + layers * (attn + mlp)
    embed_params = config["hidden"] * 50000  # vocab embedding
    layer_params = config["layers"] * (
        4 * config["hidden"] * config["hidden"] +  # attention
        8 * config["hidden"] * config["hidden"]    # MLP
    )
    total_params = (embed_params + layer_params) / 1e6
    print(f"  {name}: ~{total_params:.0f}M parameters")
print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
if all(all_checks):
    print("✅ ALL CHECKS PASSED - Ready to train!")
    print()
    print("To start training:")
    print("  370M model:  bash scripts/train_370M_paper.sh")
    print("  1.3B model:  bash scripts/train_1.3B_paper.sh")
    print("  2.7B model:  bash scripts/train_2.7B_paper.sh (requires 80GB GPUs)")
else:
    print("⚠️  SOME CHECKS FAILED - Fix issues before training")
    failed = sum(1 for c in all_checks if not c)
    print(f"   {failed} checks failed out of {len(all_checks)}")
print("=" * 80)
