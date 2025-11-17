#!/usr/bin/env python3
"""
Improved DDP training script for MatMul-Free Language Model
Features:
- Proper DDP multi-GPU training
- Gradient accumulation
- Mixed precision training (optional)
- Better checkpointing
- Support for SlimPajama dataset
- Gradient clipping
- CosineAnnealing scheduler
- Comprehensive logging
"""

import argparse
import os
import time
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


class PajamaDataset(Dataset):
    """SlimPajama dataset for training"""
    def __init__(self, data_dir, seq_length=1024):
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.files = sorted(self.data_dir.glob("*.ds"))

        if not self.files:
            raise ValueError(f"No .ds files found in {data_dir}")

        # Calculate total tokens
        self.total_tokens = sum(os.path.getsize(f) // 2 for f in self.files)  # 2 bytes per token
        self.total_sequences = self.total_tokens // seq_length

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Calculate which file and offset
        target_token = idx * self.seq_length
        cumulative = 0

        for file_path in self.files:
            file_tokens = os.path.getsize(file_path) // 2
            if cumulative + file_tokens > target_token:
                # Found the right file
                offset_in_file = target_token - cumulative
                with open(file_path, 'rb') as f:
                    f.seek(offset_in_file * 2)
                    data = f.read(self.seq_length * 2)
                    # Clone to make writable tensor (fixes PyTorch warning)
                    tokens = torch.frombuffer(data, dtype=torch.int16).long().clone()
                    # Pad if necessary
                    if len(tokens) < self.seq_length:
                        tokens = torch.cat([tokens, torch.zeros(self.seq_length - len(tokens), dtype=torch.long)])
                    return tokens
            cumulative += file_tokens

        # Fallback to zeros if something went wrong
        return torch.zeros(self.seq_length, dtype=torch.long)


def setup_ddp(rank, world_size):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP process group"""
    dist.destroy_process_group()


def get_model(config, rank):
    """Create and configure model"""
    model = HGRNBitForCausalLM(config).to(rank)

    # Enable gradient checkpointing if needed
    if hasattr(model.config, 'gradient_checkpointing') and model.config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_dir, rank):
    """Save training checkpoint (only on rank 0)"""
    if rank != 0:
        return

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    save_path = Path(save_dir) / f"checkpoint_epoch{epoch}_step{step}.pt"
    torch.save(checkpoint, save_path)
    print(f"[Rank {rank}] Saved checkpoint to {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, rank, world_size,
                     epoch, args, start_step=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    log_interval = args.log_interval

    for step, batch in enumerate(train_loader, start=start_step):
        # Move to device
        input_ids = batch.to(rank)

        # Forward pass
        if args.use_amp:
            with autocast():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss / args.gradient_accumulation_steps
        else:
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss / args.gradient_accumulation_steps

        # Backward pass
        if args.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            if args.max_grad_norm > 0:
                if args.use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Optimizer step
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # Logging
        total_loss += loss.item() * args.gradient_accumulation_steps

        if rank == 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch} | Step {step+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            total_loss = 0

        # Save checkpoint periodically
        if rank == 0 and (step + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, step+1,
                          loss.item(), args.save_dir, rank)


def train(rank, world_size, args):
    """Main training function for each process"""
    setup_ddp(rank, world_size)

    if rank == 0:
        print(f"Starting training on {world_size} GPUs")
        print(f"Config: {args}")

    # Create model
    config = HGRNBitConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        max_position_embeddings=args.max_seq_length,
        num_heads=args.num_heads,
        expand_ratio=args.expand_ratio,
        hidden_ratio=args.hidden_ratio,
        rms_norm_eps=1e-6,
        use_cache=False,
        attn_mode=args.attn_mode,
    )

    if rank == 0:
        num_params = sum(p.numel() for p in HGRNBitForCausalLM(config).parameters())
        print(f"Model parameters: {num_params / 1e6:.1f}M")

    model = get_model(config, rank)
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    # Create dataset and dataloader
    if rank == 0:
        print(f"Loading dataset from {args.data_dir}")

    train_dataset = PajamaDataset(args.data_dir, seq_length=args.seq_length)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    total_steps = args.total_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1
    )

    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None

    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume_from}")
        start_epoch, start_step, _ = load_checkpoint(
            args.resume_from, ddp_model, optimizer, scheduler
        )

    # Training loop
    if rank == 0:
        print("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train_one_epoch(
            ddp_model, train_loader, optimizer, scheduler, scaler,
            rank, world_size, epoch, args, start_step if epoch == start_epoch else 0
        )

        # Save end-of-epoch checkpoint
        if rank == 0:
            save_checkpoint(ddp_model, optimizer, scheduler, epoch+1, 0,
                          0.0, args.save_dir, rank)
            print(f"Epoch {epoch} completed")

    # Save final model
    if rank == 0:
        final_path = Path(args.save_dir) / "final_model"
        ddp_model.module.save_pretrained(final_path)
        print(f"Final model saved to {final_path}")

    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description="DDP training for MatMul-Free LM")

    # Model config
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--expand_ratio", type=int, default=1)
    parser.add_argument("--hidden_ratio", type=int, default=4)
    parser.add_argument("--attn_mode", type=str, default="fused_recurrent")

    # Training config
    parser.add_argument("--data_dir", type=str, required=True, help="Path to SlimPajama .ds files")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save checkpoints")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # System config
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(args.save_dir) / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Get number of GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise ValueError("No GPUs available!")

    print(f"Training on {world_size} GPUs")
    print(f"Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")

    # Launch training
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
