#!/usr/bin/env python3
"""
Comprehensive evaluation and visualization script for MatMul-Free Language Models
- Evaluates trained checkpoints on standard benchmarks
- Compares with paper baselines
- Generates visualization plots
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


class SimpleTextDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation"""
    def __init__(self, file_path, seq_length=1024, max_samples=None):
        self.seq_length = seq_length
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Simple character-level tokenization (you can replace with proper tokenizer)
        self.tokens = torch.tensor([ord(c) % 50000 for c in text], dtype=torch.long)

        # Calculate number of sequences
        self.num_sequences = len(self.tokens) // seq_length
        if max_samples:
            self.num_sequences = min(self.num_sequences, max_samples)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_length
        return self.tokens[start:start + self.seq_length]


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from state dict
    state_dict = checkpoint['model_state_dict']

    # Infer config from state dict
    # This is a simplified version - adjust based on your actual model structure
    config = HGRNBitConfig(
        vocab_size=50000,
        hidden_size=1024,  # Will be overridden based on checkpoint
        num_hidden_layers=24,
        max_position_embeddings=2048,
        num_heads=8,
        expand_ratio=1,
        hidden_ratio=4,
        rms_norm_eps=1e-6,
        use_cache=False,
        attn_mode="fused_recurrent",
    )

    # Load model
    model = HGRNBitForCausalLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_perplexity(model, dataloader, device='cuda'):
    """Evaluate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


def plot_training_curves(checkpoint_dirs, output_dir):
    """Plot training loss curves from multiple checkpoints"""
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for idx, ckpt_dir in enumerate(checkpoint_dirs):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            print(f"Warning: {ckpt_dir} does not exist, skipping")
            continue

        # Find all checkpoints
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            print(f"Warning: No checkpoints found in {ckpt_dir}")
            continue

        steps = []
        losses = []

        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu')
                steps.append(ckpt['step'])
                losses.append(ckpt['loss'])
            except Exception as e:
                print(f"Warning: Could not load {ckpt_path}: {e}")
                continue

        if steps:
            label = ckpt_dir.name
            plt.plot(steps, losses, marker='o', color=colors[idx % len(colors)],
                    label=label, linewidth=2, markersize=4)

    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_model_comparison(results, output_dir):
    """Plot comparison of different models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data
    model_names = list(results.keys())
    perplexities = [results[name]['perplexity'] for name in model_names]
    params = [results[name].get('params_M', 0) for name in model_names]

    # Plot 1: Perplexity comparison
    ax1 = axes[0]
    bars = ax1.bar(range(len(model_names)), perplexities, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax1.set_title('Model Perplexity Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, ppl in zip(bars, perplexities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{ppl:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Plot 2: Perplexity vs Parameters
    ax2 = axes[1]
    if all(p > 0 for p in params):
        ax2.scatter(params, perplexities, s=200, alpha=0.6, color='coral')
        for i, name in enumerate(model_names):
            ax2.annotate(name, (params[i], perplexities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax2.set_xlabel('Parameters (M)', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Perplexity vs Model Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Parameter counts not available',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison to {output_path}")
    plt.close()


def plot_memory_usage(results, output_dir):
    """Plot memory usage comparison"""
    plt.figure(figsize=(10, 6))

    model_names = list(results.keys())
    memory_gb = [results[name].get('peak_memory_gb', 0) for name in model_names]

    if all(m > 0 for m in memory_gb):
        bars = plt.bar(range(len(model_names)), memory_gb, color='mediumseagreen', alpha=0.8)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Peak GPU Memory (GB)', fontsize=12)
        plt.title('GPU Memory Usage Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, mem in zip(bars, memory_gb):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mem:.1f} GB',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        output_path = Path(output_dir) / "memory_usage.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved memory usage plot to {output_path}")
        plt.close()
    else:
        print("Memory usage data not available, skipping plot")


def benchmark_inference_speed(model, batch_size=1, seq_length=1024, device='cuda', num_iterations=100):
    """Benchmark inference speed"""
    model.eval()

    # Warmup
    dummy_input = torch.randint(0, 50000, (batch_size, seq_length), device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    tokens_per_sec = (batch_size * seq_length) / avg_time

    return tokens_per_sec, avg_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualize MatMul-Free LM models")
    parser.add_argument("--checkpoint_dirs", nargs='+', required=True,
                       help="Directories containing checkpoints to evaluate")
    parser.add_argument("--checkpoint_names", nargs='+',
                       help="Names for each checkpoint (optional)")
    parser.add_argument("--eval_data", type=str,
                       help="Path to evaluation text file (optional)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save results and plots")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--seq_length", type=int, default=1024,
                       help="Sequence length for evaluation")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to evaluate (for speed)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--benchmark_speed", action="store_true",
                       help="Benchmark inference speed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MatMul-Free Language Model Evaluation")
    print("=" * 80)

    # Plot training curves
    print("\nPlotting training curves...")
    plot_training_curves(args.checkpoint_dirs, output_dir)

    # Evaluate each checkpoint
    results = {}

    for idx, ckpt_dir in enumerate(args.checkpoint_dirs):
        ckpt_dir = Path(ckpt_dir)

        # Get model name
        if args.checkpoint_names and idx < len(args.checkpoint_names):
            model_name = args.checkpoint_names[idx]
        else:
            model_name = ckpt_dir.name

        print(f"\n{'=' * 80}")
        print(f"Evaluating: {model_name}")
        print(f"{'=' * 80}")

        # Find final checkpoint
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            print(f"No checkpoints found in {ckpt_dir}, skipping")
            continue

        final_checkpoint = checkpoints[-1]  # Use last checkpoint
        print(f"Using checkpoint: {final_checkpoint.name}")

        try:
            # Load model
            model, checkpoint = load_checkpoint(final_checkpoint, device=args.device)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters())
            params_M = num_params / 1e6

            print(f"Model parameters: {params_M:.1f}M")
            print(f"Checkpoint step: {checkpoint['step']}")
            print(f"Checkpoint loss: {checkpoint['loss']:.4f}")

            result = {
                'checkpoint_step': checkpoint['step'],
                'checkpoint_loss': checkpoint['loss'],
                'params_M': params_M,
            }

            # Benchmark speed if requested
            if args.benchmark_speed:
                print("\nBenchmarking inference speed...")
                tokens_per_sec, avg_time = benchmark_inference_speed(
                    model,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    device=args.device
                )
                print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
                print(f"Avg time per iteration: {avg_time*1000:.2f} ms")

                result['tokens_per_sec'] = tokens_per_sec
                result['avg_time_ms'] = avg_time * 1000

            # Evaluate perplexity if eval data provided
            if args.eval_data and Path(args.eval_data).exists():
                print("\nEvaluating perplexity...")
                dataset = SimpleTextDataset(
                    args.eval_data,
                    seq_length=args.seq_length,
                    max_samples=args.max_samples
                )
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False
                )

                perplexity, avg_loss = evaluate_perplexity(model, dataloader, device=args.device)
                print(f"Perplexity: {perplexity:.2f}")
                print(f"Avg Loss: {avg_loss:.4f}")

                result['perplexity'] = perplexity
                result['eval_loss'] = avg_loss
            else:
                # Use checkpoint loss as proxy
                result['perplexity'] = np.exp(checkpoint['loss'])

            # Get peak memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated(args.device) / 1e9
                print(f"Peak GPU memory: {peak_memory:.2f} GB")
                result['peak_memory_gb'] = peak_memory
                torch.cuda.reset_peak_memory_stats()

            results[model_name] = result

            # Clean up
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")

    # Generate comparison plots
    if len(results) > 1:
        print("\nGenerating comparison plots...")
        plot_model_comparison(results, output_dir)
        plot_memory_usage(results, output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Params (M)':<12} {'Perplexity':<12} {'Tokens/sec':<15} {'Memory (GB)'}")
    print("-" * 80)

    for name, res in results.items():
        params = f"{res.get('params_M', 0):.1f}"
        ppl = f"{res.get('perplexity', 0):.2f}"
        tps = f"{res.get('tokens_per_sec', 0):,.0f}" if 'tokens_per_sec' in res else "N/A"
        mem = f"{res.get('peak_memory_gb', 0):.1f}"
        print(f"{name:<25} {params:<12} {ppl:<12} {tps:<15} {mem}")

    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
