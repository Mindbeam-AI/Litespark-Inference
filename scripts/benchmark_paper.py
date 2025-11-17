#!/usr/bin/env python3
"""
Benchmarking script matching the paper's methodology
Paper: "Scalable MatMul-free Language Modeling" (arXiv:2406.02528)

Benchmarks:
- Memory efficiency across sequence lengths (512, 1024, 2048, 4096)
- Throughput (tokens/sec)
- Latency
- Comparison with baseline models (Pythia)
"""

import argparse
import json
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


def load_matmul_free_model(checkpoint_path, device='cuda'):
    """Load MatMul-Free model from checkpoint"""
    print(f"Loading MatMul-Free model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', None)
    if config is None:
        raise ValueError("Checkpoint does not contain config")

    model = HGRNBitForCausalLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_baseline_model(model_name, device='cuda'):
    """Load baseline model (e.g., Pythia)"""
    print(f"Loading baseline model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model


def benchmark_memory(model, sequence_lengths, batch_size=4, device='cuda'):
    """
    Benchmark memory usage across different sequence lengths
    Following paper's Table 2 methodology
    """
    results = {}

    for seq_len in sequence_lengths:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Create dummy input
        input_ids = torch.randint(0, 50000, (batch_size, seq_len), device=device)

        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)

        # Measure memory
        memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

        results[seq_len] = {
            'memory_gb': memory_allocated,
            'memory_per_token_mb': (memory_allocated * 1024) / (batch_size * seq_len)
        }

        print(f"Seq Length {seq_len}: {memory_allocated:.2f} GB")

    return results


def benchmark_throughput(model, seq_length=1024, batch_size=4, num_iterations=100, warmup=10, device='cuda'):
    """
    Benchmark throughput (tokens/sec)
    Following paper's benchmarking methodology
    """
    print(f"\nBenchmarking throughput (seq_len={seq_length}, batch={batch_size})...")

    # Create dummy input
    input_ids = torch.randint(0, 50000, (batch_size, seq_length), device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_ids)

    torch.cuda.synchronize()
    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    total_tokens = batch_size * seq_length * num_iterations
    throughput = total_tokens / total_time
    latency_per_batch = (total_time / num_iterations) * 1000  # ms

    return {
        'throughput_tokens_per_sec': throughput,
        'latency_ms_per_batch': latency_per_batch,
        'total_time_sec': total_time,
        'iterations': num_iterations
    }


def benchmark_comparison(matmul_free_model, baseline_model, seq_length=1024, batch_size=4, device='cuda'):
    """
    Compare MatMul-Free model with baseline
    Following paper's comparison methodology
    """
    print(f"\n{'='*80}")
    print(f"Comparing MatMul-Free vs Baseline")
    print(f"{'='*80}")

    # Benchmark both models
    print("\n--- MatMul-Free Model ---")
    mf_throughput = benchmark_throughput(matmul_free_model, seq_length, batch_size, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    input_ids = torch.randint(0, 50000, (batch_size, seq_length), device=device)
    with torch.no_grad():
        _ = matmul_free_model(input_ids)
    mf_memory = torch.cuda.max_memory_allocated(device) / 1024**3

    print("\n--- Baseline Model ---")
    baseline_throughput = benchmark_throughput(baseline_model, seq_length, batch_size, device=device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        _ = baseline_model(input_ids)
    baseline_memory = torch.cuda.max_memory_allocated(device) / 1024**3

    # Calculate improvements
    memory_reduction = ((baseline_memory - mf_memory) / baseline_memory) * 100
    throughput_change = ((mf_throughput['throughput_tokens_per_sec'] - baseline_throughput['throughput_tokens_per_sec']) / baseline_throughput['throughput_tokens_per_sec']) * 100

    return {
        'matmul_free': {
            'memory_gb': mf_memory,
            'throughput': mf_throughput['throughput_tokens_per_sec'],
            'latency_ms': mf_throughput['latency_ms_per_batch']
        },
        'baseline': {
            'memory_gb': baseline_memory,
            'throughput': baseline_throughput['throughput_tokens_per_sec'],
            'latency_ms': baseline_throughput['latency_ms_per_batch']
        },
        'improvements': {
            'memory_reduction_percent': memory_reduction,
            'throughput_change_percent': throughput_change
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark MatMul-Free LM following paper methodology')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to MatMul-Free model checkpoint')
    parser.add_argument('--baseline', type=str, default='EleutherAI/pythia-410m-deduped',
                        help='Baseline model for comparison')
    parser.add_argument('--seq_lengths', nargs='+', type=int, default=[512, 1024, 2048, 4096],
                        help='Sequence lengths to benchmark')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations for throughput')
    parser.add_argument('--output', type=str, default='benchmark_results.json', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip baseline comparison')

    args = parser.parse_args()

    print("="*80)
    print("MatMul-Free LM Benchmarking - Paper Methodology")
    print("="*80)
    print(f"MatMul-Free Checkpoint: {args.checkpoint}")
    print(f"Baseline Model: {args.baseline}")
    print(f"Sequence Lengths: {args.seq_lengths}")
    print(f"Batch Size: {args.batch_size}")
    print("="*80)
    print()

    # Load models
    mf_model, config = load_matmul_free_model(args.checkpoint, args.device)
    print(f"MatMul-Free Model: {sum(p.numel() for p in mf_model.parameters()) / 1e6:.1f}M parameters")

    results = {
        'checkpoint': args.checkpoint,
        'model_config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_heads,
        }
    }

    # Benchmark memory across sequence lengths
    print(f"\n{'='*80}")
    print("Benchmarking Memory Efficiency")
    print(f"{'='*80}")
    memory_results = benchmark_memory(mf_model, args.seq_lengths, args.batch_size, args.device)
    results['memory_by_seq_length'] = memory_results

    # Benchmark throughput
    print(f"\n{'='*80}")
    print("Benchmarking Throughput")
    print(f"{'='*80}")
    throughput_results = benchmark_throughput(
        mf_model,
        seq_length=1024,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        device=args.device
    )
    results['throughput'] = throughput_results

    print(f"\nThroughput: {throughput_results['throughput_tokens_per_sec']:,.0f} tokens/sec")
    print(f"Latency: {throughput_results['latency_ms_per_batch']:.2f} ms/batch")

    # Compare with baseline
    if not args.skip_baseline:
        baseline_model = load_baseline_model(args.baseline, args.device)
        print(f"Baseline Model: {sum(p.numel() for p in baseline_model.parameters()) / 1e6:.1f}M parameters")

        comparison = benchmark_comparison(
            mf_model, baseline_model,
            seq_length=1024,
            batch_size=args.batch_size,
            device=args.device
        )
        results['comparison'] = comparison

        print(f"\n{'='*80}")
        print("Comparison Summary")
        print(f"{'='*80}")
        print(f"Memory Reduction: {comparison['improvements']['memory_reduction_percent']:+.1f}%")
        print(f"Throughput Change: {comparison['improvements']['throughput_change_percent']:+.1f}%")

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to: {args.output}")
    print(f"{'='*80}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nBenchmarking complete!")


if __name__ == '__main__':
    main()
