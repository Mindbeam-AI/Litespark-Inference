#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) Comprehensive Benchmark

Compares pre-quantization vs post-quantization:
1. Original FP16/FP32 model (baseline)
2. PTQ Ternary model (our kernels)
3. BitNet native ternary (reference, optional)

Tests (matching benchmark_comparison_short.py and benchmark_comparison_long.py):
- TTFT (Time-to-First-Token)
- Throughput (tokens/sec)
- Memory usage
- Perplexity on WikiText-2 (FULL evaluation - key accuracy metric)
- Prefill scaling (time vs prompt length)
- Batch scaling (throughput vs batch size)
- Generation scaling (throughput vs generation length)
"""

import torch
import torch.nn.functional as F
import time
import gc
import platform
import resource
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ptq_models import load_ptq_model, load_fp16_model

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Short benchmark settings
    num_runs: int = 5
    gen_tokens: int = 20
    warmup_iters: int = 2

    # Long benchmark settings
    prompt_lengths: List[int] = None
    batch_sizes: List[int] = None
    gen_lengths: List[int] = None
    bench_iters: int = 3

    # Perplexity settings
    ppl_max_samples: int = 100  # More samples for accuracy focus
    ppl_max_length: int = 512

    def __post_init__(self):
        if self.prompt_lengths is None:
            self.prompt_lengths = [32, 128, 256, 512]
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]
        if self.gen_lengths is None:
            self.gen_lengths = [20, 50, 100]


# ============================================================================
# Memory Tracking
# ============================================================================

def get_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    if platform.system() == 'Darwin':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    else:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ============================================================================
# Perplexity Evaluation (Comprehensive)
# ============================================================================

def evaluate_perplexity(
    model,
    tokenizer,
    max_samples: int = 100,
    max_length: int = 512,
    verbose: bool = True
) -> Tuple[float, Dict]:
    """
    Comprehensive perplexity evaluation on WikiText-2.

    Returns:
        perplexity: The perplexity score (lower is better)
        stats: Additional statistics (loss, tokens evaluated, etc.)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] datasets library not installed, skipping perplexity")
        return float('nan'), {}

    if verbose:
        print("  Loading WikiText-2 test set...")

    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception as e:
        print(f"  [!] Failed to load WikiText-2: {e}")
        return float('nan'), {}

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    sample_losses = []

    if verbose:
        print(f"  Evaluating on {min(max_samples, len(dataset))} samples...")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            text = sample['text']
            if not text.strip():
                continue

            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encodings['input_ids']

            if input_ids.shape[1] < 2:
                continue

            # Forward pass
            logits = model(input_ids)
            if hasattr(logits, 'logits'):
                logits = logits.logits

            # Cross-entropy loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            num_tokens = shift_labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            sample_losses.append(loss.item() / num_tokens)

            if verbose and (i + 1) % 20 == 0:
                current_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
                print(f"    Progress: {i+1}/{min(max_samples, len(dataset))}, Current PPL: {current_ppl:.2f}")

    if total_tokens == 0:
        return float('nan'), {}

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    stats = {
        'total_loss': total_loss,
        'total_tokens': total_tokens,
        'avg_loss': avg_loss,
        'num_samples': len(sample_losses),
        'loss_std': torch.tensor(sample_losses).std().item() if sample_losses else 0,
    }

    return perplexity, stats


# ============================================================================
# Short Benchmark
# ============================================================================

def benchmark_model_short(
    model,
    tokenizer,
    name: str,
    config: BenchmarkConfig
) -> Dict:
    """Short benchmark: basic TTFT and throughput."""
    model.eval()
    gc.collect()

    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away",
    ]

    mem_before = get_memory_mb()

    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(config.warmup_iters):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        with torch.no_grad():
            _ = model(input_ids)

    mem_after_warmup = get_memory_mb()

    # TTFT benchmark
    print(f"  Benchmarking TTFT...")
    ttft_times = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        for _ in range(config.num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            ttft_times.append((time.perf_counter() - start) * 1000)

    # Generation benchmark
    print(f"  Benchmarking generation ({config.gen_tokens} tokens)...")
    gen_times = []
    for _ in range(config.num_runs):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        generated = input_ids.clone()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(config.gen_tokens):
                logits = model(generated)
                if hasattr(logits, 'logits'):
                    logits = logits.logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        total_time = (time.perf_counter() - start) * 1000
        gen_times.append(total_time)

    # Sample generation
    input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(config.gen_tokens):
            logits = model(generated)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    sample_output = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {
        'name': name,
        'mean_ttft_ms': sum(ttft_times) / len(ttft_times),
        'std_ttft_ms': torch.tensor(ttft_times).std().item(),
        'mean_gen_time_ms': sum(gen_times) / len(gen_times),
        'tokens_per_sec': config.gen_tokens / (sum(gen_times) / len(gen_times) / 1000),
        'memory_mb': mem_after_warmup,
        'sample_output': sample_output,
    }


# ============================================================================
# Long Benchmark (Scaling Tests)
# ============================================================================

def benchmark_prefill_scaling(
    model,
    tokenizer,
    name: str,
    config: BenchmarkConfig
) -> Dict[str, Dict]:
    """Benchmark prefill time across different prompt lengths."""
    model.eval()
    results = {}

    # Create prompts of different lengths
    base_text = "The quick brown fox jumps over the lazy dog. " * 50

    for length in config.prompt_lengths:
        print(f"    Prompt length: {length} tokens")

        # Encode and truncate to exact length
        tokens = tokenizer.encode(base_text, add_special_tokens=False)[:length]
        input_ids = torch.tensor([tokens])

        # Warmup
        with torch.no_grad():
            for _ in range(config.warmup_iters):
                _ = model(input_ids)

        # Benchmark
        times = []
        for _ in range(config.bench_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            times.append((time.perf_counter() - start) * 1000)

        results[str(length)] = {
            'mean_ms': sum(times) / len(times),
            'std_ms': torch.tensor(times).std().item(),
            'min_ms': min(times),
        }

    return results


def benchmark_batch_scaling(
    model,
    tokenizer,
    name: str,
    config: BenchmarkConfig,
    prompt_length: int = 128
) -> Dict[str, Dict]:
    """Benchmark throughput across different batch sizes."""
    model.eval()
    results = {}

    # Create base prompt
    base_text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer.encode(base_text, add_special_tokens=False)[:prompt_length]

    for batch_size in config.batch_sizes:
        print(f"    Batch size: {batch_size}")

        # Create batched input
        input_ids = torch.tensor([tokens] * batch_size)

        # Warmup
        with torch.no_grad():
            for _ in range(config.warmup_iters):
                _ = model(input_ids)

        # Benchmark
        times = []
        for _ in range(config.bench_iters):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        total_tokens = batch_size * prompt_length
        tokens_per_sec = total_tokens / (mean_time / 1000)

        results[str(batch_size)] = {
            'mean_ms': mean_time,
            'std_ms': torch.tensor(times).std().item(),
            'min_ms': min(times),
            'tokens_per_sec': tokens_per_sec,
        }

    return results


def benchmark_generation_scaling(
    model,
    tokenizer,
    name: str,
    config: BenchmarkConfig,
    prompt_length: int = 32
) -> Dict[str, Dict]:
    """Benchmark generation throughput at different generation lengths."""
    model.eval()
    results = {}

    # Create base prompt
    base_text = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(base_text, return_tensors='pt')

    for gen_length in config.gen_lengths:
        print(f"    Generation length: {gen_length} tokens")

        # Warmup
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(5):
                logits = model(generated)
                if hasattr(logits, 'logits'):
                    logits = logits.logits
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

        # Benchmark
        times = []
        for _ in range(config.bench_iters):
            generated = input_ids.clone()

            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(gen_length):
                    logits = model(generated)
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        tokens_per_sec = gen_length / (mean_time / 1000)

        results[str(gen_length)] = {
            'mean_ms': mean_time,
            'std_ms': torch.tensor(times).std().item(),
            'tokens_per_sec': tokens_per_sec,
        }

    return results


# ============================================================================
# Graph Generation
# ============================================================================

def create_ptq_graphs(results: Dict, output_dir: Path):
    """Create comparison graphs for PTQ benchmark."""
    if not HAS_MATPLOTLIB:
        print("Skipping graphs (matplotlib not available)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Colors for different models
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    # 1. Accuracy comparison (Perplexity)
    if 'perplexity' in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results['perplexity'].keys())
        ppls = [results['perplexity'][m] for m in models]

        bars = ax.bar(models, ppls, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
        ax.set_title('Model Quality: Perplexity on WikiText-2', fontsize=14)

        for bar, ppl in zip(bars, ppls):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{ppl:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'perplexity_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'perplexity_comparison.png'}")

    # 2. Performance comparison (TTFT, Throughput, Memory)
    if 'short_benchmark' in results:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = list(results['short_benchmark'].keys())
        ttfts = [results['short_benchmark'][m]['mean_ttft_ms'] for m in models]
        tps = [results['short_benchmark'][m]['tokens_per_sec'] for m in models]
        mems = [results['short_benchmark'][m]['memory_mb'] for m in models]

        # TTFT
        bars1 = axes[0].bar(models, ttfts, color=colors[:len(models)], alpha=0.8)
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Time to First Token (TTFT)')
        axes[0].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars1, ttfts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        # Throughput
        bars2 = axes[1].bar(models, tps, color=colors[:len(models)], alpha=0.8)
        axes[1].set_ylabel('Tokens/sec')
        axes[1].set_title('Generation Throughput')
        axes[1].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars2, tps):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        # Memory
        bars3 = axes[2].bar(models, mems, color=colors[:len(models)], alpha=0.8)
        axes[2].set_ylabel('Memory (MB)')
        axes[2].set_title('Memory Usage')
        axes[2].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars3, mems):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'performance_comparison.png'}")

    # 3. Quality vs Speed tradeoff
    if 'perplexity' in results and 'short_benchmark' in results:
        fig, ax = plt.subplots(figsize=(10, 6))

        models = list(results['perplexity'].keys())

        for i, model in enumerate(models):
            if model in results['short_benchmark']:
                ppl = results['perplexity'][model]
                tps = results['short_benchmark'][model]['tokens_per_sec']
                ax.scatter(tps, ppl, s=200, c=colors[i], label=model, alpha=0.8, edgecolors='black')
                ax.annotate(model, (tps, ppl), xytext=(10, 10), textcoords='offset points', fontsize=10)

        ax.set_xlabel('Throughput (tokens/sec)', fontsize=12)
        ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
        ax.set_title('Quality vs Speed Tradeoff', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'quality_vs_speed.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'quality_vs_speed.png'}")


# ============================================================================
# Main Benchmark
# ============================================================================

def run_ptq_benchmark(
    model_name: str = 'microsoft/phi-2',
    quant_method: str = 'absmean',
    include_bitnet: bool = False,
    run_long: bool = True,
    output_dir: Optional[Path] = None,
    config: BenchmarkConfig = None
):
    """Run full PTQ benchmark comparison."""

    if config is None:
        config = BenchmarkConfig()

    print("=" * 70)
    print("PTQ Comprehensive Benchmark")
    print("Pre-Quantization vs Post-Quantization Comparison")
    print("=" * 70)

    arch = platform.machine()
    print(f"Platform: {arch}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Model: {model_name}")
    print(f"Quantization method: {quant_method}")
    print(f"Perplexity samples: {config.ppl_max_samples}")
    print()

    if output_dir is None:
        output_dir = Path(__file__).parent / 'benchmark_ptq_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'platform': arch,
        'model': model_name,
        'quant_method': quant_method,
        'config': {
            'ppl_max_samples': config.ppl_max_samples,
            'ppl_max_length': config.ppl_max_length,
        },
        'perplexity': {},
        'short_benchmark': {},
        'prefill_scaling': {},
        'batch_scaling': {},
        'generation_scaling': {},
    }

    models_to_test = []

    # ========================================================================
    # 1. PTQ Ternary Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. Loading PTQ Ternary Model...")
    print("=" * 70)

    ptq_model, tokenizer = load_ptq_model(model_name, method=quant_method)
    ptq_name = f"PTQ Ternary ({quant_method})"
    models_to_test.append((ptq_model, tokenizer, ptq_name))

    # ========================================================================
    # 2. Original FP16 Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. Loading Original FP16 Model...")
    print("=" * 70)

    fp16_model, fp16_tokenizer = load_fp16_model(model_name)
    fp16_name = "Original FP16"
    models_to_test.append((fp16_model, fp16_tokenizer, fp16_name))

    # ========================================================================
    # 3. BitNet (optional)
    # ========================================================================
    if include_bitnet:
        print("\n" + "=" * 70)
        print("3. Loading BitNet (native ternary)...")
        print("=" * 70)
        try:
            from ternary_models import load_ternary_model
            bitnet_model, bitnet_tokenizer = load_ternary_model('bitnet-2b')
            bitnet_name = "BitNet (native)"
            models_to_test.append((bitnet_model, bitnet_tokenizer, bitnet_name))
        except Exception as e:
            print(f"  [!] Failed to load BitNet: {e}")

    # ========================================================================
    # Run Benchmarks for Each Model
    # ========================================================================
    for model, tok, name in models_to_test:
        print("\n" + "=" * 70)
        print(f"Benchmarking: {name}")
        print("=" * 70)

        # Perplexity (most important for PTQ)
        print("\n  [Perplexity Evaluation]")
        ppl, ppl_stats = evaluate_perplexity(
            model, tok,
            max_samples=config.ppl_max_samples,
            max_length=config.ppl_max_length
        )
        all_results['perplexity'][name] = ppl
        print(f"  Final Perplexity: {ppl:.2f}")

        # Short benchmark
        print("\n  [Short Benchmark]")
        short_result = benchmark_model_short(model, tok, name, config)
        all_results['short_benchmark'][name] = short_result
        print(f"  TTFT: {short_result['mean_ttft_ms']:.1f}ms, Throughput: {short_result['tokens_per_sec']:.2f} tok/s")

        # Long benchmarks (if requested)
        if run_long:
            print("\n  [Prefill Scaling]")
            prefill = benchmark_prefill_scaling(model, tok, name, config)
            all_results['prefill_scaling'][name] = prefill

            print("\n  [Batch Scaling]")
            batch = benchmark_batch_scaling(model, tok, name, config)
            all_results['batch_scaling'][name] = batch

            print("\n  [Generation Scaling]")
            gen = benchmark_generation_scaling(model, tok, name, config)
            all_results['generation_scaling'][name] = gen

        # Clean up
        del model
        gc.collect()

    # ========================================================================
    # Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Perplexity comparison (key metric)
    print("\n--- Perplexity (Quality) ---")
    print(f"{'Model':<30} {'Perplexity':<15} {'Change':<15}")
    print("-" * 60)

    fp16_ppl = all_results['perplexity'].get(fp16_name, float('nan'))
    for name, ppl in all_results['perplexity'].items():
        change = ppl - fp16_ppl if fp16_ppl != float('nan') else 0
        change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        print(f"{name:<30} {ppl:<15.2f} {change_str:<15}")

    # Performance comparison
    print("\n--- Performance ---")
    print(f"{'Model':<30} {'TTFT (ms)':<12} {'Tok/s':<10} {'Memory (MB)':<14}")
    print("-" * 70)
    for name, result in all_results['short_benchmark'].items():
        print(f"{name:<30} {result['mean_ttft_ms']:<12.1f} {result['tokens_per_sec']:<10.2f} {result['memory_mb']:<14.1f}")

    # Speedups
    if ptq_name in all_results['short_benchmark'] and fp16_name in all_results['short_benchmark']:
        ptq = all_results['short_benchmark'][ptq_name]
        fp16 = all_results['short_benchmark'][fp16_name]

        print("\n--- PTQ vs FP16 Summary ---")
        print(f"  TTFT Speedup: {fp16['mean_ttft_ms'] / ptq['mean_ttft_ms']:.2f}x")
        print(f"  Throughput Speedup: {ptq['tokens_per_sec'] / fp16['tokens_per_sec']:.2f}x")
        print(f"  Memory Reduction: {fp16['memory_mb'] / ptq['memory_mb']:.2f}x")
        print(f"  Perplexity Degradation: {all_results['perplexity'][fp16_name]:.2f} -> {all_results['perplexity'][ptq_name]:.2f}")

    # Sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (Quality Check)")
    print("=" * 70)
    for name, result in all_results['short_benchmark'].items():
        print(f"\n{name}:")
        print(f"  {result['sample_output'][:200]}...")

    # ========================================================================
    # Save Results and Generate Graphs
    # ========================================================================
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    # Save JSON
    results_file = output_dir / 'ptq_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {results_file}")

    # Create graphs
    create_ptq_graphs(all_results, output_dir)

    print("\n" + "=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)

    return all_results


def compare_all_methods(
    model_name: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    output_dir: Optional[Path] = None,
    config: BenchmarkConfig = None
):
    """Compare all quantization methods on the same model."""

    if config is None:
        config = BenchmarkConfig()

    if output_dir is None:
        output_dir = Path(__file__).parent / 'benchmark_ptq_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ['absmean', 'percentile', 'optimal', 'gptq', 'smoothquant', 'awq', 'pt2']

    print("=" * 70)
    print("PTQ Method Comparison")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Methods: {', '.join(methods)}")
    print()

    # First load FP16 baseline (only once)
    print("Loading FP16 baseline...")
    fp16_model, tokenizer = load_fp16_model(model_name)

    print("\nEvaluating FP16 perplexity...")
    fp16_ppl, _ = evaluate_perplexity(fp16_model, tokenizer, max_samples=config.ppl_max_samples)
    print(f"  FP16 Perplexity: {fp16_ppl:.2f}")

    del fp16_model
    gc.collect()

    # Test each quantization method
    results = {
        'model': model_name,
        'fp16_perplexity': fp16_ppl,
        'methods': {}
    }

    for method in methods:
        print(f"\n{'='*70}")
        print(f"Testing method: {method.upper()}")
        print("=" * 70)

        try:
            ptq_model, tokenizer = load_ptq_model(model_name, method=method)

            print("\n  Evaluating perplexity...")
            ppl, ppl_stats = evaluate_perplexity(ptq_model, tokenizer, max_samples=config.ppl_max_samples)

            print(f"\n  Evaluating performance...")
            short_result = benchmark_model_short(ptq_model, tokenizer, method, config)

            results['methods'][method] = {
                'perplexity': ppl,
                'ppl_stats': ppl_stats,
                'ttft_ms': short_result['mean_ttft_ms'],
                'tokens_per_sec': short_result['tokens_per_sec'],
                'memory_mb': short_result['memory_mb'],
                'sample_output': short_result['sample_output'][:100],
            }

            print(f"\n  Results for {method}:")
            print(f"    Perplexity: {ppl:.2f} (FP16: {fp16_ppl:.2f}, degradation: +{ppl - fp16_ppl:.2f})")
            print(f"    TTFT: {short_result['mean_ttft_ms']:.1f}ms")
            print(f"    Throughput: {short_result['tokens_per_sec']:.2f} tok/s")

            del ptq_model
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] Method {method} failed: {e}")
            results['methods'][method] = {'error': str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\nBaseline: FP16 Perplexity = {fp16_ppl:.2f}")
    print()
    print(f"{'Method':<15} {'Perplexity':<12} {'Degradation':<12} {'TTFT (ms)':<12} {'Tok/s':<10}")
    print("-" * 70)

    for method in methods:
        if method in results['methods'] and 'perplexity' in results['methods'][method]:
            r = results['methods'][method]
            ppl = r['perplexity']
            deg = ppl - fp16_ppl
            print(f"{method:<15} {ppl:<12.2f} +{deg:<11.2f} {r['ttft_ms']:<12.1f} {r['tokens_per_sec']:<10.2f}")
        else:
            print(f"{method:<15} {'ERROR':<12}")

    # Save results
    results_file = output_dir / 'method_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PTQ Comprehensive Benchmark')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model to benchmark')
    parser.add_argument('--method', type=str, default='absmean',
                        choices=['absmean', 'percentile', 'optimal', 'gptq', 'smoothquant', 'awq', 'pt2'],
                        help='Quantization method')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all quantization methods')
    parser.add_argument('--include-bitnet', action='store_true',
                        help='Include BitNet comparison')
    parser.add_argument('--short-only', action='store_true',
                        help='Skip long scaling benchmarks')
    parser.add_argument('--ppl-samples', type=int, default=100,
                        help='Number of WikiText-2 samples for perplexity')
    args = parser.parse_args()

    config = BenchmarkConfig(ppl_max_samples=args.ppl_samples)

    if args.compare_all:
        compare_all_methods(model_name=args.model, config=config)
    else:
        run_ptq_benchmark(
            model_name=args.model,
            quant_method=args.method,
            include_bitnet=args.include_bitnet,
            run_long=not args.short_only,
            config=config
        )
