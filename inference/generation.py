#!/usr/bin/env python3
"""
Text Generation Utilities with TTFT Measurement

Key metrics:
- Time-to-First-Token (TTFT): Latency from prompt to first token
- Tokens per second (TPS): Throughput during generation
- End-to-end latency: Total time for complete generation
"""

import torch
import time
from typing import Optional, Tuple, List, Dict, Callable, Union
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    """Metrics from a single generation run."""
    ttft_ms: float  # Time to first token in milliseconds
    total_time_ms: float  # Total generation time
    num_tokens: int  # Number of generated tokens
    tokens_per_sec: float  # Tokens per second (excluding first token)
    prompt_length: int  # Input prompt length in tokens
    output_text: str  # Generated text


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    do_sample: bool = False  # False = greedy
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None


def _sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig
) -> torch.Tensor:
    """Sample next token from logits."""
    # Greedy decoding (no temperature needed)
    if not config.do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    # Apply temperature (only for sampling, avoid division by zero)
    if config.temperature > 0 and config.temperature != 1.0:
        logits = logits / config.temperature

    # Apply top-k filtering
    if config.top_k > 0:
        indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    # Apply top-p (nucleus) filtering
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

    # Sample from distribution
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_metrics(
    model,
    input_ids: torch.Tensor,
    config: GenerationConfig,
    tokenizer=None
) -> GenerationMetrics:
    """
    Generate text and measure TTFT and throughput.

    Args:
        model: The language model
        input_ids: Input token IDs [batch_size, seq_len]
        config: Generation configuration
        tokenizer: Optional tokenizer for decoding

    Returns:
        GenerationMetrics with timing information
    """
    prompt_length = input_ids.shape[1]
    generated_ids = input_ids.clone()

    ttft_ms = 0.0
    token_times = []

    with torch.no_grad():
        for i in range(config.max_new_tokens):
            step_start = time.perf_counter()

            # Forward pass
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]

            # Sample next token
            next_token = _sample_next_token(next_token_logits, config)

            step_end = time.perf_counter()
            step_time = (step_end - step_start) * 1000  # ms

            if i == 0:
                ttft_ms = step_time
            else:
                token_times.append(step_time)

            # Append token
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS
            if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
                break

    num_tokens = generated_ids.shape[1] - prompt_length
    total_time_ms = ttft_ms + sum(token_times)

    # Calculate tokens per second (excluding first token which includes prefill)
    if len(token_times) > 0:
        tokens_per_sec = len(token_times) / (sum(token_times) / 1000)
    else:
        tokens_per_sec = 0.0

    # Decode output
    if tokenizer is not None:
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    else:
        output_text = ""

    return GenerationMetrics(
        ttft_ms=ttft_ms,
        total_time_ms=total_time_ms,
        num_tokens=num_tokens,
        tokens_per_sec=tokens_per_sec,
        prompt_length=prompt_length,
        output_text=output_text
    )


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = False,
    return_metrics: bool = False
) -> Union[str, Tuple[str, GenerationMetrics]]:
    """
    High-level generation function.

    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        do_sample: Whether to sample (False = greedy)
        return_metrics: Whether to return timing metrics

    Returns:
        Generated text, or tuple of (text, metrics) if return_metrics=True
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    metrics = generate_with_metrics(model, input_ids, config, tokenizer)

    if return_metrics:
        return metrics.output_text, metrics
    return metrics.output_text


def benchmark_ttft(
    model,
    tokenizer,
    prompts: List[str],
    warmup: int = 3,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark Time-to-First-Token (TTFT) across multiple prompts.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompts: List of test prompts
        warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with TTFT statistics
    """
    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        with torch.no_grad():
            _ = model(input_ids)

    # Collect TTFT measurements
    ttft_values = []

    print(f"  Running {num_runs} iterations over {len(prompts)} prompts...")
    for run in range(num_runs):
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            prompt_len = input_ids.shape[1]

            start = time.perf_counter()
            with torch.no_grad():
                logits = model(input_ids)
                _ = logits[:, -1, :].argmax(dim=-1)
            end = time.perf_counter()

            ttft_ms = (end - start) * 1000
            ttft_values.append({
                'ttft_ms': ttft_ms,
                'prompt_tokens': prompt_len
            })

    # Compute statistics
    all_ttft = [v['ttft_ms'] for v in ttft_values]

    return {
        'mean_ttft_ms': sum(all_ttft) / len(all_ttft),
        'min_ttft_ms': min(all_ttft),
        'max_ttft_ms': max(all_ttft),
        'p50_ttft_ms': sorted(all_ttft)[len(all_ttft) // 2],
        'p95_ttft_ms': sorted(all_ttft)[int(len(all_ttft) * 0.95)],
        'p99_ttft_ms': sorted(all_ttft)[int(len(all_ttft) * 0.99)],
        'num_samples': len(all_ttft),
        'measurements': ttft_values
    }


def benchmark_throughput(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = 50,
    warmup: int = 3,
    num_runs: int = 5
) -> Dict:
    """
    Benchmark token generation throughput.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Test prompt
        num_tokens: Number of tokens to generate per run
        warmup: Number of warmup runs
        num_runs: Number of benchmark runs

    Returns:
        Dictionary with throughput statistics
    """
    config = GenerationConfig(
        max_new_tokens=num_tokens,
        do_sample=False,
        eos_token_id=None  # Don't stop early for benchmark
    )

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Warmup
    print("  Warming up...")
    for _ in range(warmup):
        _ = generate_with_metrics(model, input_ids.clone(), config)

    # Benchmark
    print(f"  Running {num_runs} generations of {num_tokens} tokens each...")
    results = []
    for run in range(num_runs):
        metrics = generate_with_metrics(model, input_ids.clone(), config, tokenizer)
        results.append({
            'ttft_ms': metrics.ttft_ms,
            'total_time_ms': metrics.total_time_ms,
            'tokens_per_sec': metrics.tokens_per_sec,
            'num_tokens': metrics.num_tokens
        })

    # Compute statistics
    all_tps = [r['tokens_per_sec'] for r in results]
    all_ttft = [r['ttft_ms'] for r in results]
    all_total = [r['total_time_ms'] for r in results]

    return {
        'mean_tokens_per_sec': sum(all_tps) / len(all_tps),
        'mean_ttft_ms': sum(all_ttft) / len(all_ttft),
        'mean_total_time_ms': sum(all_total) / len(all_total),
        'num_tokens_per_run': num_tokens,
        'num_runs': num_runs,
        'results': results
    }


def compare_implementations(
    our_model,
    baseline_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    num_runs: int = 3
) -> Dict:
    """
    Compare our implementation vs baseline (e.g., HuggingFace).

    Returns:
        Comparison metrics including speedups
    """
    print("Benchmarking our implementation...")
    our_ttft = benchmark_ttft(our_model, tokenizer, prompts, num_runs=num_runs)
    our_throughput = benchmark_throughput(
        our_model, tokenizer, prompts[0],
        num_tokens=max_new_tokens, num_runs=num_runs
    )

    print("\nBenchmarking baseline implementation...")
    baseline_ttft = benchmark_ttft(baseline_model, tokenizer, prompts, num_runs=num_runs)
    baseline_throughput = benchmark_throughput(
        baseline_model, tokenizer, prompts[0],
        num_tokens=max_new_tokens, num_runs=num_runs
    )

    return {
        'our_implementation': {
            'mean_ttft_ms': our_ttft['mean_ttft_ms'],
            'p95_ttft_ms': our_ttft['p95_ttft_ms'],
            'tokens_per_sec': our_throughput['mean_tokens_per_sec'],
        },
        'baseline': {
            'mean_ttft_ms': baseline_ttft['mean_ttft_ms'],
            'p95_ttft_ms': baseline_ttft['p95_ttft_ms'],
            'tokens_per_sec': baseline_throughput['mean_tokens_per_sec'],
        },
        'speedup': {
            'ttft': baseline_ttft['mean_ttft_ms'] / our_ttft['mean_ttft_ms'],
            'throughput': our_throughput['mean_tokens_per_sec'] / baseline_throughput['mean_tokens_per_sec'],
        }
    }


# ============================================================================
# Streaming Generation
# ============================================================================

def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    callback: Optional[Callable[[str, int], None]] = None
):
    """
    Stream tokens as they are generated.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        callback: Called with (new_token_text, token_idx) for each token

    Yields:
        Tuples of (token_text, is_first_token, time_ms)
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    generated_ids = input_ids.clone()

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        eos_token_id=tokenizer.eos_token_id
    )

    with torch.no_grad():
        for i in range(max_new_tokens):
            step_start = time.perf_counter()

            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            next_token = _sample_next_token(next_token_logits, config)

            step_time = (time.perf_counter() - step_start) * 1000

            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Decode just the new token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)

            yield (token_text, i == 0, step_time)

            if callback:
                callback(token_text, i)

            if config.eos_token_id is not None and next_token.item() == config.eos_token_id:
                break


if __name__ == '__main__':
    # Quick test
    print("Generation utilities loaded successfully")
    print("\nAvailable functions:")
    print("  - generate(): High-level text generation")
    print("  - generate_with_metrics(): Generation with TTFT measurement")
    print("  - benchmark_ttft(): Benchmark time-to-first-token")
    print("  - benchmark_throughput(): Benchmark tokens per second")
    print("  - stream_generate(): Streaming token generation")
    print("  - compare_implementations(): Compare our kernels vs baseline")
