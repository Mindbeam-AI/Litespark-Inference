#!/usr/bin/env python3
"""
Profile token generation (tg128) to find bottlenecks.
Measures time spent in: kernel calls, attention, Python overhead.
"""

import torch
import time
import platform
import os
import sys
import json
from collections import defaultdict
from functools import wraps

# Timing storage
TIMINGS = defaultdict(list)
CALL_COUNTS = defaultdict(int)

def time_function(name):
    """Decorator to time function calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            TIMINGS[name].append(elapsed)
            CALL_COUNTS[name] += 1
            return result
        return wrapper
    return decorator

def profile_inference(num_threads=4, num_tokens=32):
    """Profile token generation with detailed timing."""
    from litespark_inf.models import load_ternary_model, get_kernel, get_kernel_type
    import gc

    print(f"{'='*70}")
    print(f"PROFILING tg128 - {platform.machine()}")
    print(f"{'='*70}")

    # Get system info
    if platform.system() == 'Darwin':
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                              capture_output=True, text=True)
        cpu_name = result.stdout.strip()
    elif os.path.exists('/proc/cpuinfo'):
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    cpu_name = line.split(':')[1].strip()
                    break
    else:
        cpu_name = platform.processor()

    print(f"CPU: {cpu_name}")
    print(f"Threads: {num_threads}")
    print(f"Tokens to generate: {num_tokens}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_ternary_model('bitnet-2b', num_threads=num_threads, mode='neon')
    kernel_type = get_kernel_type()
    print(f"Kernel: {kernel_type}")
    gc.collect()

    # Patch kernel functions to measure time
    kernel = get_kernel()
    original_funcs = {}

    # Find and patch kernel functions
    kernel_func_names = [name for name in dir(kernel) if 'matmul' in name.lower()]
    print(f"Kernel functions found: {kernel_func_names}")

    for func_name in kernel_func_names:
        original_func = getattr(kernel, func_name)
        original_funcs[func_name] = original_func

        def make_timed_func(fname, orig):
            def timed_func(*args, **kwargs):
                start = time.perf_counter()
                result = orig(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                TIMINGS[f'kernel_{fname}'].append(elapsed)
                CALL_COUNTS[f'kernel_{fname}'] += 1
                return result
            return timed_func

        setattr(kernel, func_name, make_timed_func(func_name, original_func))

    # Prepare input
    prompt = "The quick brown fox"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Input tokens: {input_ids.shape[1]}")

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = model.generate(input_ids, max_new_tokens=5, do_sample=False)

    # Clear timings from warmup
    TIMINGS.clear()
    CALL_COUNTS.clear()

    # Profile token generation
    print(f"\nGenerating {num_tokens} tokens...")

    total_start = time.perf_counter()

    with torch.no_grad():
        # Manual token-by-token generation for detailed profiling
        generated = input_ids.clone()

        for i in range(num_tokens):
            token_start = time.perf_counter()

            # Forward pass
            forward_start = time.perf_counter()
            outputs = model(generated)
            forward_time = (time.perf_counter() - forward_start) * 1000
            TIMINGS['forward_pass'].append(forward_time)

            # Get next token
            sample_start = time.perf_counter()
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            sample_time = (time.perf_counter() - sample_start) * 1000
            TIMINGS['sampling'].append(sample_time)

            # Append token
            append_start = time.perf_counter()
            generated = torch.cat([generated, next_token], dim=1)
            append_time = (time.perf_counter() - append_start) * 1000
            TIMINGS['tensor_concat'].append(append_time)

            token_time = (time.perf_counter() - token_start) * 1000
            TIMINGS['total_per_token'].append(token_time)

    total_time = (time.perf_counter() - total_start) * 1000

    # Restore original functions
    for func_name, original_func in original_funcs.items():
        setattr(kernel, func_name, original_func)

    # Compute statistics
    print(f"\n{'='*70}")
    print("PROFILING RESULTS")
    print(f"{'='*70}")

    print(f"\nTotal time: {total_time:.2f} ms for {num_tokens} tokens")
    print(f"Throughput: {num_tokens / (total_time/1000):.2f} tok/s")

    print(f"\n--- Per-token breakdown (avg) ---")

    results = {
        'platform': cpu_name,
        'kernel_type': kernel_type,
        'num_threads': num_threads,
        'num_tokens': num_tokens,
        'total_time_ms': total_time,
        'throughput_tps': num_tokens / (total_time/1000),
        'timings': {},
        'call_counts': dict(CALL_COUNTS),
    }

    # Sort by total time
    timing_totals = []
    for name, times in TIMINGS.items():
        total = sum(times)
        avg = total / len(times) if times else 0
        timing_totals.append((name, total, avg, len(times)))
        results['timings'][name] = {
            'total_ms': total,
            'avg_ms': avg,
            'count': len(times),
            'pct_of_total': (total / total_time) * 100
        }

    timing_totals.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Category':<40} {'Total (ms)':<12} {'Avg (ms)':<12} {'Count':<8} {'% of Total':<10}")
    print("-" * 82)

    for name, total, avg, count in timing_totals:
        pct = (total / total_time) * 100
        print(f"{name:<40} {total:>10.2f}   {avg:>10.3f}   {count:>6}   {pct:>8.1f}%")

    # Compute kernel vs non-kernel time
    kernel_time = sum(sum(times) for name, times in TIMINGS.items() if 'kernel_' in name)
    forward_time = sum(TIMINGS.get('forward_pass', [0]))

    print(f"\n--- Summary ---")
    print(f"Total kernel time:    {kernel_time:>10.2f} ms ({kernel_time/total_time*100:.1f}%)")
    print(f"Total forward time:   {forward_time:>10.2f} ms ({forward_time/total_time*100:.1f}%)")
    print(f"Python overhead:      {total_time - forward_time:>10.2f} ms ({(total_time-forward_time)/total_time*100:.1f}%)")

    # Estimate what's NOT kernel
    non_kernel_in_forward = forward_time - kernel_time
    print(f"Non-kernel in forward:{non_kernel_in_forward:>10.2f} ms ({non_kernel_in_forward/total_time*100:.1f}%) <- attention, embeddings, etc")

    results['summary'] = {
        'kernel_time_ms': kernel_time,
        'kernel_pct': kernel_time/total_time*100,
        'forward_time_ms': forward_time,
        'forward_pct': forward_time/total_time*100,
        'python_overhead_ms': total_time - forward_time,
        'python_overhead_pct': (total_time-forward_time)/total_time*100,
        'non_kernel_in_forward_ms': non_kernel_in_forward,
        'non_kernel_in_forward_pct': non_kernel_in_forward/total_time*100,
    }

    # Kernel calls per token
    total_kernel_calls = sum(CALL_COUNTS.get(name, 0) for name in CALL_COUNTS if 'kernel_' in name)
    print(f"\nKernel calls per token: {total_kernel_calls / num_tokens:.1f}")
    results['kernel_calls_per_token'] = total_kernel_calls / num_tokens

    # Save results
    output_file = f"profile_results_{platform.machine().lower()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', '-t', type=int, default=4)
    parser.add_argument('--tokens', '-n', type=int, default=32)
    args = parser.parse_args()

    profile_inference(num_threads=args.threads, num_tokens=args.tokens)
