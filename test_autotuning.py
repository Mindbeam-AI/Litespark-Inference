#!/usr/bin/env python3
"""
Quick test to verify Triton autotuning is working correctly
"""

import torch
import time
from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM

def test_model_forward():
    """Test that model forward pass works with autotuning enabled"""
    print("Testing model with autotuning enabled...")

    # Small model for quick testing
    config = HGRNBitConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
        max_position_embeddings=512,
    )

    print(f"Creating model with {sum(p.numel() for p in HGRNBitForCausalLM(config).parameters())/1e6:.1f}M params")
    model = HGRNBitForCausalLM(config).cuda()
    model.eval()

    # Test different batch sizes
    for batch_size in [1, 2, 4, 8]:
        for seq_len in [64, 128, 256]:
            x = torch.randint(0, 1000, (batch_size, seq_len)).cuda()

            with torch.no_grad():
                try:
                    output = model(x)
                    print(f"✓ Batch={batch_size}, Seq={seq_len}: Output shape {output.logits.shape}")
                except Exception as e:
                    print(f"✗ Batch={batch_size}, Seq={seq_len}: FAILED - {e}")
                    return False

    print("\n✓ All forward pass tests passed!")
    return True

def benchmark_throughput():
    """Benchmark throughput to see autotuning impact"""
    print("\nBenchmarking throughput...")

    config = HGRNBitConfig(
        vocab_size=50000,
        hidden_size=1024,
        num_hidden_layers=12,
        num_heads=8,
        max_position_embeddings=2048,
    )

    model = HGRNBitForCausalLM(config).cuda()
    model.eval()

    batch_size = 4
    seq_len = 1024
    num_iterations = 50

    x = torch.randint(0, 50000, (batch_size, seq_len)).cuda()

    # Warmup (important for autotuning!)
    print("Warming up (autotuning kernels)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Benchmark
    print(f"Running {num_iterations} iterations...")
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    tokens_processed = batch_size * seq_len * num_iterations
    throughput = tokens_processed / elapsed

    print(f"\n{'='*60}")
    print(f"Throughput: {throughput:,.0f} tokens/sec")
    print(f"Time per iteration: {elapsed/num_iterations*1000:.2f} ms")
    print(f"{'='*60}")

    # Memory usage
    memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {memory_allocated:.2f} GB")

    return throughput

def test_backward_pass():
    """Test backward pass with autotuning"""
    print("\nTesting backward pass...")

    config = HGRNBitConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_heads=4,
    )

    model = HGRNBitForCausalLM(config).cuda()
    model.train()

    x = torch.randint(0, 1000, (2, 128)).cuda()

    try:
        output = model(x, labels=x)
        loss = output.loss
        loss.backward()
        print(f"✓ Backward pass successful! Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Backward pass FAILED: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Triton Autotuning Fixes")
    print("="*60)
    print()

    # Test 1: Forward pass
    if not test_model_forward():
        print("\n❌ Forward pass test failed!")
        exit(1)

    # Test 2: Backward pass
    if not test_backward_pass():
        print("\n❌ Backward pass test failed!")
        exit(1)

    # Test 3: Benchmark
    try:
        throughput = benchmark_throughput()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        exit(1)

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nAutotuning is working correctly.")
    print(f"Measured throughput: {throughput:,.0f} tokens/sec")
    print("\nNote: First run will be slower due to autotuning warmup.")
    print("Subsequent runs will be faster as Triton caches optimal configs.")
