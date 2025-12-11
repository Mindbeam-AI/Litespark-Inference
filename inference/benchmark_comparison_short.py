#!/usr/bin/env python3
"""
Benchmark comparison: VNNI kernels vs PyTorch baseline

Compares:
1. Our VNNI/Graviton kernels (ternary quantized)
2. PyTorch baseline (ternary weights with torch.matmul)

Metrics:
- TTFT (Time-to-First-Token)
- Throughput (tokens/sec)
- Memory usage (RSS)
- Perplexity on WikiText-2
- Numerical accuracy (logits comparison)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import gc
import platform
import resource
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import matplotlib for graphs
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping graphs")

from inference.ternary_models import (
    BitNet, BitNetConfig, load_ternary_model, get_arch_info,
    prepare_ternary_weight
)


# ============================================================================
# Memory Tracking Utilities
# ============================================================================

def get_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    if platform.system() == 'Darwin':
        # macOS: ru_maxrss is in bytes
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    else:
        # Linux: ru_maxrss is in KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def get_peak_memory_mb() -> float:
    """Get peak RSS memory usage."""
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmHWM:'):  # High water mark
                    return int(line.split()[1]) / 1024.0
    except:
        pass
    return get_memory_mb()


# ============================================================================
# Perplexity Evaluation
# ============================================================================

def evaluate_perplexity(model, tokenizer, max_samples: int = 100, max_length: int = 512) -> float:
    """
    Evaluate perplexity on WikiText-2 test set.

    Args:
        model: The language model
        tokenizer: Tokenizer
        max_samples: Maximum number of samples to evaluate
        max_length: Maximum sequence length per sample

    Returns:
        Perplexity score (lower is better)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [!] datasets library not installed, skipping perplexity")
        return float('nan')

    print("  Loading WikiText-2 test set...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    except Exception as e:
        print(f"  [!] Failed to load WikiText-2: {e}")
        return float('nan')

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    print(f"  Evaluating on {min(max_samples, len(dataset))} samples...")

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break

            text = sample['text']
            if not text.strip():
                continue

            # Tokenize
            encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
            input_ids = encodings['input_ids']

            if input_ids.shape[1] < 2:
                continue

            # Get model outputs
            logits = model(input_ids)

            # Calculate cross-entropy loss
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float('nan')

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


# ============================================================================
# Numerical Accuracy Comparison
# ============================================================================

def compare_outputs(model1, model2, tokenizer, prompts: List[str], tolerance: float = 1e-4) -> Dict:
    """
    Compare outputs between two models for numerical accuracy.

    Args:
        model1: First model (e.g., VNNI kernels)
        model2: Second model (e.g., PyTorch baseline)
        tokenizer: Tokenizer
        prompts: List of test prompts
        tolerance: Tolerance for considering outputs "matching"

    Returns:
        Dictionary with comparison metrics
    """
    model1.eval()
    model2.eval()

    results = {
        'max_abs_diff': 0.0,
        'mean_abs_diff': 0.0,
        'matches_within_tolerance': 0,
        'total_comparisons': 0,
        'token_agreement': 0,
        'total_tokens': 0,
    }

    all_diffs = []

    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')

            logits1 = model1(input_ids)
            logits2 = model2(input_ids)

            # Compare logits
            diff = (logits1 - logits2).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            all_diffs.append(mean_diff)
            results['max_abs_diff'] = max(results['max_abs_diff'], max_diff)
            results['total_comparisons'] += 1

            if max_diff < tolerance:
                results['matches_within_tolerance'] += 1

            # Check if top tokens agree
            top1_model1 = logits1[0, -1, :].argmax().item()
            top1_model2 = logits2[0, -1, :].argmax().item()

            results['total_tokens'] += 1
            if top1_model1 == top1_model2:
                results['token_agreement'] += 1

    results['mean_abs_diff'] = sum(all_diffs) / len(all_diffs) if all_diffs else 0.0

    return results


# ============================================================================
# PyTorch Baseline Model (ternary weights, but using torch.matmul)
# ============================================================================

class PyTorchTernaryLinear(nn.Module):
    """Ternary linear using standard PyTorch (for baseline comparison)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight', None)
        self.scale = 1.0

    def load_from_weight(self, weight: torch.Tensor):
        """Quantize weight to ternary."""
        weight_abs_mean = weight.abs().mean()
        if weight_abs_mean > 0:
            self.scale = weight_abs_mean.item()
            scale_inv = 1.0 / self.scale
        else:
            self.scale = 1.0
            scale_inv = 1.0

        # Quantize to ternary {-1, 0, +1} and store as float for matmul
        w_ternary = (weight * scale_inv).round().clamp(-1, 1)
        self.weight = w_ternary.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard PyTorch matmul
        return F.linear(x, self.weight) * self.scale


def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


class PyTorchBitNetAttention(nn.Module):
    """BitNet attention using PyTorch baseline."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads

        self.q_proj = PyTorchTernaryLinear(config.hidden_size, config.hidden_size)
        kv_dim = config.num_kv_heads * self.head_dim
        self.k_proj = PyTorchTernaryLinear(config.hidden_size, kv_dim)
        self.v_proj = PyTorchTernaryLinear(config.hidden_size, kv_dim)
        self.o_proj = PyTorchTernaryLinear(config.hidden_size, config.hidden_size)

        self.attn_sub_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale = 1.0 / (self.head_dim ** 0.5)

        # RoPE
        self.rope_theta = config.rope_theta
        self._init_rope(config.max_position_embeddings)

    def _init_rope(self, max_seq_len: int):
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        x1 = x[..., :self.head_dim // 2]
        x2 = x[..., self.head_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        out = self.attn_sub_norm(out)
        return self.o_proj(out)


class PyTorchBitNetMLP(nn.Module):
    """BitNet MLP using PyTorch baseline."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.gate_proj = PyTorchTernaryLinear(config.hidden_size, config.intermediate_size)
        self.up_proj = PyTorchTernaryLinear(config.hidden_size, config.intermediate_size)
        self.down_proj = PyTorchTernaryLinear(config.intermediate_size, config.hidden_size)
        self.ffn_sub_norm = nn.RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = relu2(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class PyTorchBitNetBlock(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = PyTorchBitNetAttention(config)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = PyTorchBitNetMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class PyTorchBitNet(nn.Module):
    """BitNet using standard PyTorch (baseline for comparison)."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PyTorchBitNetBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)

    @classmethod
    def from_safetensors(cls, model_name: str):
        """Load from safetensors."""
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        import json

        config_path = hf_hub_download(model_name, 'config.json')
        with open(config_path) as f:
            hf_config = json.load(f)

        config = BitNetConfig(
            hidden_size=hf_config['hidden_size'],
            num_layers=hf_config['num_hidden_layers'],
            num_heads=hf_config['num_attention_heads'],
            num_kv_heads=hf_config.get('num_key_value_heads', hf_config['num_attention_heads']),
            intermediate_size=hf_config['intermediate_size'],
            vocab_size=hf_config['vocab_size'],
            max_position_embeddings=hf_config.get('max_position_embeddings', 4096),
            rms_norm_eps=hf_config.get('rms_norm_eps', 1e-5),
            rope_theta=hf_config.get('rope_theta', 500000.0),
        )

        model = cls(config)
        model_path = hf_hub_download(model_name, 'model.safetensors')

        with safe_open(model_path, framework="pt") as f:
            model.embed_tokens.weight.data = f.get_tensor('model.embed_tokens.weight').float()

            for i, layer in enumerate(model.layers):
                prefix = f'model.layers.{i}'

                layer.attn.q_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.q_proj.weight').float())
                layer.attn.k_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.k_proj.weight').float())
                layer.attn.v_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.v_proj.weight').float())
                layer.attn.o_proj.load_from_weight(f.get_tensor(f'{prefix}.self_attn.o_proj.weight').float())
                layer.attn.attn_sub_norm.weight.data = f.get_tensor(f'{prefix}.self_attn.attn_sub_norm.weight').float()

                layer.mlp.gate_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.gate_proj.weight').float())
                layer.mlp.up_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.up_proj.weight').float())
                layer.mlp.down_proj.load_from_weight(f.get_tensor(f'{prefix}.mlp.down_proj.weight').float())
                layer.mlp.ffn_sub_norm.weight.data = f.get_tensor(f'{prefix}.mlp.ffn_sub_norm.weight').float()

                layer.input_norm.weight.data = f.get_tensor(f'{prefix}.input_layernorm.weight').float()
                layer.post_attn_norm.weight.data = f.get_tensor(f'{prefix}.post_attention_layernorm.weight').float()

            model.norm.weight.data = f.get_tensor('model.norm.weight').float()

        return model


# ============================================================================
# HuggingFace Official Model Wrapper
# ============================================================================

class HuggingFaceModelWrapper(nn.Module):
    """Wrapper to make HuggingFace model compatible with our benchmark interface."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits in same format as our models."""
        outputs = self.model(input_ids)
        return outputs.logits

    @classmethod
    def from_pretrained(cls, model_name: str):
        """
        Load HuggingFace model.

        BitNet is now supported in transformers >= 4.48.0
        Use the packed model (bitnet-b1.58-2B-4T), NOT bf16 version
        Do NOT use trust_remote_code for BitNet models
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"  Loading HuggingFace model: {model_name}")

        # Check the config
        try:
            config = AutoConfig.from_pretrained(model_name)
            model_type = getattr(config, 'model_type', 'unknown')
            print(f"  Model type: {model_type}")
        except Exception as e:
            print(f"  Could not load config: {e}")
            raise

        # BitNet models - now in standard transformers (>= 4.48.0)
        # Do NOT use trust_remote_code, use bfloat16 as recommended
        if model_type == 'bitnet':
            print("  Loading BitNet with bfloat16 (HF recommended)")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=None,
            )
            return cls(hf_model)

        # For standard models
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None,
            trust_remote_code=True,
        )
        return cls(hf_model)


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_model(model, tokenizer, name: str, num_runs: int = 5, gen_tokens: int = 20):
    """Benchmark a single model with memory tracking."""
    model.eval()
    gc.collect()

    prompts = [
        "The future of AI is",
        "Once upon a time",
    ]

    # Measure memory before
    mem_before = get_memory_mb()

    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(2):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        with torch.no_grad():
            _ = model(input_ids)

    # Measure memory after warmup (model fully loaded)
    mem_after_warmup = get_memory_mb()

    # TTFT benchmark
    ttft_times = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            ttft_times.append((time.perf_counter() - start) * 1000)

    # Generation benchmark
    gen_times = []
    for _ in range(num_runs):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        generated = input_ids.clone()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(gen_tokens):
                logits = model(generated)
                next_token = logits[0, -1, :].argmax(keepdim=True)
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        total_time = (time.perf_counter() - start) * 1000
        gen_times.append(total_time)

    # Measure peak memory
    mem_peak = get_peak_memory_mb()

    # Sample generation for quality check
    input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(gen_tokens):
            logits = model(generated)
            next_token = logits[0, -1, :].argmax(keepdim=True)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    sample_output = tokenizer.decode(generated[0], skip_special_tokens=True)

    return {
        'name': name,
        'mean_ttft_ms': sum(ttft_times) / len(ttft_times),
        'mean_gen_time_ms': sum(gen_times) / len(gen_times),
        'tokens_per_sec': gen_tokens / (sum(gen_times) / len(gen_times) / 1000),
        'sample_output': sample_output,
        'memory_mb': mem_after_warmup,
        'peak_memory_mb': mem_peak,
    }


def profile_vnni_kernel(model, tokenizer):
    """Profile where time is spent in VNNI kernel model."""
    import torch.profiler as profiler

    print("\n" + "=" * 70)
    print("VNNI Kernel Profiling")
    print("=" * 70)

    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids)

    # Manual timing breakdown
    print("\nManual timing breakdown (single forward pass):")

    # Time embedding
    start = time.perf_counter()
    with torch.no_grad():
        x = model.embed_tokens(input_ids)
    embed_time = (time.perf_counter() - start) * 1000

    # Time each layer
    layer_times = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            start = time.perf_counter()
            x = layer(x)
            layer_times.append((time.perf_counter() - start) * 1000)

    # Time final norm + lm_head
    start = time.perf_counter()
    with torch.no_grad():
        x = model.norm(x)
        logits = F.linear(x, model.embed_tokens.weight)
    head_time = (time.perf_counter() - start) * 1000

    print(f"  Embedding: {embed_time:.2f} ms")
    print(f"  Layers (avg): {sum(layer_times)/len(layer_times):.2f} ms x {len(layer_times)} = {sum(layer_times):.2f} ms")
    print(f"  Layer breakdown:")
    print(f"    First 3: {layer_times[0]:.2f}, {layer_times[1]:.2f}, {layer_times[2]:.2f} ms")
    print(f"    Last 3: {layer_times[-3]:.2f}, {layer_times[-2]:.2f}, {layer_times[-1]:.2f} ms")
    print(f"  Norm + LM Head: {head_time:.2f} ms")
    print(f"  Total: {embed_time + sum(layer_times) + head_time:.2f} ms")

    # Profile a single layer in detail
    print("\nDetailed layer 0 breakdown:")
    layer = model.layers[0]
    x = model.embed_tokens(input_ids)

    # Input norm
    start = time.perf_counter()
    with torch.no_grad():
        normed = layer.input_norm(x)
    norm_time = (time.perf_counter() - start) * 1000

    # Attention
    start = time.perf_counter()
    with torch.no_grad():
        attn_out = layer.attn(normed)
    attn_time = (time.perf_counter() - start) * 1000

    # Post-attn norm
    x_after_attn = x + attn_out
    start = time.perf_counter()
    with torch.no_grad():
        normed2 = layer.post_attn_norm(x_after_attn)
    norm2_time = (time.perf_counter() - start) * 1000

    # MLP
    start = time.perf_counter()
    with torch.no_grad():
        mlp_out = layer.mlp(normed2)
    mlp_time = (time.perf_counter() - start) * 1000

    print(f"  Input norm: {norm_time:.2f} ms")
    print(f"  Attention: {attn_time:.2f} ms")
    print(f"  Post-attn norm: {norm2_time:.2f} ms")
    print(f"  MLP: {mlp_time:.2f} ms")

    # Profile attention in detail
    print("\nDetailed attention breakdown:")
    attn = layer.attn
    x_in = normed

    start = time.perf_counter()
    with torch.no_grad():
        q = attn.q_proj(x_in)
    q_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with torch.no_grad():
        k = attn.k_proj(x_in)
    k_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with torch.no_grad():
        v = attn.v_proj(x_in)
    v_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with torch.no_grad():
        o = attn.o_proj(x_in)  # Just timing the projection
    o_time = (time.perf_counter() - start) * 1000

    print(f"  Q proj: {q_time:.2f} ms")
    print(f"  K proj: {k_time:.2f} ms")
    print(f"  V proj: {v_time:.2f} ms")
    print(f"  O proj: {o_time:.2f} ms")
    print(f"  Total projections: {q_time + k_time + v_time + o_time:.2f} ms")

    # Profile MLP in detail
    print("\nDetailed MLP breakdown:")
    mlp = layer.mlp
    x_in = normed2

    start = time.perf_counter()
    with torch.no_grad():
        gate = mlp.gate_proj(x_in)
    gate_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    with torch.no_grad():
        up = mlp.up_proj(x_in)
    up_time = (time.perf_counter() - start) * 1000

    # Use correct intermediate size input for down_proj
    hidden = gate * up
    start = time.perf_counter()
    with torch.no_grad():
        down = mlp.down_proj(hidden)
    down_time = (time.perf_counter() - start) * 1000

    print(f"  Gate proj: {gate_time:.2f} ms")
    print(f"  Up proj: {up_time:.2f} ms")
    print(f"  Down proj: {down_time:.2f} ms")
    print(f"  Total projections: {gate_time + up_time + down_time:.2f} ms")

    # Profile quantization overhead vs kernel time
    print("\n" + "=" * 70)
    print("Quantization vs Kernel Breakdown")
    print("=" * 70)

    from inference.ternary_models import quantize_activation, get_kernel, get_kernel_type

    # Get a projection to test
    proj = layer.attn.q_proj
    x_test = normed.clone()

    # Time full forward (includes quantization)
    times_full = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = proj(x_test)
        times_full.append((time.perf_counter() - start) * 1000)
    full_time = sum(times_full) / len(times_full)

    # Time just quantization
    times_quant = []
    for _ in range(10):
        x_flat = x_test.view(-1, proj.in_features)
        start = time.perf_counter()
        x_int8, x_scale = quantize_activation(x_flat)
        times_quant.append((time.perf_counter() - start) * 1000)
    quant_time = sum(times_quant) / len(times_quant)

    # Time just kernel call
    kernel = get_kernel()
    kernel_type = get_kernel_type()
    x_flat = x_test.view(-1, proj.in_features)
    x_int8, x_scale = quantize_activation(x_flat)
    M = x_flat.shape[0]
    y = torch.zeros(M, proj.out_features, dtype=torch.float32)
    bias = torch.Tensor()

    # Select correct kernel function based on platform
    if kernel_type == 'graviton':
        kernel_fn = kernel.matmul_free_graviton_v3
    else:
        kernel_fn = kernel.matmul_free_vnni_v3

    times_kernel = []
    for _ in range(10):
        y.zero_()
        start = time.perf_counter()
        kernel_fn(
            x_int8, x_scale,
            proj.w_int8, proj.w_sum,
            y, bias,
            M, proj.out_features, proj.in_features, proj.num_threads
        )
        times_kernel.append((time.perf_counter() - start) * 1000)
    kernel_time = sum(times_kernel) / len(times_kernel)

    # Time scale multiplication
    times_scale = []
    for _ in range(10):
        start = time.perf_counter()
        _ = y * proj.scale
        times_scale.append((time.perf_counter() - start) * 1000)
    scale_time = sum(times_scale) / len(times_scale)

    print(f"\nQ projection (2560 -> 2560) breakdown:")
    print(f"  Full forward:     {full_time:.3f} ms")
    print(f"  - Quantization:   {quant_time:.3f} ms ({100*quant_time/full_time:.1f}%)")
    print(f"  - Kernel:         {kernel_time:.3f} ms ({100*kernel_time/full_time:.1f}%)")
    print(f"  - Scale multiply: {scale_time:.3f} ms ({100*scale_time/full_time:.1f}%)")
    print(f"  - Other overhead: {full_time - quant_time - kernel_time - scale_time:.3f} ms")

    # Test MLP projection too (larger)
    proj = layer.mlp.gate_proj
    x_test = normed2.clone()

    times_full = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            _ = proj(x_test)
        times_full.append((time.perf_counter() - start) * 1000)
    full_time = sum(times_full) / len(times_full)

    times_quant = []
    for _ in range(10):
        x_flat = x_test.view(-1, proj.in_features)
        start = time.perf_counter()
        x_int8, x_scale = quantize_activation(x_flat)
        times_quant.append((time.perf_counter() - start) * 1000)
    quant_time = sum(times_quant) / len(times_quant)

    x_flat = x_test.view(-1, proj.in_features)
    x_int8, x_scale = quantize_activation(x_flat)
    M = x_flat.shape[0]
    y = torch.zeros(M, proj.out_features, dtype=torch.float32)

    times_kernel = []
    for _ in range(10):
        y.zero_()
        start = time.perf_counter()
        kernel_fn(
            x_int8, x_scale,
            proj.w_int8, proj.w_sum,
            y, bias,
            M, proj.out_features, proj.in_features, proj.num_threads
        )
        times_kernel.append((time.perf_counter() - start) * 1000)
    kernel_time = sum(times_kernel) / len(times_kernel)

    print(f"\nGate projection (2560 -> 6912) breakdown:")
    print(f"  Full forward:     {full_time:.3f} ms")
    print(f"  - Quantization:   {quant_time:.3f} ms ({100*quant_time/full_time:.1f}%)")
    print(f"  - Kernel:         {kernel_time:.3f} ms ({100*kernel_time/full_time:.1f}%)")
    print(f"  - Other overhead: {full_time - quant_time - kernel_time:.3f} ms")

    # Compare with PyTorch matmul for same operation
    print(f"\nComparison with PyTorch F.linear:")
    w_float = proj.w_int8.float() * proj.scale  # Reconstruct float weight
    x_float = x_test.view(-1, proj.in_features)

    times_pytorch = []
    for _ in range(10):
        start = time.perf_counter()
        _ = F.linear(x_float, w_float)
        times_pytorch.append((time.perf_counter() - start) * 1000)
    pytorch_time = sum(times_pytorch) / len(times_pytorch)

    print(f"  PyTorch F.linear: {pytorch_time:.3f} ms")
    print(f"  Our kernel only:  {kernel_time:.3f} ms")
    print(f"  Kernel speedup:   {pytorch_time/kernel_time:.2f}x")

    # Test kernel performance across different M sizes
    print("\n" + "=" * 70)
    print("Kernel Performance vs Batch Size (M)")
    print("=" * 70)
    print(f"\nGate projection shape: K=2560 -> N=6912")
    print(f"{'M':<6} {'VNNI (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 45)

    K, N = 2560, 6912
    K_pad = ((K + 63) // 64) * 64
    w_test = torch.randint(-1, 2, (N, K_pad), dtype=torch.int8)
    w_sum_test = w_test.sum(dim=1, dtype=torch.int32)
    w_float = w_test[:, :K].float()

    # Collect kernel benchmark results
    kernel_results = {
        'batch_sizes': [],
        'vnni_times': [],
        'pytorch_times': [],
        'speedups': [],
    }

    for M in [1, 6, 16, 64, 128, 256]:
        x_int8 = torch.randint(-127, 127, (M, K_pad), dtype=torch.int8)
        x_scale = torch.ones(M)
        x_float = x_int8[:, :K].float()
        y = torch.zeros(M, N)
        bias = torch.Tensor()

        # Time kernel (platform-aware)
        for _ in range(3):
            kernel_fn(x_int8, x_scale, w_test, w_sum_test, y, bias, M, N, K, 16)
        times_vnni = []
        for _ in range(10):
            y.zero_()
            start = time.perf_counter()
            kernel_fn(x_int8, x_scale, w_test, w_sum_test, y, bias, M, N, K, 16)
            times_vnni.append((time.perf_counter() - start) * 1000)
        vnni_time = sum(times_vnni) / len(times_vnni)

        # Time PyTorch
        for _ in range(3):
            _ = F.linear(x_float, w_float)
        times_pt = []
        for _ in range(10):
            start = time.perf_counter()
            _ = F.linear(x_float, w_float)
            times_pt.append((time.perf_counter() - start) * 1000)
        pt_time = sum(times_pt) / len(times_pt)

        speedup = pt_time / vnni_time
        print(f"{M:<6} {vnni_time:<12.3f} {pt_time:<14.3f} {speedup:<10.2f}x")

        kernel_results['batch_sizes'].append(M)
        kernel_results['vnni_times'].append(vnni_time)
        kernel_results['pytorch_times'].append(pt_time)
        kernel_results['speedups'].append(speedup)

    return kernel_results


def create_comparison_graphs(results: List[Dict], kernel_results: Dict, output_dir: Path, arch_info: Dict):
    """Create and save benchmark comparison graphs."""
    if not HAS_MATPLOTLIB:
        print("Skipping graphs - matplotlib not available")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine display name based on architecture
    arch_display = "X86" if arch_info['machine'] == 'x86_64' else "Graviton"

    # Graph 1: Model comparison (TTFT, Throughput, Memory)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = [r['name'].replace(' (ternary)', '') for r in results]
    colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(results)]

    # TTFT
    ttfts = [r['mean_ttft_ms'] for r in results]
    bars1 = axes[0].bar(model_names, ttfts, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Time to First Token (TTFT)', fontsize=14)
    axes[0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars1, ttfts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Throughput
    toks = [r['tokens_per_sec'] for r in results]
    bars2 = axes[1].bar(model_names, toks, color=colors, alpha=0.8, edgecolor='black')
    axes[1].set_ylabel('Tokens/sec', fontsize=12)
    axes[1].set_title('Generation Throughput', fontsize=14)
    axes[1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars2, toks):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Memory
    mems = [r['memory_mb'] for r in results]
    bars3 = axes[2].bar(model_names, mems, color=colors, alpha=0.8, edgecolor='black')
    axes[2].set_ylabel('Memory (MB)', fontsize=12)
    axes[2].set_title('Memory Usage', fontsize=14)
    axes[2].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars3, mems):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison.png'}")

    # Graph 2: Kernel speedup vs batch size
    if kernel_results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        batch_sizes = kernel_results['batch_sizes']
        vnni_times = kernel_results['vnni_times']
        pytorch_times = kernel_results['pytorch_times']
        speedups = kernel_results['speedups']

        # Left: Execution time
        ax1.plot(batch_sizes, vnni_times, 'b-o', label=f'{arch_display} Kernel', linewidth=2, markersize=8)
        ax1.plot(batch_sizes, pytorch_times, 'r-s', label='PyTorch F.linear', linewidth=2, markersize=8)
        ax1.set_xlabel('Batch Size (M)', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Kernel Execution Time vs Batch Size', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)

        # Right: Speedup
        colors_bar = ['green' if s >= 1 else 'red' for s in speedups]
        bars = ax2.bar(range(len(batch_sizes)), speedups, color=colors_bar, alpha=0.7, edgecolor='black')
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Batch Size (M)', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title(f'{arch_display} Kernel Speedup vs PyTorch', fontsize=14)
        ax2.set_xticks(range(len(batch_sizes)))
        ax2.set_xticklabels([str(b) for b in batch_sizes])
        ax2.grid(True, alpha=0.3, axis='y')

        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'kernel_speedup.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'kernel_speedup.png'}")

    # Graph 3: Speedup summary
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))

        pytorch_idx = next((i for i, r in enumerate(results) if 'PyTorch' in r['name']), 1)
        vnni_idx = 0

        speedup_ttft = results[pytorch_idx]['mean_ttft_ms'] / results[vnni_idx]['mean_ttft_ms']
        speedup_throughput = results[vnni_idx]['tokens_per_sec'] / results[pytorch_idx]['tokens_per_sec']
        memory_reduction = results[pytorch_idx]['memory_mb'] / results[vnni_idx]['memory_mb']

        metrics = ['TTFT\nSpeedup', 'Throughput\nSpeedup', 'Memory\nReduction']
        values = [speedup_ttft, speedup_throughput, memory_reduction]
        colors_summary = ['#3498db', '#2ecc71', '#9b59b6']

        bars = ax.bar(metrics, values, color=colors_summary, alpha=0.8, edgecolor='black')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Parity')
        ax.set_ylabel('Factor (x)', fontsize=12)
        ax.set_title(f'{arch_display} Kernel vs PyTorch Summary', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / 'speedup_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'speedup_summary.png'}")


def save_results(results: List[Dict], kernel_results: Dict, accuracy: Dict,
                 arch_info: Dict, output_dir: Path):
    """Save benchmark results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'platform': arch_info['machine'],
            'kernel_type': arch_info['kernel_type'],
            'num_threads': torch.get_num_threads(),
            'torch_version': torch.__version__,
        },
        'model_benchmarks': results,
        'kernel_benchmarks': kernel_results,
        'numerical_accuracy': accuracy,
    }

    results_file = output_dir / 'benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved: {results_file}")

    # Also save a human-readable summary
    # Determine display name based on architecture
    arch_display = "X86" if arch_info['machine'] == 'x86_64' else "GRAVITON"
    summary_file = output_dir / 'benchmark_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{arch_display} BENCHMARK SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {all_results['timestamp']}\n")
        f.write(f"Platform: {arch_info['machine']}\n")
        f.write(f"Kernel: {arch_info['kernel_type']}\n")
        f.write(f"Threads: {torch.get_num_threads()}\n\n")

        f.write("-" * 70 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 70 + "\n\n")
        f.write(f"{'Model':<25} {'TTFT (ms)':<12} {'Tok/s':<10} {'Memory (MB)':<14}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['name']:<25} {r['mean_ttft_ms']:<12.1f} {r['tokens_per_sec']:<10.2f} {r['memory_mb']:<14.1f}\n")

        if len(results) >= 2:
            f.write("\n")
            pytorch_idx = next((i for i, r in enumerate(results) if 'PyTorch' in r['name']), 1)
            vnni_idx = 0
            f.write(f"Speedup vs PyTorch:\n")
            f.write(f"  TTFT: {results[pytorch_idx]['mean_ttft_ms'] / results[vnni_idx]['mean_ttft_ms']:.2f}x\n")
            f.write(f"  Throughput: {results[vnni_idx]['tokens_per_sec'] / results[pytorch_idx]['tokens_per_sec']:.2f}x\n")
            f.write(f"  Memory: {results[pytorch_idx]['memory_mb'] / results[vnni_idx]['memory_mb']:.2f}x reduction\n")

        if kernel_results:
            f.write("\n" + "-" * 70 + "\n")
            f.write("KERNEL PERFORMANCE (vs PyTorch F.linear)\n")
            f.write("-" * 70 + "\n\n")
            f.write(f"{'Batch (M)':<12} {'Kernel (ms)':<14} {'PyTorch (ms)':<14} {'Speedup':<10}\n")
            f.write("-" * 50 + "\n")
            for i, bs in enumerate(kernel_results['batch_sizes']):
                f.write(f"{bs:<12} {kernel_results['vnni_times'][i]:<14.3f} {kernel_results['pytorch_times'][i]:<14.3f} {kernel_results['speedups'][i]:<10.2f}x\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Saved: {summary_file}")


def main():
    print("=" * 70)
    print("BitNet Benchmark: VNNI/Graviton Kernels vs PyTorch")
    print("=" * 70)

    arch = get_arch_info()
    print(f"Platform: {arch['machine']}")
    print(f"Kernel: {arch['kernel_type']}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    # Create output directory for results (architecture-specific)
    arch_suffix = "x86" if arch['machine'] == 'x86_64' else "graviton"
    output_dir = Path(__file__).parent / f'benchmark_inference_{arch_suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    MODEL_NAME = 'microsoft/bitnet-b1.58-2B-4T-bf16'

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []
    kernel_results = {}

    # 1. VNNI/Graviton kernels (ternary)
    print("\n" + "=" * 70)
    print("1. Loading VNNI/Graviton kernel model (ternary)...")
    print("=" * 70)
    vnni_model, _ = load_ternary_model('bitnet-2b')
    vnni_model.eval()
    results.append(benchmark_model(vnni_model, tokenizer, "VNNI Kernels (ternary)"))

    # Profile VNNI kernel and collect kernel benchmark results
    kernel_results = profile_vnni_kernel(vnni_model, tokenizer)

    # 2. PyTorch baseline (ternary)
    print("\n" + "=" * 70)
    print("2. Loading PyTorch baseline model (ternary)...")
    print("=" * 70)
    pytorch_model = PyTorchBitNet.from_safetensors(MODEL_NAME)
    pytorch_model.eval()
    results.append(benchmark_model(pytorch_model, tokenizer, "PyTorch (ternary)"))

    # 3. HuggingFace official BitNet model
    # Note: Use the packed model (bitnet-b1.58-2B-4T), not the bf16 version
    # BitNet is now in transformers - no trust_remote_code needed
    print("\n" + "=" * 70)
    print("3. Loading HuggingFace official BitNet model...")
    print("=" * 70)
    HF_BITNET_MODEL = 'microsoft/bitnet-b1.58-2B-4T'  # Packed 1.58-bit weights
    try:
        hf_model = HuggingFaceModelWrapper.from_pretrained(HF_BITNET_MODEL)
        hf_model.eval()
        results.append(benchmark_model(hf_model, tokenizer, "HuggingFace (BitNet)"))
        hf_loaded = True
    except Exception as e:
        print(f"  [!] Failed to load HuggingFace model: {e}")
        print("  [!] You may need: pip install transformers>=4.48.0")
        hf_model = None
        hf_loaded = False

    # ========================================================================
    # Numerical Accuracy Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Numerical Accuracy Comparison")
    print("=" * 70)

    test_prompts = [
        "The future of AI is",
        "Once upon a time",
        "The meaning of life is",
        "In a world where",
        "Scientists have discovered",
    ]

    accuracy = compare_outputs(vnni_model, pytorch_model, tokenizer, test_prompts)
    print(f"\n  Comparing VNNI Kernels vs PyTorch baseline:")
    print(f"    Max absolute difference: {accuracy['max_abs_diff']:.6f}")
    print(f"    Mean absolute difference: {accuracy['mean_abs_diff']:.6f}")
    print(f"    Token agreement: {accuracy['token_agreement']}/{accuracy['total_tokens']} ({100*accuracy['token_agreement']/accuracy['total_tokens']:.1f}%)")

    # ========================================================================
    # Perplexity Evaluation
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Perplexity Evaluation (WikiText-2)")
    print("=" * 70)

    print("\n  VNNI Kernels:")
    vnni_ppl = evaluate_perplexity(vnni_model, tokenizer, max_samples=50)
    results[0]['perplexity'] = vnni_ppl
    print(f"    Perplexity: {vnni_ppl:.2f}")

    print("\n  PyTorch baseline:")
    pytorch_ppl = evaluate_perplexity(pytorch_model, tokenizer, max_samples=50)
    results[1]['perplexity'] = pytorch_ppl
    print(f"    Perplexity: {pytorch_ppl:.2f}")

    if hf_loaded:
        print("\n  HuggingFace official:")
        hf_ppl = evaluate_perplexity(hf_model, tokenizer, max_samples=50)
        results[2]['perplexity'] = hf_ppl
        print(f"    Perplexity: {hf_ppl:.2f}")

    # Clean up models
    del vnni_model
    del pytorch_model
    if hf_loaded:
        del hf_model
    gc.collect()

    # ========================================================================
    # Print Results Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'TTFT (ms)':<12} {'Tok/s':<10} {'Memory (MB)':<14} {'Perplexity':<12}")
    print("-" * 75)
    for r in results:
        ppl_str = f"{r.get('perplexity', float('nan')):.2f}"
        print(f"{r['name']:<25} {r['mean_ttft_ms']:<12.1f} {r['tokens_per_sec']:<10.2f} {r['memory_mb']:<14.1f} {ppl_str:<12}")

    # Speedups
    print("\n" + "-" * 75)
    pytorch_ttft = results[1]['mean_ttft_ms']
    pytorch_tps = results[1]['tokens_per_sec']
    vnni_ttft = results[0]['mean_ttft_ms']
    vnni_tps = results[0]['tokens_per_sec']

    print(f"\nSpeedup vs PyTorch (ternary):")
    print(f"  TTFT: {pytorch_ttft / vnni_ttft:.2f}x")
    print(f"  Throughput: {vnni_tps / pytorch_tps:.2f}x")

    if len(results) > 2:
        hf_ttft = results[2]['mean_ttft_ms']
        hf_tps = results[2]['tokens_per_sec']
        print(f"\nSpeedup vs HuggingFace (official):")
        print(f"  TTFT: {hf_ttft / vnni_ttft:.2f}x")
        print(f"  Throughput: {vnni_tps / hf_tps:.2f}x")

    # Memory comparison
    print(f"\nMemory Usage:")
    print(f"  VNNI Kernels: {results[0]['memory_mb']:.1f} MB")
    print(f"  PyTorch: {results[1]['memory_mb']:.1f} MB")
    if len(results) > 2:
        print(f"  HuggingFace: {results[2]['memory_mb']:.1f} MB")

    # Numerical accuracy summary
    print(f"\nNumerical Accuracy:")
    print(f"  Max logit difference: {accuracy['max_abs_diff']:.6f}")
    print(f"  Top-1 token agreement: {100*accuracy['token_agreement']/accuracy['total_tokens']:.1f}%")

    # Sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (quality check)")
    print("=" * 70)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  {r['sample_output']}")

    # ========================================================================
    # Save Results and Generate Graphs
    # ========================================================================
    print("\n" + "=" * 70)
    print("Saving Results and Generating Graphs")
    print("=" * 70)

    # Save results to JSON and text summary
    save_results(results, kernel_results, accuracy, arch, output_dir)

    # Create graphs
    create_comparison_graphs(results, kernel_results, output_dir, arch)

    # List all generated files
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
