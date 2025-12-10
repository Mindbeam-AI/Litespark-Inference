#!/usr/bin/env python3
"""
Benchmark comparison: VNNI kernels vs PyTorch baseline vs Microsoft bf16

Compares:
1. Our VNNI/Graviton kernels (ternary quantized)
2. PyTorch baseline (ternary weights with torch.matmul)
3. Microsoft bf16 original (full precision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.ternary_models import (
    BitNet, BitNetConfig, load_ternary_model, get_arch_info,
    prepare_ternary_weight
)


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
# BF16 Baseline (Microsoft original weights, no quantization)
# ============================================================================

class BF16BitNetAttention(nn.Module):
    """BitNet attention with bf16 weights (no quantization)."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        kv_dim = config.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.attn_sub_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.scale = 1.0 / (self.head_dim ** 0.5)

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


class BF16BitNetMLP(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.ffn_sub_norm = nn.RMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = relu2(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.ffn_sub_norm(hidden)
        return self.down_proj(hidden)


class BF16BitNetBlock(nn.Module):
    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BF16BitNetAttention(config)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BF16BitNetMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class BF16BitNet(nn.Module):
    """BitNet with original bf16 weights (no quantization)."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BF16BitNetBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)

    @classmethod
    def from_safetensors(cls, model_name: str):
        """Load bf16 weights directly (no quantization)."""
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

                layer.attn.q_proj.weight.data = f.get_tensor(f'{prefix}.self_attn.q_proj.weight').float()
                layer.attn.k_proj.weight.data = f.get_tensor(f'{prefix}.self_attn.k_proj.weight').float()
                layer.attn.v_proj.weight.data = f.get_tensor(f'{prefix}.self_attn.v_proj.weight').float()
                layer.attn.o_proj.weight.data = f.get_tensor(f'{prefix}.self_attn.o_proj.weight').float()
                layer.attn.attn_sub_norm.weight.data = f.get_tensor(f'{prefix}.self_attn.attn_sub_norm.weight').float()

                layer.mlp.gate_proj.weight.data = f.get_tensor(f'{prefix}.mlp.gate_proj.weight').float()
                layer.mlp.up_proj.weight.data = f.get_tensor(f'{prefix}.mlp.up_proj.weight').float()
                layer.mlp.down_proj.weight.data = f.get_tensor(f'{prefix}.mlp.down_proj.weight').float()
                layer.mlp.ffn_sub_norm.weight.data = f.get_tensor(f'{prefix}.mlp.ffn_sub_norm.weight').float()

                layer.input_norm.weight.data = f.get_tensor(f'{prefix}.input_layernorm.weight').float()
                layer.post_attn_norm.weight.data = f.get_tensor(f'{prefix}.post_attention_layernorm.weight').float()

            model.norm.weight.data = f.get_tensor('model.norm.weight').float()

        return model


# ============================================================================
# Benchmark Functions
# ============================================================================

def benchmark_model(model, tokenizer, name: str, num_runs: int = 5, gen_tokens: int = 20):
    """Benchmark a single model."""
    model.eval()

    prompts = [
        "The future of AI is",
        "Once upon a time",
    ]

    # Warmup
    print(f"  Warming up {name}...")
    for _ in range(2):
        input_ids = tokenizer.encode(prompts[0], return_tensors='pt')
        with torch.no_grad():
            _ = model(input_ids)

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
    }


def main():
    print("=" * 70)
    print("BitNet Benchmark: VNNI Kernels vs PyTorch vs BF16")
    print("=" * 70)

    arch = get_arch_info()
    print(f"Platform: {arch['machine']}")
    print(f"Kernel: {arch['kernel_type']}")
    print(f"Threads: {torch.get_num_threads()}")
    print()

    MODEL_NAME = 'microsoft/bitnet-b1.58-2B-4T-bf16'

    # Load tokenizer
    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = []

    # 1. VNNI/Graviton kernels (ternary)
    print("\n" + "=" * 70)
    print("1. Loading VNNI/Graviton kernel model (ternary)...")
    print("=" * 70)
    vnni_model, _ = load_ternary_model('bitnet-2b')
    vnni_model.eval()
    results.append(benchmark_model(vnni_model, tokenizer, "VNNI Kernels (ternary)"))
    del vnni_model

    # 2. PyTorch baseline (ternary)
    print("\n" + "=" * 70)
    print("2. Loading PyTorch baseline model (ternary)...")
    print("=" * 70)
    pytorch_model = PyTorchBitNet.from_safetensors(MODEL_NAME)
    pytorch_model.eval()
    results.append(benchmark_model(pytorch_model, tokenizer, "PyTorch (ternary)"))
    del pytorch_model

    # 3. BF16 baseline (no quantization)
    print("\n" + "=" * 70)
    print("3. Loading BF16 baseline model (full precision)...")
    print("=" * 70)
    bf16_model = BF16BitNet.from_safetensors(MODEL_NAME)
    bf16_model.eval()
    results.append(benchmark_model(bf16_model, tokenizer, "BF16 (full precision)"))
    del bf16_model

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Model':<25} {'TTFT (ms)':<12} {'Gen (ms)':<12} {'Tok/s':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['mean_ttft_ms']:<12.1f} {r['mean_gen_time_ms']:<12.1f} {r['tokens_per_sec']:<10.2f}")

    # Speedups
    print("\n" + "-" * 60)
    pytorch_ttft = results[1]['mean_ttft_ms']
    pytorch_tps = results[1]['tokens_per_sec']
    bf16_ttft = results[2]['mean_ttft_ms']
    bf16_tps = results[2]['tokens_per_sec']
    vnni_ttft = results[0]['mean_ttft_ms']
    vnni_tps = results[0]['tokens_per_sec']

    print(f"\nSpeedup vs PyTorch (ternary):")
    print(f"  TTFT: {pytorch_ttft / vnni_ttft:.2f}x")
    print(f"  Throughput: {vnni_tps / pytorch_tps:.2f}x")

    print(f"\nSpeedup vs BF16 (full precision):")
    print(f"  TTFT: {bf16_ttft / vnni_ttft:.2f}x")
    print(f"  Throughput: {vnni_tps / bf16_tps:.2f}x")

    # Sample outputs
    print("\n" + "=" * 70)
    print("Sample Outputs (quality check)")
    print("=" * 70)
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  {r['sample_output']}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")


if __name__ == '__main__':
    main()
