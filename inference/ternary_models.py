#!/usr/bin/env python3
"""
Ternary Model Loaders

Supports loading pre-trained ternary models:
1. MatMul-free LM (ridger/MMfreeLM-370M, 1.3B, 2.7B)
2. BitNet b1.58 (1bitLLM/bitnet_b1_58-3B, microsoft/bitnet-b1.58-2B-4T)
3. HGRN-Bit (same as MatMul-free LM architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from torch.utils.cpp_extension import load
from dataclasses import dataclass
import json


# ============================================================================
# Load VNNI Kernel
# ============================================================================

_kernel = None

def get_kernel():
    """Load the VNNI kernel (cached)."""
    global _kernel
    if _kernel is None:
        machine = platform.machine().lower()
        if machine not in ['x86_64', 'amd64']:
            raise RuntimeError(f"Ternary inference requires x86_64. Got: {machine}")

        kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp'

        _kernel = load(
            name='matmul_free_vnni',
            sources=[str(kernel_path)],
            extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
            extra_ldflags=['-fopenmp'],
            verbose=False
        )
    return _kernel


# ============================================================================
# Weight Preparation
# ============================================================================

def quantize_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float activation to int8 with per-row scaling."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    M, K = x.shape
    K_padded = ((K + 63) // 64) * 64

    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_int8 = torch.zeros(M, K_padded, dtype=torch.int8)
    x_int8[:, :K] = (x / scale).round().clamp(-127, 127).to(torch.int8)

    return x_int8.contiguous(), scale.squeeze(1).contiguous()


def prepare_ternary_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Prepare weight tensor for VNNI kernel.

    Args:
        weight: Float tensor [out_features, in_features] with values ~{-1, 0, +1}

    Returns:
        w_int8: Padded int8 tensor [N, K_padded]
        w_sum: Row sums as int32 [N]
        scale: Scale factor for the weights
    """
    N, K = weight.shape
    K_padded = ((K + 63) // 64) * 64

    # Convert to ternary int8
    # For properly trained ternary models, weights should already be ~{-1, 0, +1}
    w_ternary = weight.round().clamp(-1, 1).to(torch.int8)

    # Compute scale from original weights
    mask = w_ternary != 0
    if mask.any():
        scale = weight[mask].abs().mean().item()
    else:
        scale = 1.0

    # Pad
    w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
    w_int8[:, :K] = w_ternary

    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return w_int8.contiguous(), w_sum.contiguous(), scale


# ============================================================================
# Ternary Linear Layer
# ============================================================================

class TernaryLinear(nn.Module):
    """Ternary linear layer using VNNI kernel."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_threads: int = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K_padded = ((in_features + 63) // 64) * 64
        self.num_threads = num_threads or torch.get_num_threads()

        # Will be set by load_from_weight
        self.register_buffer('w_int8', None)
        self.register_buffer('w_sum', None)
        self.register_buffer('bias_tensor', None)
        self.scale = 1.0

    def load_from_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Load from a float weight tensor."""
        self.w_int8, self.w_sum, self.scale = prepare_ternary_weight(weight)
        if bias is not None:
            self.bias_tensor = bias.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using VNNI kernel."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        # Quantize input
        x_int8, x_scale = quantize_activation(x)

        # Allocate output
        y = torch.zeros(M, self.out_features, dtype=torch.float32)
        bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

        # Select kernel based on dimensions
        kernel = get_kernel()
        if self.in_features <= 1024 and self.out_features <= 4096:
            kernel.matmul_free_vnni_v2(
                x_int8, x_scale,
                self.w_int8, self.w_sum,
                y, bias,
                M, self.out_features, self.in_features, self.num_threads
            )
        else:
            kernel.matmul_free_vnni_v3(
                x_int8, x_scale,
                self.w_int8, self.w_sum,
                y, bias,
                M, self.out_features, self.in_features, self.num_threads
            )

        # Apply scale
        y = y * self.scale

        # Reshape back
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# ============================================================================
# MatMul-free LM Loader
# ============================================================================

@dataclass
class MMFreeLMConfig:
    """Configuration for MatMul-free LM."""
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 1  # HGRN uses single head
    intermediate_size: int = 5632  # ~2.75x hidden
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6


class MMFreeLMAttention(nn.Module):
    """HGRN-style attention for MatMul-free LM."""

    def __init__(self, config: MMFreeLMConfig, num_threads: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_threads = num_threads

        # HGRN uses i, f, g, o projections
        self.i_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.f_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.g_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.o_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # HGRN recurrence (simplified)
        i = torch.sigmoid(self.i_proj(x))
        f = torch.sigmoid(self.f_proj(x))
        g = self.g_proj(x)
        o = torch.sigmoid(self.o_proj(x))

        # Simple gated output (actual HGRN has recurrence)
        return o * torch.tanh(g * i)


class MMFreeLMMLP(nn.Module):
    """MLP for MatMul-free LM."""

    def __init__(self, config: MMFreeLMConfig, num_threads: int):
        super().__init__()
        self.gate_proj = TernaryLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.up_proj = TernaryLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.down_proj = TernaryLinear(config.intermediate_size, config.hidden_size, num_threads=num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MMFreeLMBlock(nn.Module):
    """Single block of MatMul-free LM."""

    def __init__(self, config: MMFreeLMConfig, num_threads: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MMFreeLMAttention(config, num_threads)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MMFreeLMMLP(config, num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class MMFreeLM(nn.Module):
    """MatMul-free Language Model with VNNI kernels."""

    def __init__(self, config: MMFreeLMConfig, num_threads: int = None):
        super().__init__()
        self.config = config
        self.num_threads = num_threads or torch.get_num_threads()

        # Embeddings (kept as float)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.layers = nn.ModuleList([
            MMFreeLMBlock(config, self.num_threads)
            for _ in range(config.num_layers)
        ])

        # Output
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_name: str, num_threads: int = None):
        """Load from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading {model_name}...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Map HF config to our config
        config = MMFreeLMConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            intermediate_size=getattr(hf_config, 'intermediate_size', hf_config.hidden_size * 4),
            vocab_size=hf_config.vocab_size,
            rms_norm_eps=getattr(hf_config, 'rms_norm_eps', 1e-6),
        )

        model = cls(config, num_threads)

        # Copy weights
        print("Converting weights to ternary format...")
        model._load_hf_weights(hf_model)

        return model

    def _load_hf_weights(self, hf_model):
        """Load weights from HuggingFace model."""
        # Embeddings
        self.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

        # Layers
        for i, layer in enumerate(self.layers):
            hf_layer = hf_model.model.layers[i]

            # Attention
            if hasattr(hf_layer, 'attn'):
                layer.attn.i_proj.load_from_weight(hf_layer.attn.i_proj.weight.data)
                layer.attn.f_proj.load_from_weight(hf_layer.attn.f_proj.weight.data)
                layer.attn.g_proj.load_from_weight(hf_layer.attn.g_proj.weight.data)
                layer.attn.o_proj.load_from_weight(hf_layer.attn.o_proj.weight.data)

            # MLP
            if hasattr(hf_layer, 'mlp'):
                layer.mlp.gate_proj.load_from_weight(hf_layer.mlp.gate_proj.weight.data)
                layer.mlp.up_proj.load_from_weight(hf_layer.mlp.up_proj.weight.data)
                layer.mlp.down_proj.load_from_weight(hf_layer.mlp.down_proj.weight.data)

            # Norms
            layer.attn_norm.weight.data = hf_layer.attn_norm.weight.data.clone()
            layer.mlp_norm.weight.data = hf_layer.mlp_norm.weight.data.clone()

        # Final norm and head
        self.norm.weight.data = hf_model.model.norm.weight.data.clone()
        self.lm_head.weight.data = hf_model.lm_head.weight.data.clone()


# ============================================================================
# BitNet Loader
# ============================================================================

@dataclass
class BitNetConfig:
    """Configuration for BitNet."""
    hidden_size: int = 3200
    num_layers: int = 26
    num_heads: int = 32
    intermediate_size: int = 8640
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6


class BitNetAttention(nn.Module):
    """BitNet attention layer."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_threads = num_threads

        self.q_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.k_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.v_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)
        self.o_proj = TernaryLinear(config.hidden_size, config.hidden_size, num_threads=num_threads)

        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out)


class BitNetMLP(nn.Module):
    """BitNet MLP layer."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.gate_proj = TernaryLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.up_proj = TernaryLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.down_proj = TernaryLinear(config.intermediate_size, config.hidden_size, num_threads=num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class BitNetBlock(nn.Module):
    """Single BitNet transformer block."""

    def __init__(self, config: BitNetConfig, num_threads: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config, num_threads)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config, num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class BitNet(nn.Module):
    """BitNet b1.58 with VNNI kernels."""

    def __init__(self, config: BitNetConfig, num_threads: int = None):
        super().__init__()
        self.config = config
        self.num_threads = num_threads or torch.get_num_threads()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BitNetBlock(config, self.num_threads)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(cls, model_name: str, num_threads: int = None):
        """Load from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoConfig

        print(f"Loading {model_name}...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        config = BitNetConfig(
            hidden_size=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            vocab_size=hf_config.vocab_size,
            rms_norm_eps=getattr(hf_config, 'rms_norm_eps', 1e-6),
        )

        model = cls(config, num_threads)

        print("Converting weights to ternary format...")
        model._load_hf_weights(hf_model)

        return model

    def _load_hf_weights(self, hf_model):
        """Load weights from HuggingFace model."""
        # Embeddings
        self.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

        # Layers
        for i, layer in enumerate(self.layers):
            hf_layer = hf_model.model.layers[i]

            # Attention
            layer.attn.q_proj.load_from_weight(hf_layer.self_attn.q_proj.weight.data)
            layer.attn.k_proj.load_from_weight(hf_layer.self_attn.k_proj.weight.data)
            layer.attn.v_proj.load_from_weight(hf_layer.self_attn.v_proj.weight.data)
            layer.attn.o_proj.load_from_weight(hf_layer.self_attn.o_proj.weight.data)

            # MLP
            layer.mlp.gate_proj.load_from_weight(hf_layer.mlp.gate_proj.weight.data)
            layer.mlp.up_proj.load_from_weight(hf_layer.mlp.up_proj.weight.data)
            layer.mlp.down_proj.load_from_weight(hf_layer.mlp.down_proj.weight.data)

            # Norms
            layer.input_norm.weight.data = hf_layer.input_layernorm.weight.data.clone()
            layer.post_attn_norm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

        # Final norm and head
        self.norm.weight.data = hf_model.model.norm.weight.data.clone()
        if hasattr(hf_model, 'lm_head') and hf_model.lm_head is not None:
            self.lm_head.weight.data = hf_model.lm_head.weight.data.clone()


# ============================================================================
# Model Registry
# ============================================================================

AVAILABLE_MODELS = {
    # MatMul-free LM
    'mmfreelm-370m': ('ridger/MMfreeLM-370M', MMFreeLM),
    'mmfreelm-1.3b': ('ridger/MMfreeLM-1.3B', MMFreeLM),
    'mmfreelm-2.7b': ('ridger/MMfreeLM-2.7B', MMFreeLM),

    # BitNet
    'bitnet-3b': ('1bitLLM/bitnet_b1_58-3B', BitNet),
    'bitnet-large': ('1bitLLM/bitnet_b1_58-large', BitNet),

    # Microsoft BitNet (bf16 weights for loading)
    'bitnet-2b-ms': ('microsoft/bitnet-b1.58-2B-4T-bf16', BitNet),
}


def load_ternary_model(model_key: str, num_threads: int = None):
    """
    Load a pre-trained ternary model.

    Args:
        model_key: One of the keys in AVAILABLE_MODELS
        num_threads: Number of threads for kernel

    Returns:
        model: Loaded model with VNNI kernels
        tokenizer: HuggingFace tokenizer
    """
    from transformers import AutoTokenizer

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

    hf_name, model_class = AVAILABLE_MODELS[model_key]

    print(f"Loading model: {model_key} ({hf_name})")
    model = model_class.from_pretrained(hf_name, num_threads)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def list_models():
    """List available models."""
    print("Available ternary models:")
    print("-" * 60)
    for key, (hf_name, model_class) in AVAILABLE_MODELS.items():
        print(f"  {key:20s} -> {hf_name}")
    print()


if __name__ == '__main__':
    list_models()
