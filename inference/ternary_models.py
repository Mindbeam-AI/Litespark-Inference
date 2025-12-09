#!/usr/bin/env python3
"""
Ternary Model Loaders

Supports loading pre-trained ternary models:
1. MatMul-free LM (ridger/MMfreeLM-370M, 1.3B, 2.7B)
2. BitNet b1.58 (1bitLLM/bitnet_b1_58-3B, microsoft/bitnet-b1.58-2B-4T)
3. HGRN-Bit (same as MatMul-free LM architecture)

Supported architectures:
- x86_64 with AVX-512 VNNI (Intel Ice Lake+, AMD Zen4+)
- aarch64/arm64 with NEON SDOT (AWS Graviton 2/3/4, Apple Silicon)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from torch.utils.cpp_extension import load
from dataclasses import dataclass
import json


# ============================================================================
# Architecture Detection
# ============================================================================

def is_graviton() -> bool:
    """Detect if running on AWS Graviton."""
    try:
        # Check MIDR register for ARM implementer
        midr_path = '/sys/devices/system/cpu/cpu0/regs/identification/midr_el1'
        if os.path.exists(midr_path):
            with open(midr_path) as f:
                midr = int(f.read().strip(), 16)
                # Graviton uses ARM implementer (0x41)
                return (midr >> 24) == 0x41
    except:
        pass

    # Alternative: check /proc/cpuinfo
    try:
        with open('/proc/cpuinfo') as f:
            cpuinfo = f.read().lower()
            # Graviton 2 reports neoverse-n1, Graviton 3/4 reports neoverse-v1
            if 'neoverse' in cpuinfo:
                return True
    except:
        pass

    return False


def get_arch_info() -> Dict:
    """Get architecture information."""
    machine = platform.machine().lower()

    info = {
        'machine': machine,
        'is_x86_64': machine in ['x86_64', 'amd64'],
        'is_arm64': machine in ['aarch64', 'arm64'],
        'is_graviton': False,
        'is_apple_silicon': False,
        'kernel_type': None,
    }

    if info['is_x86_64']:
        info['kernel_type'] = 'vnni'
    elif info['is_arm64']:
        info['is_graviton'] = is_graviton()
        info['is_apple_silicon'] = platform.system() == 'Darwin'
        info['kernel_type'] = 'graviton' if info['is_graviton'] else 'neon'

    return info


# ============================================================================
# Load Architecture-Specific Kernel
# ============================================================================

_kernel = None
_kernel_type = None

def get_kernel():
    """Load the appropriate kernel for this architecture (cached)."""
    global _kernel, _kernel_type

    if _kernel is None:
        arch_info = get_arch_info()
        machine = arch_info['machine']

        if arch_info['is_x86_64']:
            # x86_64 with AVX-512 VNNI
            kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp'

            _kernel = load(
                name='matmul_free_vnni',
                sources=[str(kernel_path)],
                extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
                extra_ldflags=['-fopenmp'],
                verbose=False
            )
            _kernel_type = 'vnni'

        elif arch_info['is_arm64']:
            if arch_info['is_graviton']:
                # AWS Graviton with NEON SDOT
                kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/arm64/matmul_free_graviton.cpp'

                _kernel = load(
                    name='matmul_free_graviton',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-fopenmp'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'graviton'
            else:
                # Apple Silicon or other ARM64 with NEON SDOT
                kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/arm64/matmul_free_neon_int8.cpp'

                _kernel = load(
                    name='matmul_free_neon',
                    sources=[str(kernel_path)],
                    extra_cflags=['-O3', '-march=native', '-fopenmp'],
                    extra_ldflags=['-fopenmp'],
                    verbose=False
                )
                _kernel_type = 'neon'
        else:
            raise RuntimeError(f"Unsupported architecture: {machine}")

    return _kernel


def get_kernel_type() -> str:
    """Get the kernel type being used."""
    global _kernel_type
    if _kernel_type is None:
        get_kernel()  # This will set _kernel_type
    return _kernel_type


# ============================================================================
# Weight Preparation
# ============================================================================

def get_k_padding() -> int:
    """Get K padding alignment based on architecture."""
    kernel_type = get_kernel_type()
    if kernel_type == 'vnni':
        return 64  # AVX-512 = 512 bits = 64 bytes
    else:
        return 16  # NEON = 128 bits = 16 bytes


def quantize_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float activation to int8 with per-row scaling."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    M, K = x.shape
    k_pad = get_k_padding()
    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    x_int8 = torch.zeros(M, K_padded, dtype=torch.int8)
    x_int8[:, :K] = (x / scale).round().clamp(-127, 127).to(torch.int8)

    return x_int8.contiguous(), scale.squeeze(1).contiguous()


def prepare_ternary_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Prepare weight tensor for kernel.

    Args:
        weight: Float tensor [out_features, in_features] with values ~{-1, 0, +1}

    Returns:
        w_int8: Padded int8 tensor [N, K_padded]
        w_sum: Row sums as int32 [N]
        scale: Scale factor for the weights
    """
    N, K = weight.shape
    k_pad = get_k_padding()
    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

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
    """Ternary linear layer with architecture-adaptive kernel selection."""

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
        k_pad = get_k_padding()
        self.K_padded = ((in_features + k_pad - 1) // k_pad) * k_pad
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
        """Forward pass using architecture-appropriate kernel."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        # Quantize input
        x_int8, x_scale = quantize_activation(x)

        # Allocate output
        y = torch.zeros(M, self.out_features, dtype=torch.float32)
        bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

        # Select kernel based on architecture and dimensions
        kernel = get_kernel()
        kernel_type = get_kernel_type()

        if kernel_type == 'vnni':
            # x86_64 AVX-512 VNNI
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
        elif kernel_type == 'graviton':
            # AWS Graviton NEON SDOT
            if self.in_features <= 1024 and self.out_features <= 4096:
                kernel.matmul_free_graviton_v2(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
            else:
                kernel.matmul_free_graviton_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
        elif kernel_type == 'neon':
            # Apple Silicon / Generic ARM NEON SDOT
            if self.in_features <= 1024 and self.out_features <= 4096:
                kernel.matmul_free_neon_sdot_v2(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
            else:
                kernel.matmul_free_neon_sdot_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
        else:
            raise RuntimeError(f"Unknown kernel type: {kernel_type}")

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
    from transformers import AutoTokenizer, LlamaTokenizer

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")

    hf_name, model_class = AVAILABLE_MODELS[model_key]

    print(f"Loading model: {model_key} ({hf_name})")
    model = model_class.from_pretrained(hf_name, num_threads)
    model.eval()

    print("Loading tokenizer...")
    # BitNet uses a custom tokenizer class that may not load with AutoTokenizer
    # Use LlamaTokenizer directly with the tokenizer.model file
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    except (ValueError, OSError, ImportError) as e:
        print(f"  AutoTokenizer failed: {e}")
        print("  Loading tokenizer.model directly with LlamaTokenizer...")
        try:
            # BitNet tokenizer is sentencepiece-based, compatible with LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(hf_name)
        except Exception as e2:
            print(f"  LlamaTokenizer failed: {e2}")
            # Last resort: use the tokenizer.json with a generic tokenizer
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_name)

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
