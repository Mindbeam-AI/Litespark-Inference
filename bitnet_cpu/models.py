"""
BitNet model loading and inference.

Supports Microsoft BitNet b1.58 2B-4T model with optimized CPU kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import os

from .arch import get_kernel, get_kernel_type, get_k_padding


# Available models
AVAILABLE_MODELS = {
    'bitnet-2b': {
        'hf_name': 'microsoft/bitnet-b1.58-2B-4T-bf16',
        'description': 'BitNet b1.58 2B parameters, 4T tokens trained',
        'size_mb': 556,
    },
}


def list_available_models() -> Dict:
    """List available models."""
    return AVAILABLE_MODELS


def get_model_info(model_name: str) -> Dict:
    """Get information about a model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name]


# ============================================================================
# Weight Preparation
# ============================================================================

def quantize_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float activation to int8 with per-row scaling."""
    if x.dim() == 1:
        x = x.unsqueeze(0)

    x_flat = x.view(-1, x.shape[-1])
    x_abs_max = x_flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
    x_scale = x_abs_max / 127.0

    x_int8 = (x_flat / x_scale).round().clamp(-127, 127).to(torch.int8)
    x_scale = x_scale.squeeze(1).float()

    return x_int8, x_scale


def prepare_ternary_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Prepare ternary weight for kernel: quantize and pad K dimension."""
    # Extract ternary values and scale
    w = weight.float()
    w_abs = w.abs()
    scale = w_abs.max().item()

    if scale > 0:
        w_ternary = (w / scale).round().clamp(-1, 1)
    else:
        w_ternary = torch.zeros_like(w)
        scale = 1.0

    # Convert to int8
    w_int8 = w_ternary.to(torch.int8)

    # Pad K dimension for SIMD alignment
    N, K = w_int8.shape
    k_pad = get_k_padding()
    K_padded = ((K + k_pad - 1) // k_pad) * k_pad

    if K_padded > K:
        padding = torch.zeros(N, K_padded - K, dtype=torch.int8)
        w_int8 = torch.cat([w_int8, padding], dim=1)

    # Pre-compute column sums for zero-point correction
    w_sum = w_int8.sum(dim=1).float()

    return w_int8.contiguous(), w_sum.contiguous(), scale


# ============================================================================
# BitNet Linear Layer
# ============================================================================

class BitNetLinear(nn.Module):
    """BitNet linear layer with optimized CPU kernels."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        num_threads: int = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k_pad = get_k_padding()
        self.K_padded = ((in_features + k_pad - 1) // k_pad) * k_pad
        self.num_threads = num_threads or torch.get_num_threads()

        self.register_buffer('w_int8', None)
        self.register_buffer('w_sum', None)
        self.register_buffer('bias_tensor', None)
        self.scale = 1.0

        # Use v4_large_n for large output dimensions
        self.use_v4_large_n = out_features >= 1024

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

        kernel = get_kernel()
        kernel_type = get_kernel_type()

        if kernel_type == 'vnni':
            # x86_64 AVX-512 VNNI
            x_int8, x_scale = quantize_activation(x)

            if self.use_v4_large_n:
                y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                kernel.matmul_free_vnni_v4_large_n(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y_int32,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
            else:
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_vnni_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y * self.scale

            if self.use_v4_large_n and self.bias_tensor is not None:
                y = y + self.bias_tensor

        elif kernel_type == 'avx_vnni':
            # Intel Core Ultra with AVX-VNNI (256-bit)
            x_int8, x_scale = quantize_activation(x)

            if self.use_v4_large_n:
                y_int32 = torch.zeros(M, self.out_features, dtype=torch.int32)
                kernel.matmul_free_avx_vnni_v4_large_n(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y_int32,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y_int32.float() * x_scale.unsqueeze(1) * self.scale
            else:
                y = torch.zeros(M, self.out_features, dtype=torch.float32)
                bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()
                kernel.matmul_free_avx_vnni_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.in_features, self.num_threads
                )
                y = y * self.scale

            if self.use_v4_large_n and self.bias_tensor is not None:
                y = y + self.bias_tensor

        elif kernel_type == 'neon':
            # Apple Silicon NEON SDOT
            y = torch.zeros(M, self.out_features, dtype=torch.float32)
            bias = self.bias_tensor if self.bias_tensor is not None else torch.Tensor()

            kernel.matmul_free_neon_sdot_v4_fused(
                x.contiguous(),
                self.w_int8, self.w_sum,
                y, bias,
                self.scale,
                M, self.out_features, self.in_features, self.num_threads
            )
        else:
            raise RuntimeError(f"Unknown kernel type: {kernel_type}")

        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# ============================================================================
# BitNet Model Components
# ============================================================================

class RMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class BitNetAttention(nn.Module):
    """BitNet multi-head attention."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.layer_idx = layer_idx

        self.q_proj = BitNetLinear(self.hidden_size, self.hidden_size)
        self.k_proj = BitNetLinear(self.hidden_size, self.kv_dim)
        self.v_proj = BitNetLinear(self.hidden_size, self.kv_dim)
        self.o_proj = BitNetLinear(self.hidden_size, self.hidden_size)

        self.rotary_emb = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        if self.rotary_emb is not None and position_ids is not None:
            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        # GQA: expand KV heads
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.shape[2], dtype=torch.bool, device=q.device),
                diagonal=k.shape[2] - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(attn_output), new_cache


class BitNetMLP(nn.Module):
    """BitNet MLP with SwiGLU."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = BitNetLinear(self.hidden_size, self.intermediate_size)
        self.up_proj = BitNetLinear(self.hidden_size, self.intermediate_size)
        self.down_proj = BitNetLinear(self.intermediate_size, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BitNetDecoderLayer(nn.Module):
    """BitNet transformer decoder layer."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config, layer_idx)
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states, new_cache = self.attn(
            hidden_states, position_ids, past_key_value, use_cache
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


# ============================================================================
# Rotary Position Embeddings
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""

    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        seq_len = position_ids.max().item() + 1

        if self._cos_cached is None or seq_len > self._cos_cached.shape[0]:
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(x.dtype)
            self._sin_cached = emb.sin().to(x.dtype)

        return (
            self._cos_cached[position_ids].unsqueeze(1),
            self._sin_cached[position_ids].unsqueeze(1),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# BitNet Model
# ============================================================================

@dataclass
class BitNetConfig:
    """BitNet model configuration."""
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 4096


class BitNetModel(nn.Module):
    """BitNet b1.58 model for CPU inference."""

    def __init__(self, config: BitNetConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            BitNetDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Setup rotary embeddings
        head_dim = config.hidden_size // config.num_attention_heads
        for layer in self.layers:
            layer.attn.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, seq_len = input_ids.shape

        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_len = past_key_values[0][0].shape[2]
            else:
                past_len = 0
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)

        hidden_states = self.embed_tokens(input_ids)

        new_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, layer_cache = layer(
                hidden_states, position_ids, past_kv, use_cache
            )
            if use_cache:
                new_cache.append(layer_cache)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                curr_input = generated
            else:
                curr_input = generated[:, -1:]

            logits, past_key_values = self(
                curr_input,
                past_key_values=past_key_values,
                use_cache=True
            )

            next_logits = logits[:, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature

                if top_k > 0:
                    topk_logits, topk_indices = torch.topk(next_logits, top_k)
                    next_logits = torch.full_like(next_logits, float('-inf'))
                    next_logits.scatter_(1, topk_indices, topk_logits)

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits = next_logits.masked_fill(indices_to_remove, float('-inf'))

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop on EOS
            if next_token.item() in [151643, 151645]:  # <|endoftext|>, <|im_end|>
                break

        return generated


# ============================================================================
# Model Loading
# ============================================================================

def load_model(model_name: str = 'bitnet-2b', device: str = 'cpu'):
    """
    Load a BitNet model with optimized CPU kernels.

    Args:
        model_name: Model identifier ('bitnet-2b')
        device: Device to load model on (only 'cpu' supported)

    Returns:
        (model, tokenizer) tuple
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

    model_info = AVAILABLE_MODELS[model_name]
    hf_name = model_info['hf_name']

    print(f"Loading {model_name} from {hf_name}...")

    # Import transformers for tokenizer and weights
    try:
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install: pip install transformers safetensors huggingface_hub")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Create model
    config = BitNetConfig()
    model = BitNetModel(config)

    # Download and load weights
    print("Downloading model weights...")
    weight_file = hf_hub_download(repo_id=hf_name, filename="model.safetensors")
    state_dict = load_file(weight_file)

    # Load embedding and lm_head
    model.embed_tokens.weight.data = state_dict['model.embed_tokens.weight'].float()
    if 'lm_head.weight' in state_dict:
        model.lm_head.weight.data = state_dict['lm_head.weight'].float()
    else:
        model.lm_head.weight = model.embed_tokens.weight

    # Load norm
    model.norm.weight.data = state_dict['model.norm.weight'].float()

    # Load layers
    print("Loading layers with optimized kernels...")
    for i in range(config.num_hidden_layers):
        prefix = f'model.layers.{i}.'

        # Norms
        model.layers[i].input_norm.weight.data = state_dict[f'{prefix}input_layernorm.weight'].float()
        model.layers[i].post_attn_norm.weight.data = state_dict[f'{prefix}post_attention_layernorm.weight'].float()

        # Attention projections
        model.layers[i].attn.q_proj.load_from_weight(state_dict[f'{prefix}self_attn.q_proj.weight'])
        model.layers[i].attn.k_proj.load_from_weight(state_dict[f'{prefix}self_attn.k_proj.weight'])
        model.layers[i].attn.v_proj.load_from_weight(state_dict[f'{prefix}self_attn.v_proj.weight'])
        model.layers[i].attn.o_proj.load_from_weight(state_dict[f'{prefix}self_attn.o_proj.weight'])

        # MLP projections
        model.layers[i].mlp.gate_proj.load_from_weight(state_dict[f'{prefix}mlp.gate_proj.weight'])
        model.layers[i].mlp.up_proj.load_from_weight(state_dict[f'{prefix}mlp.up_proj.weight'])
        model.layers[i].mlp.down_proj.load_from_weight(state_dict[f'{prefix}mlp.down_proj.weight'])

    model.eval()
    print(f"Model loaded! Using {get_kernel_type()} kernel")

    return model, tokenizer


def generate(
    model: BitNetModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> str:
    """
    Generate text from a prompt.

    Args:
        model: Loaded BitNet model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling threshold

    Returns:
        Generated text (including prompt)
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
