#!/usr/bin/env python3
"""
GPT-2 Inference with Ternary MatMul Kernels

Replaces all linear layers with our VNNI ternary matmul for inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from pathlib import Path
from typing import Optional, Tuple, Dict
from torch.utils.cpp_extension import load

from ternarize_model import (
    TernaryLinear,
    ternarize_gpt2,
    load_ternary_weights
)


# ============================================================================
# Load VNNI Kernel
# ============================================================================

def load_kernel():
    """Load the appropriate kernel for this architecture."""
    machine = platform.machine().lower()

    if machine not in ['x86_64', 'amd64']:
        raise RuntimeError(f"Ternary inference requires x86_64. Got: {machine}")

    kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp'

    kernel = load(
        name='matmul_free_vnni',
        sources=[str(kernel_path)],
        extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
        extra_ldflags=['-fopenmp'],
        verbose=False
    )

    return kernel


# ============================================================================
# Input Quantization
# ============================================================================

def quantize_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize float activation to int8 with per-row scaling.

    Args:
        x: Float tensor [M, K]

    Returns:
        x_int8: Int8 tensor [M, K_padded]
        scale: Float tensor [M]
    """
    M, K = x.shape
    K_padded = ((K + 63) // 64) * 64

    # Per-row max abs
    max_abs = x.abs().max(dim=1, keepdim=True).values
    scale = max_abs / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    # Quantize
    x_int8 = torch.zeros(M, K_padded, dtype=torch.int8, device=x.device)
    x_int8[:, :K] = (x / scale).round().clamp(-127, 127).to(torch.int8)

    return x_int8.contiguous(), scale.squeeze(1).contiguous()


# ============================================================================
# Ternary Linear Helper
# ============================================================================

def ternary_linear(x: torch.Tensor, layer: TernaryLinear, kernel, num_threads: int) -> torch.Tensor:
    """Apply ternary linear with our kernel. Uses v2 for small dims, v3 for large."""
    M = x.shape[0]
    N = layer.out_features
    K = layer.in_features

    # Quantize input
    x_int8, x_scale = quantize_activation(x)

    # Allocate output
    y = torch.zeros(M, N, dtype=torch.float32)
    bias = layer.bias if layer.bias is not None else torch.Tensor()

    # Use v2 for small dimensions (less overhead), v3 for large
    if K <= 1024 and N <= 4096:
        kernel.matmul_free_vnni_v2(
            x_int8, x_scale,
            layer.w_int8, layer.w_sum,
            y, bias,
            M, N, K, num_threads
        )
    else:
        kernel.matmul_free_vnni_v3(
            x_int8, x_scale,
            layer.w_int8, layer.w_sum,
            y, bias,
            M, N, K, num_threads
        )

    # Apply ternary scale
    return y * layer.scale


# ============================================================================
# Ternary GPT-2 Model
# ============================================================================

class TernaryGPT2Attention(nn.Module):
    """GPT-2 attention using ternary matmul."""

    def __init__(
        self,
        config,
        c_attn_ternary: TernaryLinear,
        c_proj_ternary: TernaryLinear,
        kernel,
        num_threads: int
    ):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = c_attn_ternary
        self.c_proj = c_proj_ternary
        self.kernel = kernel
        self.num_threads = num_threads

        # Scaling factor for attention
        self.scale = 1.0 / (self.head_dim ** 0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
            attention_mask: Optional mask
        """
        batch, seq_len, hidden = hidden_states.shape

        # Reshape for matmul: [batch * seq_len, hidden]
        x = hidden_states.view(-1, hidden)

        # QKV projection: [M, 3*hidden]
        qkv = ternary_linear(x, self.c_attn, self.kernel, self.num_threads)

        # Split Q, K, V
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Reshape to [batch, n_head, seq_len, head_dim]
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: [batch, n_head, seq_len, seq_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [batch, seq_len, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)

        # Output projection
        x = attn_output.view(-1, hidden)
        output = ternary_linear(x, self.c_proj, self.kernel, self.num_threads)
        output = output.view(batch, seq_len, hidden)

        return output


class TernaryGPT2MLP(nn.Module):
    """GPT-2 MLP using ternary matmul."""

    def __init__(
        self,
        config,
        c_fc_ternary: TernaryLinear,
        c_proj_ternary: TernaryLinear,
        kernel,
        num_threads: int
    ):
        super().__init__()
        self.c_fc = c_fc_ternary
        self.c_proj = c_proj_ternary
        self.kernel = kernel
        self.num_threads = num_threads

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = hidden_states.shape

        x = hidden_states.view(-1, hidden)

        # Up projection
        h = ternary_linear(x, self.c_fc, self.kernel, self.num_threads)

        # GELU activation
        h = F.gelu(h)

        # Down projection
        output = ternary_linear(h, self.c_proj, self.kernel, self.num_threads)

        return output.view(batch, seq_len, hidden)


class TernaryGPT2Block(nn.Module):
    """Single GPT-2 transformer block with ternary matmul."""

    def __init__(
        self,
        config,
        block_idx: int,
        ternary_layers: Dict[str, TernaryLinear],
        original_block: nn.Module,
        kernel,
        num_threads: int
    ):
        super().__init__()

        # Layer norms from original model
        self.ln_1 = original_block.ln_1
        self.ln_2 = original_block.ln_2

        # Ternary attention
        self.attn = TernaryGPT2Attention(
            config,
            ternary_layers[f"h.{block_idx}.attn.c_attn"],
            ternary_layers[f"h.{block_idx}.attn.c_proj"],
            kernel,
            num_threads
        )

        # Ternary MLP
        self.mlp = TernaryGPT2MLP(
            config,
            ternary_layers[f"h.{block_idx}.mlp.c_fc"],
            ternary_layers[f"h.{block_idx}.mlp.c_proj"],
            kernel,
            num_threads
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention with residual
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask)
        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class TernaryGPT2(nn.Module):
    """GPT-2 model with ternary matmul for all linear layers."""

    def __init__(
        self,
        ternary_layers: Dict[str, TernaryLinear],
        original_model,
        num_threads: int = None
    ):
        super().__init__()

        self.config = original_model.config
        self.num_threads = num_threads or torch.get_num_threads()

        # Load kernel
        print("Loading VNNI kernel...")
        self.kernel = load_kernel()

        # Embeddings from original model (not ternarized)
        self.wte = original_model.transformer.wte
        self.wpe = original_model.transformer.wpe

        # Final layer norm
        self.ln_f = original_model.transformer.ln_f

        # LM head (tied to wte, not ternarized)
        self.lm_head = original_model.lm_head

        # Ternary transformer blocks
        self.blocks = nn.ModuleList([
            TernaryGPT2Block(
                self.config,
                i,
                ternary_layers,
                original_model.transformer.h[i],
                self.kernel,
                self.num_threads
            )
            for i in range(self.config.n_layer)
        ])

        print(f"Initialized TernaryGPT2 with {self.config.n_layer} layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = input_ids.shape

        # Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # LM head
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Simple greedy/sampling generation."""
        for _ in range(max_new_tokens):
            # Get logits for last position
            logits = self.forward(input_ids)[:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_ternary_gpt2(
    model_name: str = 'gpt2',
    method: str = 'threshold',
    threshold_ratio: float = 0.05,
    num_threads: int = None
) -> TernaryGPT2:
    """
    Create a ternary GPT-2 model.

    Args:
        model_name: HuggingFace model name
        method: Ternarization method
        threshold_ratio: For threshold method
        num_threads: Number of threads for kernel

    Returns:
        TernaryGPT2 model ready for inference
    """
    # Ternarize weights
    ternary_layers, original_model = ternarize_gpt2(
        model_name=model_name,
        method=method,
        threshold_ratio=threshold_ratio
    )

    # Create ternary model
    model = TernaryGPT2(ternary_layers, original_model, num_threads)

    return model


if __name__ == '__main__':
    from transformers import GPT2Tokenizer

    print("Creating ternary GPT-2...")
    model = create_ternary_gpt2('gpt2', method='threshold', threshold_ratio=0.05)

    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Test generation
    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print("Generating...")
    output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nOutput: {output_text}")
