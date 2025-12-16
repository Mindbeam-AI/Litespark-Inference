#!/usr/bin/env python3
"""
W8A8 Quantized Models - Full INT8 Weight and Activation Quantization

This module provides INT8 quantization for both weights and activations,
enabling maximum VNNI throughput.

Compared to ternary (1.58-bit) quantization:
- Better accuracy (8-bit vs 1.58-bit)
- Same or better throughput (full VNNI utilization)
- Larger model size (8x vs 1.58x compression)

Usage:
    from w8a8_models import load_w8a8_model

    model, tokenizer = load_w8a8_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ============================================================================
# Kernel Loading
# ============================================================================

_w8a8_kernel = None
_w8a8_kernel_type = None


def get_w8a8_kernel():
    """Load the W8A8 VNNI kernel."""
    global _w8a8_kernel, _w8a8_kernel_type

    if _w8a8_kernel is not None:
        return _w8a8_kernel, _w8a8_kernel_type

    from torch.utils.cpp_extension import load
    import sys

    machine = platform.machine().lower()
    is_macos = sys.platform == 'darwin'

    if machine in ['x86_64', 'amd64'] and not is_macos:
        kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/x86_64/matmul_w8a8_vnni.cpp'
        try:
            _w8a8_kernel = load(
                name='matmul_w8a8_vnni',
                sources=[str(kernel_path)],
                extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
                extra_ldflags=['-fopenmp'],
                verbose=False
            )
            _w8a8_kernel_type = 'vnni_w8a8'
        except Exception as e:
            print(f"Failed to compile W8A8 kernel: {e}")
            _w8a8_kernel = None
            _w8a8_kernel_type = 'fallback'
    else:
        _w8a8_kernel = None
        _w8a8_kernel_type = 'fallback'

    return _w8a8_kernel, _w8a8_kernel_type


# ============================================================================
# Quantization Statistics
# ============================================================================

@dataclass
class W8A8Stats:
    """Statistics for W8A8 quantization."""
    weight_scale: float
    weight_min: float
    weight_max: float
    mse: float


# ============================================================================
# W8A8 Linear Layer
# ============================================================================

class W8A8Linear(nn.Module):
    """
    Linear layer with INT8 weight and activation quantization.

    Uses VNNI for maximum throughput with 8-bit computation.
    """

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
        self.num_threads = num_threads or torch.get_num_threads()

        # Padded K dimension (multiple of 64 for AVX-512)
        self.K_padded = ((in_features + 63) // 64) * 64

        # INT8 weight storage
        self.register_buffer('w_int8', None)       # [N, K_padded] int8
        self.register_buffer('scale_w', None)      # [N] float32
        self.register_buffer('sum_w', None)        # [N] int32

        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

        # Stats
        self.quant_stats: Optional[W8A8Stats] = None

    def quantize_from_linear(self, linear: nn.Linear):
        """Quantize weights from a regular linear layer."""
        weight = linear.weight.data.float()
        N, K = weight.shape

        # Allocate
        self.w_int8 = torch.zeros(N, self.K_padded, dtype=torch.int8)
        self.scale_w = torch.zeros(N, dtype=torch.float32)
        self.sum_w = torch.zeros(N, dtype=torch.int32)

        # Quantize each row
        for n in range(N):
            w_row = weight[n, :]
            max_abs = w_row.abs().max().item()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            self.scale_w[n] = scale

            w_quant = (w_row / scale).round().clamp(-127, 127).to(torch.int8)
            self.w_int8[n, :K] = w_quant
            self.sum_w[n] = self.w_int8[n, :].sum().to(torch.int32)

        # Calculate reconstruction error
        w_reconstructed = self.w_int8[:, :K].float() * self.scale_w.unsqueeze(1)
        mse = ((weight - w_reconstructed) ** 2).mean().item()

        self.quant_stats = W8A8Stats(
            weight_scale=self.scale_w.mean().item(),
            weight_min=weight.min().item(),
            weight_max=weight.max().item(),
            mse=mse,
        )

        # Copy bias
        if linear.bias is not None:
            self.bias = linear.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with W8A8 quantization."""
        original_shape = x.shape
        x = x.view(-1, self.in_features).float()
        M = x.shape[0]

        kernel, kernel_type = get_w8a8_kernel()

        if kernel_type == 'fallback':
            # PyTorch fallback
            w_float = self.w_int8[:, :self.in_features].float() * self.scale_w.unsqueeze(1)
            y = F.linear(x, w_float, self.bias)
        else:
            # Quantize activations to uint8
            x_uint8 = torch.zeros(M, self.K_padded, dtype=torch.uint8)
            scale_x = torch.zeros(M, dtype=torch.float32)

            for m in range(M):
                x_row = x[m, :]
                max_abs = x_row.abs().max().item()
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                scale_x[m] = scale

                # Quantize and shift to uint8 [1, 255] with zero point 128
                x_quant = (x_row / scale).round().clamp(-127, 127) + 128
                x_uint8[m, :self.in_features] = x_quant.to(torch.uint8)
                x_uint8[m, self.in_features:] = 128  # Zero padding

            # Allocate output
            y = torch.zeros(M, self.out_features, dtype=torch.float32)

            bias = self.bias if self.bias is not None else torch.Tensor()

            # Call kernel
            kernel.matmul_w8a8_vnni_tiled(
                x_uint8, scale_x,
                self.w_int8, self.scale_w, self.sum_w,
                y, bias,
                M, self.out_features, self.K_padded, self.num_threads
            )

        # Reshape output
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)


# ============================================================================
# Model Quantization
# ============================================================================

def quantize_to_w8a8(
    model: nn.Module,
    num_threads: int = None,
    skip_layers: Optional[list] = None,
) -> Dict[str, W8A8Stats]:
    """
    Replace all Linear layers with W8A8Linear.

    Returns:
        Dictionary of layer names to quantization stats
    """
    skip_layers = skip_layers or []
    stats = {}

    def should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not should_skip(full_name):
            # Get parent
            parts = full_name.rsplit('.', 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)

            # Replace
            w8a8_linear = W8A8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                num_threads=num_threads
            )
            w8a8_linear.quantize_from_linear(module)
            setattr(parent, attr_name, w8a8_linear)
            stats[full_name] = w8a8_linear.quant_stats
            print(f"  Quantized {full_name}: MSE={w8a8_linear.quant_stats.mse:.2e}")

    return stats


# ============================================================================
# W8A8 Model Wrapper
# ============================================================================

class W8A8Model(nn.Module):
    """Wrapper for W8A8 quantized HuggingFace model."""

    def __init__(self, hf_model: nn.Module, config: Any):
        super().__init__()
        self.model = hf_model
        self.config = config
        self.quant_stats: Dict[str, W8A8Stats] = {}

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.model(input_ids, **kwargs)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Simple generation loop."""
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.forward(generated)
                next_logits = logits[:, -1, :]

                if temperature != 1.0:
                    next_logits = next_logits / temperature

                if do_sample:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================================
# Main Loading Function
# ============================================================================

def load_w8a8_model(
    model_name: str,
    num_threads: int = None,
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
) -> Tuple[W8A8Model, Any]:
    """
    Load and quantize a HuggingFace model to W8A8.

    Args:
        model_name: HuggingFace model name
        num_threads: Number of threads for kernels
        skip_lm_head: Don't quantize language model head
        skip_embeddings: Don't quantize embeddings

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    print("  Loading weights...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Skip layers
    skip_layers = []
    if skip_lm_head:
        skip_layers.extend(['lm_head', 'output'])
    if skip_embeddings:
        skip_layers.extend(['embed', 'wte', 'wpe'])

    # Quantize
    print("\nQuantizing to W8A8...")
    _, kernel_type = get_w8a8_kernel()
    if kernel_type == 'fallback':
        print("  [Note] Using PyTorch fallback")

    quant_stats = quantize_to_w8a8(hf_model, num_threads, skip_layers)

    # Wrap
    model = W8A8Model(hf_model, config)
    model.quant_stats = quant_stats
    model.eval()

    # Summary
    total_layers = len(quant_stats)
    avg_mse = sum(s.mse for s in quant_stats.values()) / total_layers if total_layers > 0 else 0

    print(f"\nW8A8 Quantization complete:")
    print(f"  Layers quantized: {total_layers}")
    print(f"  Average MSE: {avg_mse:.2e}")

    return model, tokenizer


# ============================================================================
# W8A8 -> Ternary Conversion
# ============================================================================

def load_w8a8_ternary_model(
    model_name: str,
    num_threads: int = None,
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
) -> Tuple[Any, Any]:
    """
    Load W8A8 model and further quantize INT8 weights to ternary {-1, 0, +1}.

    This tests the hypothesis: FP16 -> INT8 -> Ternary vs FP16 -> Ternary directly.

    The INT8 weights (range -127 to +127) are quantized to ternary using
    the absmean threshold method.

    Args:
        model_name: HuggingFace model name
        num_threads: Number of threads for kernels
        skip_lm_head: Don't quantize language model head
        skip_embeddings: Don't quantize embeddings

    Returns:
        (model, tokenizer) tuple - model uses ternary kernels
    """
    from ptq_models import PTQTernaryLinear, PTQModel, get_kernel

    print(f"Loading W8A8 model for ternary conversion: {model_name}")

    # First load the W8A8 model
    w8a8_model, tokenizer = load_w8a8_model(
        model_name,
        num_threads=num_threads,
        skip_lm_head=skip_lm_head,
        skip_embeddings=skip_embeddings,
    )

    print("\nConverting W8A8 (INT8) weights to Ternary...")

    # Now convert each W8A8Linear to PTQTernaryLinear
    converted_count = 0
    for name, module in list(w8a8_model.model.named_modules()):
        if isinstance(module, W8A8Linear):
            # Get the INT8 weights and scale
            w_int8 = module.w_int8  # [N, K_padded] int8
            scale_w = module.scale_w  # [N] per-row scale

            # Reconstruct float weights from INT8
            # w_float[n, k] = w_int8[n, k] * scale_w[n]
            w_float = w_int8[:, :module.in_features].float() * scale_w.unsqueeze(1)

            # Now quantize to ternary using absmean threshold
            # threshold = mean(|w|)
            abs_mean = w_float.abs().mean()
            threshold = abs_mean.item()

            # Ternary quantization: w_ternary = sign(w) * (|w| > threshold)
            w_ternary = torch.zeros_like(w_float, dtype=torch.int8)
            w_ternary[w_float > threshold] = 1
            w_ternary[w_float < -threshold] = -1

            # Compute ternary scale: scale = mean(|w|) for non-zero positions
            non_zero_mask = w_ternary != 0
            if non_zero_mask.any():
                ternary_scale = w_float[non_zero_mask].abs().mean().item()
            else:
                ternary_scale = abs_mean.item()

            # Create PTQTernaryLinear
            ptq_linear = PTQTernaryLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                num_threads=num_threads,
            )

            # Set the ternary weights
            N, K = w_float.shape
            K_padded = ptq_linear.K_padded

            ptq_linear.w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
            ptq_linear.w_int8[:, :K] = w_ternary
            ptq_linear.w_sum = ptq_linear.w_int8.sum(dim=1, dtype=torch.int32)
            ptq_linear.scale = ternary_scale

            # Copy bias
            if module.bias is not None:
                ptq_linear.bias = module.bias.clone()

            # Calculate stats
            w_reconstructed = ptq_linear.w_int8[:, :K].float() * ternary_scale
            mse = ((w_float - w_reconstructed) ** 2).mean().item()
            sparsity = (w_ternary == 0).float().mean().item()

            from ptq_quantize import QuantizationStats
            ptq_linear.quant_stats = QuantizationStats(
                method='w8a8_ternary',
                scale=ternary_scale,
                sparsity=sparsity,
                mse=mse,
            )

            # Replace module in model
            parts = name.rsplit('.', 1)
            if len(parts) == 1:
                parent = w8a8_model.model
                attr_name = parts[0]
            else:
                parent_name, attr_name = parts
                parent = w8a8_model.model.get_submodule(parent_name)

            setattr(parent, attr_name, ptq_linear)
            converted_count += 1

            if converted_count <= 3:
                print(f"  {name}: MSE={mse:.2e}, sparsity={sparsity:.1%}")

    print(f"\nConverted {converted_count} layers from INT8 to Ternary")

    # The model wrapper is still W8A8Model but now contains PTQTernaryLinear layers
    # We should wrap it in PTQModel for consistency
    ptq_model = PTQModel(w8a8_model.model, w8a8_model.config)
    ptq_model.eval()

    return ptq_model, tokenizer


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test W8A8 model')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name')
    args = parser.parse_args()

    print("=" * 70)
    print("W8A8 MODEL TEST")
    print("=" * 70)

    model, tokenizer = load_w8a8_model(args.model)

    # Test
    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=30, do_sample=False)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Output: {output_text}")
