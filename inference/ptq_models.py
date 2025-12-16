#!/usr/bin/env python3
"""
Post-Training Quantized Models

This module provides wrappers to apply our ternary kernels to any HuggingFace
causal language model via post-training quantization (PTQ).

Supported architectures:
- Phi-2 (microsoft/phi-2)
- Llama-based models (TinyLlama, Llama-2, etc.)
- Qwen models
- Mistral models

Usage:
    from ptq_models import load_ptq_model

    # Load and quantize any model
    model, tokenizer = load_ptq_model('microsoft/phi-2', method='absmean')

    # Run inference
    output = model.generate(input_ids, max_new_tokens=50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ptq_quantize import prepare_ternary_weight_ptq, QuantizationStats, collect_calibration_data
import hashlib
import os


# ============================================================================
# Kernel Loading
# ============================================================================

_kernel = None
_kernel_type = None


def get_kernel():
    """Load the appropriate kernel for this architecture."""
    global _kernel, _kernel_type

    if _kernel is not None:
        return _kernel, _kernel_type

    from torch.utils.cpp_extension import load
    import sys

    machine = platform.machine().lower()
    is_macos = sys.platform == 'darwin'

    if machine in ['x86_64', 'amd64']:
        kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/x86_64/matmul_free_tmac_vnni.cpp'
        if is_macos:
            # macOS clang doesn't support AVX-512 VNNI, use fallback
            _kernel = None
            _kernel_type = 'fallback'
        else:
            _kernel = load(
                name='matmul_free_vnni_ptq',
                sources=[str(kernel_path)],
                extra_cflags=['-O3', '-march=native', '-fopenmp', '-mavx512f', '-mavx512bw', '-mavx512vnni'],
                extra_ldflags=['-fopenmp'],
                verbose=False
            )
            _kernel_type = 'vnni'

    elif machine in ['arm64', 'aarch64']:
        kernel_path = Path(__file__).parent.parent / 'src/cpu_ops/kernels/arm64/matmul_free_graviton.cpp'
        if is_macos:
            # macOS clang doesn't support -fopenmp by default, use fallback
            _kernel = None
            _kernel_type = 'fallback'
        else:
            _kernel = load(
                name='matmul_free_graviton_ptq',
                sources=[str(kernel_path)],
                extra_cflags=['-O3', '-march=armv8.2-a+dotprod', '-fopenmp'],
                extra_ldflags=['-fopenmp'],
                verbose=False
            )
            _kernel_type = 'graviton'

    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    return _kernel, _kernel_type


# ============================================================================
# Quantized Linear Layer
# ============================================================================

class PTQTernaryLinear(nn.Module):
    """
    Linear layer with post-training quantized ternary weights.

    Uses our optimized VNNI/Graviton kernels for inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_threads: int = None,
        use_fp16_activations: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_threads = num_threads or torch.get_num_threads()
        self.use_fp16_activations = use_fp16_activations

        # Ternary weight storage
        self.register_buffer('w_int8', None)
        self.register_buffer('w_sum', None)
        self.scale = 1.0

        # Optional bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None

        # Quantization stats
        self.quant_stats: Optional[QuantizationStats] = None

        # Extra info for special quantization methods
        self.ssr_info: Optional[Dict] = None
        self.quip_info: Optional[Dict] = None
        self.register_buffer('U', None)  # For one-sided QuIP transform

        # LoRC (Low-Rank Compensation) matrices
        self.register_buffer('lorc_L', None)
        self.register_buffer('lorc_R', None)


    def quantize_from_linear(
        self,
        linear: nn.Linear,
        method: str = 'absmean',
        **kwargs
    ):
        """
        Quantize weights from a regular linear layer.

        Args:
            linear: Source nn.Linear layer
            method: Quantization method
            **kwargs: Method-specific arguments
        """
        weight = linear.weight.data.float()

        # Quantize
        w_int8, w_sum, scale, stats, extra_info = prepare_ternary_weight_ptq(
            weight, method=method, **kwargs
        )

        self.w_int8 = w_int8
        self.w_sum = w_sum
        self.scale = scale
        self.quant_stats = stats

        if extra_info:
            if 'salient_indices' in extra_info:
                self.ssr_info = extra_info
            if 'U' in extra_info:  # Simplified one-sided transform
                self.quip_info = extra_info
                self.U = extra_info['U'].to(weight.device)
            if 'lorc_L' in extra_info:  # LoRC compensation
                self.lorc_L = extra_info['lorc_L'].to(weight.device)
                self.lorc_R = extra_info['lorc_R'].to(weight.device)


        # Copy bias if present
        if linear.bias is not None:
            self.bias = linear.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using ternary kernel (or PyTorch fallback on macOS)."""
        original_shape = x.shape
        x = x.view(-1, self.in_features)
        M = x.shape[0]

        # Get kernel
        kernel, kernel_type = get_kernel()

        # For QuIP: W' = W @ U.T was stored, so we need x' = x @ U first
        # Then y = x' @ W'.T = (x @ U) @ (W @ U.T).T = x @ U @ U.T @ W.T = x @ W.T (since U is orthogonal)
        # Actually: y = x' @ W'.T and we need y = x @ W.T
        # W' = W @ U.T, so W'.T = U @ W.T
        # x @ W'.T = x @ U @ W.T... that's not what we want
        # We want: x @ W.T = x @ W'.T @ U.T.T = x @ W'.T @ U = (x @ W'.T) @ U
        # So: transform output by U, not input
        # But U is [K, K] and y is [M, N]... dimensions don't match for post-multiply
        #
        # Let's reconsider: W' = W @ U.T where W is [N, K], U is [K, K]
        # So W' is [N, K] too
        # y = x @ W.T where x is [M, K] and W is [N, K], so y is [M, N]
        # With QuIP: y' = x @ W'.T = x @ (W @ U.T).T = x @ U @ W.T
        # So y' = (x @ U) @ W.T, meaning we need to pre-transform x, not post-transform y
        #
        # To get correct y = x @ W.T from y' = x @ U @ W.T:
        # We need to pre-transform: x_transformed = x @ U, then y = x_transformed @ W'.T... no wait
        #
        # Actually the math: W' = W @ U.T
        # We store W' and want to compute y = x @ W.T
        # y' = x @ W'.T = x @ (W @ U.T).T = x @ U @ W.T
        # This is NOT equal to x @ W.T unless U = I
        #
        # The trick: we transform INPUT by U.T before kernel:
        # x' = x @ U.T
        # y = x' @ W'.T = (x @ U.T) @ (W @ U.T).T = x @ U.T @ U @ W.T = x @ W.T (since U.T @ U = I)
        # YES! Pre-transform input by U.T

        # Save original x for LoRC (before any transforms)
        x_for_lorc = x if (self.lorc_L is not None) else None

        if self.U is not None:
            # QuIP: pre-transform input by U.T to undo the column transform
            x = x @ self.U.T

        if kernel_type == 'fallback' or self.use_fp16_activations:
            # PyTorch fallback for macOS OR FP16 activations mode
            # Use reconstructed float weights with FP16/FP32 activations
            # This avoids activation quantization error while still using ternary weights
            w_float = self.w_int8[:, :self.in_features].float() * self.scale
            y = F.linear(x, w_float, self.bias)
        else:
            # Quantize activations (after QuIP transform if applicable)
            x_int8, x_scale = self._quantize_activation(x)

            # Allocate output
            y = torch.zeros(M, self.out_features, dtype=torch.float32)

            bias = self.bias if self.bias is not None else torch.Tensor()

            # Call kernel
            if kernel_type == 'vnni':
                kernel.matmul_free_vnni_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.w_int8.shape[1], self.num_threads
                )
            else:  # graviton
                kernel.matmul_free_graviton_v3(
                    x_int8, x_scale,
                    self.w_int8, self.w_sum,
                    y, bias,
                    M, self.out_features, self.w_int8.shape[1], self.num_threads
                )

            # Apply weight scale
            y = y * self.scale

        # Apply LoRC (Low-Rank Compensation) if available
        # y_corrected = y + (x @ L) @ R
        # Use original x (before QuIP transform) for LoRC
        if self.lorc_L is not None and self.lorc_R is not None:
            lorc_correction = (x_for_lorc @ self.lorc_L) @ self.lorc_R
            y = y + lorc_correction

        # Reshape output
        output_shape = original_shape[:-1] + (self.out_features,)
        return y.view(output_shape)

    def _quantize_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize float activation to int8 with per-row scaling."""
        M, K = x.shape
        K_padded = ((K + 63) // 64) * 64

        # Compute scale
        max_abs = x.abs().max(dim=1, keepdim=True).values
        scale = max_abs / 127.0
        scale = scale.clamp(min=1e-10)

        # Quantize
        inv_scale = 1.0 / scale

        if K_padded == K:
            x_int8 = (x * inv_scale).round().clamp(-127, 127).to(torch.int8)
        else:
            x_scaled = (x * inv_scale).round().clamp(-127, 127).to(torch.int8)
            x_int8 = F.pad(x_scaled, (0, K_padded - K), value=0)

        return x_int8.contiguous(), scale.squeeze(1).contiguous()



# ============================================================================
# Model Quantization
# ============================================================================

def quantize_linear_layers(
    model: nn.Module,
    method: str = 'absmean',
    num_threads: int = None,
    skip_layers: Optional[list] = None,
    calibration_data: Optional[Dict[str, list]] = None,
    use_fp16_activations: bool = False,
    **kwargs
) -> Dict[str, QuantizationStats]:
    """
    Replace all Linear layers with PTQTernaryLinear.

    Args:
        model: HuggingFace model to quantize
        method: Quantization method
        num_threads: Number of threads for kernels
        skip_layers: List of layer name patterns to skip (e.g., ['lm_head', 'embed'])
        calibration_data: Dict mapping layer names to activation tensors
        use_fp16_activations: Use FP16 activations instead of INT8 (more accurate but slower)
        **kwargs: Method-specific arguments

    Returns:
        Dictionary of layer names to quantization stats
    """
    skip_layers = skip_layers or []
    stats = {}

    def should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_layers)

    def replace_linear(parent: nn.Module, name: str, linear: nn.Linear, full_name: str):
        """Replace a single linear layer."""
        ptq_linear = PTQTernaryLinear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            num_threads=num_threads,
            use_fp16_activations=use_fp16_activations
        )

        # Add calibration activations for calibrated methods
        quant_kwargs = kwargs.copy()
        if calibration_data is not None and full_name in calibration_data:
            quant_kwargs['calibration_activations'] = calibration_data[full_name]

        ptq_linear.quantize_from_linear(linear, method=method, **quant_kwargs)
        setattr(parent, name, ptq_linear)
        return ptq_linear.quant_stats

    # Walk through model and replace linear layers
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not should_skip(full_name):
            # Get parent module and attribute name
            parts = full_name.rsplit('.', 1)
            if len(parts) == 1:
                parent = model
                attr_name = parts[0]
            else:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)

            # Replace
            layer_stats = replace_linear(parent, attr_name, module, full_name)
            stats[full_name] = layer_stats
            print(f"  Quantized {full_name}: sparsity={layer_stats.sparsity:.1%}, MSE={layer_stats.mse:.2e}")

    return stats


# ============================================================================
# PTQ Model Wrapper
# ============================================================================

class PTQModel(nn.Module):
    """
    Wrapper for any HuggingFace causal LM with PTQ ternary weights.
    """

    def __init__(self, hf_model: nn.Module, config: Any):
        super().__init__()
        self.model = hf_model
        self.config = config
        self.quant_stats: Dict[str, QuantizationStats] = {}

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass returning logits."""
        outputs = self.model(input_ids, **kwargs)
        if hasattr(outputs, 'logits'):
            return outputs.logits
        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
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

                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float('-inf')

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                if do_sample:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================================
# Caching Functions
# ============================================================================

def get_cache_path(model_name: str, method: str, cache_dir: Optional[str] = None) -> Path:
    """Get the cache file path for a quantized model."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / 'ptq_cache'
    else:
        cache_dir = Path(cache_dir)

    # Create a safe filename from model name and method
    safe_name = model_name.replace('/', '_').replace('\\', '_')
    cache_file = cache_dir / f"{safe_name}_{method}.pt"
    return cache_file


def save_quantized_model(model: nn.Module, cache_path: Path, quant_stats: Dict):
    """Save quantized model weights to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all PTQTernaryLinear state
    ptq_state = {}
    for name, module in model.named_modules():
        if isinstance(module, PTQTernaryLinear):
            ptq_state[name] = {
                'w_int8': module.w_int8,
                'w_sum': module.w_sum,
                'scale': module.scale,
                'bias': module.bias,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'U': module.U,
                'lorc_L': module.lorc_L,
                'lorc_R': module.lorc_R,
            }

    torch.save({
        'ptq_state': ptq_state,
        'quant_stats': quant_stats,
    }, cache_path)
    print(f"  Saved quantized model to: {cache_path}")


def load_quantized_model(
    hf_model: nn.Module,
    cache_path: Path,
    num_threads: int = None,
    skip_layers: list = None
) -> Tuple[Dict[str, QuantizationStats], bool]:
    """
    Load quantized weights from cache and replace linear layers.

    Returns:
        (quant_stats, success) - stats dict and whether loading succeeded
    """
    if not cache_path.exists():
        return {}, False

    try:
        print(f"  Loading cached quantized model from: {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        ptq_state = cached['ptq_state']
        quant_stats = cached['quant_stats']

        skip_layers = skip_layers or []

        def should_skip(name: str) -> bool:
            return any(pattern in name for pattern in skip_layers)

        # Replace linear layers with cached PTQTernaryLinear
        for full_name, module in list(hf_model.named_modules()):
            if isinstance(module, nn.Linear) and not should_skip(full_name) and full_name in ptq_state:
                state = ptq_state[full_name]

                # Get parent module
                parts = full_name.rsplit('.', 1)
                if len(parts) == 1:
                    parent = hf_model
                    attr_name = parts[0]
                else:
                    parent_name, attr_name = parts
                    parent = hf_model.get_submodule(parent_name)

                # Create PTQTernaryLinear with cached weights
                ptq_linear = PTQTernaryLinear(
                    state['in_features'],
                    state['out_features'],
                    bias=state['bias'] is not None,
                    num_threads=num_threads
                )
                ptq_linear.w_int8 = state['w_int8']
                ptq_linear.w_sum = state['w_sum']
                ptq_linear.scale = state['scale']
                ptq_linear.bias = state['bias']
                ptq_linear.U = state['U']
                ptq_linear.lorc_L = state['lorc_L']
                ptq_linear.lorc_R = state['lorc_R']

                setattr(parent, attr_name, ptq_linear)

        return quant_stats, True

    except Exception as e:
        print(f"  [!] Failed to load cache: {e}")
        return {}, False


# ============================================================================
# Main Loading Function
# ============================================================================

def load_ptq_model(
    model_name: str,
    method: str = 'absmean',
    num_threads: int = None,
    skip_lm_head: bool = True,
    skip_embeddings: bool = True,
    skip_patterns: Optional[List[str]] = None,
    calibration_samples: int = 0,
    use_lorc: bool = False,
    lorc_rank: int = 4,
    use_fp16_activations: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    force_requantize: bool = False,
    **quant_kwargs
) -> Tuple[PTQModel, Any]:
    """
    Load and quantize any HuggingFace causal LM.

    Args:
        model_name: HuggingFace model name (e.g., 'microsoft/phi-2')
        method: Quantization method
        num_threads: Number of threads for kernels
        skip_lm_head: Don't quantize the language model head (recommended)
        skip_embeddings: Don't quantize embedding layers (they're not Linear)
        skip_patterns: Additional layer name patterns to skip (e.g., ['q_proj', 'k_proj'])
        calibration_samples: Number of calibration samples for calibrated methods
        use_lorc: Enable Low-Rank Compensation for reduced quantization error
        lorc_rank: Rank for LoRC matrices (higher = better quality, more memory)
        use_fp16_activations: Use FP16 activations instead of INT8 (slower but more accurate)
        use_cache: Cache quantized weights to disk for faster subsequent loads
        cache_dir: Directory for cache files (default: inference/ptq_cache/)
        force_requantize: Ignore cache and re-quantize from scratch
        **quant_kwargs: Additional arguments for quantization method

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")

    # Load config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Load model in float32
    print("  Loading weights (this may take a while)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine layers to skip
    skip_layers = []
    if skip_lm_head:
        skip_layers.extend(['lm_head', 'output'])
    if skip_embeddings:
        skip_layers.extend(['embed', 'wte', 'wpe'])
    if skip_patterns:
        skip_layers.extend(skip_patterns)

    # Check cache
    cache_path = get_cache_path(model_name, method, cache_dir)
    loaded_from_cache = False

    if use_cache and not force_requantize:
        quant_stats, loaded_from_cache = load_quantized_model(
            hf_model, cache_path, num_threads, skip_layers
        )

    if not loaded_from_cache:
        # Collect calibration data if using calibrated methods
        calibration_data = None
        if method in ['pt2_calibrated', 'pt2_ssr', 'quip', 'omniquant'] and calibration_samples > 0:
            print(f"\nCollecting calibration data ({calibration_samples} samples)...")
            calibration_data = collect_calibration_data(
                hf_model, tokenizer,
                num_samples=calibration_samples,
                max_length=512
            )
        elif method in ['pt2_calibrated', 'pt2_ssr', 'quip', 'omniquant']:
            # Auto-enable calibration for these methods
            print(f"\nMethod '{method}' works better with calibration. Collecting 32 samples...")
            calibration_data = collect_calibration_data(
                hf_model, tokenizer,
                num_samples=32,
                max_length=512
            )

        # Quantize
        print(f"\nQuantizing with method='{method}'...")
        print("  Loading kernel...")
        _, kernel_type = get_kernel()  # Pre-load kernel
        if kernel_type == 'fallback':
            print("  [Note] Using PyTorch fallback (optimized kernels not available on this platform)")

        # Add LoRC parameters to kwargs
        quant_kwargs['use_lorc'] = use_lorc
        quant_kwargs['lorc_rank'] = lorc_rank

        quant_stats = quantize_linear_layers(
            hf_model,
            method=method,
            num_threads=num_threads,
            skip_layers=skip_layers,
            calibration_data=calibration_data,
            use_fp16_activations=use_fp16_activations,
            **quant_kwargs
        )

        # Save to cache
        if use_cache:
            save_quantized_model(hf_model, cache_path, quant_stats)

    # Wrap model
    model = PTQModel(hf_model, config)
    model.quant_stats = quant_stats
    model.eval()

    # Print summary
    total_layers = len(quant_stats)
    avg_sparsity = sum(s.sparsity for s in quant_stats.values()) / total_layers if total_layers > 0 else 0
    avg_mse = sum(s.mse for s in quant_stats.values()) / total_layers if total_layers > 0 else 0

    print(f"\nQuantization complete:")
    print(f"  Layers quantized: {total_layers}")
    print(f"  Average sparsity: {avg_sparsity:.1%}")
    print(f"  Average MSE: {avg_mse:.2e}")
    if loaded_from_cache:
        print(f"  [Loaded from cache: {cache_path}]")

    return model, tokenizer


def load_fp16_model(model_name: str) -> Tuple[nn.Module, Any]:
    """
    Load original FP16 model for baseline comparison.

    Args:
        model_name: HuggingFace model name

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading FP16 model: {model_name}")

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap for consistent interface
    class FP16Wrapper(nn.Module):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config

        def forward(self, input_ids, **kwargs):
            outputs = self.model(input_ids, **kwargs)
            return outputs.logits if hasattr(outputs, 'logits') else outputs

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = FP16Wrapper(hf_model, config)
    model.eval()

    return model, tokenizer


# ============================================================================
# Test
# ============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test PTQ model loading')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name')
    parser.add_argument('--method', type=str, default='absmean',
                        choices=['absmean', 'percentile', 'optimal'])
    args = parser.parse_args()

    print("=" * 70)
    print("PTQ Model Test")
    print("=" * 70)

    # Load PTQ model
    model, tokenizer = load_ptq_model(args.model, method=args.method)

    # Test generation
    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Input tokens: {input_ids.shape[1]}")

    print("\nGenerating...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=30, do_sample=False)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nOutput: {output_text}")
