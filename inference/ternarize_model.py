#!/usr/bin/env python3
"""
Ternarize HuggingFace Models

Converts float32 weights to ternary (-1, 0, +1) format for use with
our VNNI ternary matmul kernels.

Ternarization methods:
1. Sign-based: w_ternary = sign(w) where |w| > threshold, else 0
2. Top-k: Keep top k% of weights by magnitude, ternarize those
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import json


def ternarize_weights_threshold(
    weights: torch.Tensor,
    threshold_ratio: float = 0.05
) -> Tuple[torch.Tensor, float]:
    """
    Ternarize weights using threshold-based method.

    Args:
        weights: Float tensor of shape [out_features, in_features]
        threshold_ratio: Fraction of max abs value to use as threshold

    Returns:
        ternary_weights: Int8 tensor with values in {-1, 0, +1}
        scale: Scale factor to approximate original magnitude
    """
    # Compute threshold
    max_abs = weights.abs().max().item()
    threshold = max_abs * threshold_ratio

    # Ternarize
    ternary = torch.zeros_like(weights, dtype=torch.int8)
    ternary[weights > threshold] = 1
    ternary[weights < -threshold] = -1

    # Compute scale (mean absolute value of non-zero original weights)
    mask = ternary != 0
    if mask.any():
        scale = weights[mask].abs().mean().item()
    else:
        scale = 1.0

    return ternary, scale


def ternarize_weights_topk(
    weights: torch.Tensor,
    keep_ratio: float = 0.3
) -> Tuple[torch.Tensor, float]:
    """
    Ternarize weights keeping top-k% by magnitude.

    Args:
        weights: Float tensor of shape [out_features, in_features]
        keep_ratio: Fraction of weights to keep non-zero

    Returns:
        ternary_weights: Int8 tensor with values in {-1, 0, +1}
        scale: Scale factor to approximate original magnitude
    """
    flat = weights.flatten()
    k = int(len(flat) * keep_ratio)

    if k == 0:
        return torch.zeros_like(weights, dtype=torch.int8), 1.0

    # Find threshold for top-k
    topk_vals, _ = torch.topk(flat.abs(), k)
    threshold = topk_vals[-1].item()

    # Ternarize
    ternary = torch.zeros_like(weights, dtype=torch.int8)
    ternary[weights >= threshold] = 1
    ternary[weights <= -threshold] = -1

    # Compute scale
    mask = ternary != 0
    if mask.any():
        scale = weights[mask].abs().mean().item()
    else:
        scale = 1.0

    return ternary, scale


def prepare_ternary_weight(
    ternary: torch.Tensor,
    pad_k: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare ternary weight for VNNI kernel.

    Args:
        ternary: Int8 tensor [N, K] with values in {-1, 0, +1}
        pad_k: Pad K dimension to multiple of this

    Returns:
        w_int8: Padded int8 tensor [N, K_padded]
        w_sum: Row sums as int32 [N]
    """
    N, K = ternary.shape
    K_padded = ((K + pad_k - 1) // pad_k) * pad_k

    w_int8 = torch.zeros(N, K_padded, dtype=torch.int8)
    w_int8[:, :K] = ternary

    w_sum = w_int8.sum(dim=1, dtype=torch.int32)

    return w_int8.contiguous(), w_sum.contiguous()


class TernaryLinear:
    """Ternary linear layer compatible with our VNNI kernels."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        method: str = 'threshold',
        threshold_ratio: float = 0.05,
        keep_ratio: float = 0.3
    ):
        """
        Args:
            weight: Original float weight [out_features, in_features]
            bias: Optional bias [out_features]
            method: 'threshold' or 'topk'
            threshold_ratio: For threshold method
            keep_ratio: For topk method
        """
        self.out_features, self.in_features = weight.shape
        self.K_padded = ((self.in_features + 63) // 64) * 64

        # Ternarize
        if method == 'threshold':
            ternary, self.scale = ternarize_weights_threshold(weight, threshold_ratio)
        elif method == 'topk':
            ternary, self.scale = ternarize_weights_topk(weight, keep_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Prepare for kernel
        self.w_int8, self.w_sum = prepare_ternary_weight(ternary)

        # Bias
        self.bias = bias.clone() if bias is not None else None

        # Stats
        nonzero = (ternary != 0).sum().item()
        total = ternary.numel()
        self.sparsity = 1.0 - (nonzero / total)

    def get_stats(self) -> Dict:
        return {
            'in_features': self.in_features,
            'out_features': self.out_features,
            'scale': self.scale,
            'sparsity': self.sparsity,
            'memory_bytes': self.w_int8.numel(),  # int8 = 1 byte
        }


def ternarize_gpt2(
    model_name: str = 'gpt2',
    method: str = 'threshold',
    threshold_ratio: float = 0.05,
    keep_ratio: float = 0.3,
    save_path: Optional[str] = None
) -> Dict[str, TernaryLinear]:
    """
    Ternarize GPT-2 model weights.

    Args:
        model_name: HuggingFace model name ('gpt2', 'gpt2-medium', etc.)
        method: Ternarization method
        threshold_ratio: For threshold method
        keep_ratio: For topk method
        save_path: Optional path to save ternarized weights

    Returns:
        Dict mapping layer names to TernaryLinear objects
    """
    from transformers import GPT2LMHeadModel

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    ternary_layers = {}
    total_params = 0
    total_nonzero = 0

    print(f"Ternarizing with method={method}...")

    # Process each transformer block
    for block_idx, block in enumerate(model.transformer.h):
        # Attention weights
        # c_attn: [hidden, 3*hidden] for Q, K, V combined
        # c_proj: [hidden, hidden] for output projection

        layer_name = f"h.{block_idx}.attn.c_attn"
        w = block.attn.c_attn.weight.data.T  # HF stores as [in, out], we need [out, in]
        b = block.attn.c_attn.bias.data if block.attn.c_attn.bias is not None else None
        ternary_layers[layer_name] = TernaryLinear(
            w, b, method, threshold_ratio, keep_ratio
        )

        layer_name = f"h.{block_idx}.attn.c_proj"
        w = block.attn.c_proj.weight.data.T
        b = block.attn.c_proj.bias.data if block.attn.c_proj.bias is not None else None
        ternary_layers[layer_name] = TernaryLinear(
            w, b, method, threshold_ratio, keep_ratio
        )

        # MLP weights
        # c_fc: [hidden, 4*hidden] up projection
        # c_proj: [4*hidden, hidden] down projection

        layer_name = f"h.{block_idx}.mlp.c_fc"
        w = block.mlp.c_fc.weight.data.T
        b = block.mlp.c_fc.bias.data if block.mlp.c_fc.bias is not None else None
        ternary_layers[layer_name] = TernaryLinear(
            w, b, method, threshold_ratio, keep_ratio
        )

        layer_name = f"h.{block_idx}.mlp.c_proj"
        w = block.mlp.c_proj.weight.data.T
        b = block.mlp.c_proj.bias.data if block.mlp.c_proj.bias is not None else None
        ternary_layers[layer_name] = TernaryLinear(
            w, b, method, threshold_ratio, keep_ratio
        )

        print(f"  Block {block_idx}: done")

    # Compute stats
    print("\nTernarization Statistics:")
    print("-" * 60)

    total_original_bytes = 0
    total_ternary_bytes = 0

    for name, layer in ternary_layers.items():
        stats = layer.get_stats()
        original_bytes = stats['in_features'] * stats['out_features'] * 4  # float32
        ternary_bytes = stats['memory_bytes']

        total_original_bytes += original_bytes
        total_ternary_bytes += ternary_bytes

        print(f"{name}:")
        print(f"  Shape: [{stats['out_features']}, {stats['in_features']}]")
        print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
        print(f"  Scale: {stats['scale']:.4f}")

    compression = total_original_bytes / total_ternary_bytes
    print("-" * 60)
    print(f"Total original: {total_original_bytes / 1e6:.1f} MB")
    print(f"Total ternary: {total_ternary_bytes / 1e6:.1f} MB")
    print(f"Compression: {compression:.1f}x")

    # Save if requested
    if save_path:
        save_ternary_weights(ternary_layers, save_path, model_name, method)
        print(f"\nSaved to: {save_path}")

    return ternary_layers, model


def save_ternary_weights(
    ternary_layers: Dict[str, TernaryLinear],
    save_path: str,
    model_name: str,
    method: str
):
    """Save ternarized weights to disk."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata = {
        'model_name': model_name,
        'method': method,
        'layers': {}
    }

    for name, layer in ternary_layers.items():
        # Save weight tensors
        torch.save(layer.w_int8, save_dir / f"{name}.w_int8.pt")
        torch.save(layer.w_sum, save_dir / f"{name}.w_sum.pt")
        if layer.bias is not None:
            torch.save(layer.bias, save_dir / f"{name}.bias.pt")

        metadata['layers'][name] = {
            'in_features': layer.in_features,
            'out_features': layer.out_features,
            'scale': layer.scale,
            'sparsity': layer.sparsity,
            'has_bias': layer.bias is not None
        }

    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_ternary_weights(load_path: str) -> Tuple[Dict[str, TernaryLinear], Dict]:
    """Load ternarized weights from disk."""
    load_dir = Path(load_path)

    with open(load_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    ternary_layers = {}

    for name, info in metadata['layers'].items():
        layer = TernaryLinear.__new__(TernaryLinear)
        layer.in_features = info['in_features']
        layer.out_features = info['out_features']
        layer.scale = info['scale']
        layer.sparsity = info['sparsity']
        layer.K_padded = ((info['in_features'] + 63) // 64) * 64

        layer.w_int8 = torch.load(load_dir / f"{name}.w_int8.pt")
        layer.w_sum = torch.load(load_dir / f"{name}.w_sum.pt")

        if info['has_bias']:
            layer.bias = torch.load(load_dir / f"{name}.bias.pt")
        else:
            layer.bias = None

        ternary_layers[name] = layer

    return ternary_layers, metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Ternarize HuggingFace models')
    parser.add_argument('--model', default='gpt2', help='Model name')
    parser.add_argument('--method', default='threshold', choices=['threshold', 'topk'])
    parser.add_argument('--threshold', type=float, default=0.05, help='Threshold ratio')
    parser.add_argument('--keep-ratio', type=float, default=0.3, help='Top-k keep ratio')
    parser.add_argument('--save', type=str, help='Save path')

    args = parser.parse_args()

    ternarize_gpt2(
        model_name=args.model,
        method=args.method,
        threshold_ratio=args.threshold,
        keep_ratio=args.keep_ratio,
        save_path=args.save
    )
