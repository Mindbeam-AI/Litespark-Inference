"""
Plain-data container for a loaded, packed BitNet model. No nn.Module, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BitNetConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float


@dataclass
class PackedProjection:
    """One ternary projection: packed weights + per-tensor scale.

    BitNet b1.58 uses a single scalar scale (weight.abs().mean()) per matrix,
    not a per-row scale. w_sum is retained for potential future kernels that
    need offset correction; the current packed M=1 kernel does not use it.

    w_unpacked is an optional pre-decoded view of w_packed: each ternary
    weight expanded back to one uint8 byte in [0, 1, 2] (the unsigned form
    -- the matmul kernel applies the -1 bias on the output side). 4x bigger
    in memory but lets the matmul skip the entire port-5 unpack chain. Set
    by the loader when LITESPARK_PREUNPACK=1; None otherwise.
    """
    w_packed: np.ndarray              # uint8 [N, ceil(K/4)]
    w_sum: np.ndarray                 # int32 [N] (unused by the current M=1 kernel)
    scale: float                      # per-tensor absmean
    in_features: int
    out_features: int
    w_unpacked: "np.ndarray | None" = None   # uint8 [N, K] in [0, 1, 2]


@dataclass
class PackedLayer:
    q_proj: PackedProjection
    k_proj: PackedProjection
    v_proj: PackedProjection
    o_proj: PackedProjection
    gate_proj: PackedProjection
    up_proj: PackedProjection
    down_proj: PackedProjection
    input_norm: np.ndarray          # bf16 as uint16 or fp32, as loaded
    post_attn_norm: np.ndarray
    attn_sub_norm: np.ndarray
    ffn_sub_norm: np.ndarray


@dataclass
class PackedBitNetModel:
    config: BitNetConfig
    # embed_tokens is bf16 (uint16) when the model was loaded in the
    # full-precision path, or None if loaded with int8 embedding quant.
    embed_tokens: "np.ndarray | None"
    final_norm: np.ndarray
    layers: list = field(default_factory=list)

    # Optional int8 per-row-quantized embedding. When these are set the
    # LM head uses lm_head_int8_fp32_neon and the embedding lookup
    # dequantizes one row on the fly. Saves ~313 MB of RSS vs bf16.
    embed_int8: "np.ndarray | None" = None       # [V, H] int8
    embed_scale: "np.ndarray | None" = None      # [V] fp32

    # Optional int4 per-row-quantized embedding (2 nibbles/byte).
    # Saves an extra ~157 MB vs int8 (embedding is ~156 MB + scale).
    # When set, embed_int8 is None.
    embed_int4: "np.ndarray | None" = None       # [V, H/2] uint8

    # The embedding is a zero-copy view into an mmap'd safetensors file;
    # the loader attaches the SafetensorsFile via ._safetensors to keep
    # the mapping alive for the model's lifetime. The LM head scans the
    # whole embedding per forward, but because the pages are file-backed
    # they are reclaimable by the OS under memory pressure.
    _safetensors: "object | None" = field(default=None, repr=False)
    # Cached (offset, length) within the mmap for the embedding tensor,
    # used by release_embedding_pages() to issue madvise(DONTNEED) so the
    # 626 MB embedding doesn't stay resident between LM head calls.
    _emb_mm_offset: int = 0
    _emb_mm_length: int = 0

    def release_embedding_pages(self) -> None:
        """
        Discard resident pages backing the embedding tensor. The next
        access re-faults them from the kernel's file cache (fast) or
        disk under memory pressure (slower). Used by the torchless
        runtime after each LM head call to keep RSS flat.
        """
        sf = self._safetensors
        if sf is None or getattr(sf, "mm", None) is None:
            return
        if self._emb_mm_length <= 0:
            return
        import mmap
        opt = getattr(mmap, "MADV_DONTNEED", None)
        if opt is None:
            return
        try:
            sf.mm.madvise(opt, self._emb_mm_offset, self._emb_mm_length)
        except (OSError, ValueError):
            pass

    def tensor_bytes(self) -> dict:
        """Return a breakdown of stored tensor bytes, for memory accounting."""
        def nbytes(x):
            return int(x.nbytes) if isinstance(x, np.ndarray) else 0

        packed = 0
        sums = 0
        norms = 0
        for L in self.layers:
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"):
                p = getattr(L, proj_name)
                packed += nbytes(p.w_packed)
                sums += nbytes(p.w_sum)
            for n in ("input_norm", "post_attn_norm", "attn_sub_norm", "ffn_sub_norm"):
                norms += nbytes(getattr(L, n))
        # scales are now scalar floats -- negligible, omit from tally
        scales = 0
        return {
            "packed_weights": packed,
            "w_sum": sums,
            "scales": scales,
            "per_layer_norms": norms,
            "final_norm": nbytes(self.final_norm),
            "embedding": (
                nbytes(self.embed_tokens)
                + nbytes(self.embed_int8)
                + nbytes(self.embed_int4)
                + nbytes(self.embed_scale)
            ),
            "total_excl_embedding": packed + sums + scales + norms + nbytes(self.final_norm),
            "total_incl_embedding": (
                packed + sums + scales + norms + nbytes(self.final_norm)
                + nbytes(self.embed_tokens)
                + nbytes(self.embed_int8)
                + nbytes(self.embed_int4)
                + nbytes(self.embed_scale)
            ),
        }
