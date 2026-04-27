"""
Torchless loader for Microsoft BitNet b1.58-2B-4T-bf16.

Reads safetensors directly, quantizes every ternary projection to int8, packs
to 2-bit, and returns a PackedBitNetModel. No torch, no transformers.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import platform
from pathlib import Path

from .model import BitNetConfig, PackedBitNetModel, PackedLayer, PackedProjection
from .pack import pack_ternary_4_per_byte
from .quantize import bf16_u16_to_fp32, quantize_ternary_absmean
from .safetensors_io import open_safetensors


import numpy as np


_libc_pressure_relief = None


def _release_pages() -> None:
    """
    Hint the allocator to return freed blocks to the OS. Per-loop calls
    during weight loading keep the high-water-mark close to the actual
    held-data size; without this, macOS's libmalloc can retain freed
    pages in its magazines and the peak RSS climbs hundreds of MB above
    steady state.

    Best-effort: no-op if malloc_zone_pressure_relief isn't available.
    """
    global _libc_pressure_relief
    if platform.system() != "Darwin":
        gc.collect()
        return
    if _libc_pressure_relief is None:
        try:
            libc = ctypes.CDLL("libSystem.dylib")
            libc.malloc_zone_pressure_relief.restype = ctypes.c_size_t
            libc.malloc_zone_pressure_relief.argtypes = [
                ctypes.c_void_p, ctypes.c_size_t,
            ]
            _libc_pressure_relief = libc.malloc_zone_pressure_relief
        except Exception:
            _libc_pressure_relief = False
    gc.collect()
    if _libc_pressure_relief:
        try:
            _libc_pressure_relief(None, 0)
        except Exception:
            pass

_DEFAULT_HF_REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"
_TERNARY_KEYS = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def _resolve_hf_snapshot(repo: str) -> tuple[Path, Path]:
    """Locate a cached HF snapshot. Returns (config.json path, model.safetensors path)."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = root / ("models--" + repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"{repo} not found under {snapshots}. Download it first via "
            f"`huggingface-cli download {repo}` or let the torch-based loader "
            f"prime the cache."
        )
    # Pick first snapshot (there's typically one)
    (snap,) = list(snapshots.iterdir())[:1] or (None,)
    if snap is None:
        raise FileNotFoundError(f"{repo}: no snapshots in {snapshots}")
    return snap / "config.json", snap / "model.safetensors"


def _load_proj(sf, prefix: str, suffix: str) -> PackedProjection:
    w_u16, _ = sf.get(prefix + suffix)          # bf16 (uint16 view)
    N, K = w_u16.shape
    w_int8, scale = quantize_ternary_absmean(w_u16)   # scale is scalar
    # Drop the bf16 bytes buffer as soon as possible -- gate/up weights are
    # ~35 MB each and we don't need them after quantization.
    del w_u16
    w_sum = w_int8.sum(axis=1, dtype="int32")
    w_packed = pack_ternary_4_per_byte(w_int8)
    del w_int8
    proj = PackedProjection(
        w_packed=w_packed, w_sum=w_sum, scale=scale,
        in_features=K, out_features=N,
    )
    # Return freed transient blocks to the OS every projection, not just
    # every layer. ~210 pressure-relief calls is cheap and keeps peak
    # bounded near the steady-state footprint instead of growing with
    # allocator pool retention.
    _release_pages()
    return proj


def _quantize_embedding_int4(
    emb_u16: "np.ndarray", row_block: int = 4096,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Per-row absmax int4 quantization of a bf16 (uint16-viewed) embedding,
    packed 2 nibbles per byte.

    Returns (emb_packed, emb_scale) where
        emb_packed[v, j]        = low_nib(v, 2j) | (high_nib(v, 2j+1) << 4)
        low_nib / high_nib      are signed 4-bit values in [-7, +7]
        emb_scale[v]            = max|row v| / 7

    Requires H (embedding dim) to be even.
    """
    V, H = emb_u16.shape
    if H % 2:
        raise ValueError(f"int4 embedding requires even H, got {H}")
    emb_packed = np.empty((V, H // 2), dtype=np.uint8)
    emb_scale = np.empty(V, dtype=np.float32)
    for v0 in range(0, V, row_block):
        v1 = min(v0 + row_block, V)
        block_fp32 = bf16_u16_to_fp32(emb_u16[v0:v1])
        abs_max = np.abs(block_fp32).max(axis=1)
        abs_max = np.maximum(abs_max, 1e-5)
        scale = abs_max / 7.0
        inv_scale = 1.0 / scale
        q = np.round(block_fp32 * inv_scale[:, None]).clip(-7, 7).astype(np.int8)
        del block_fp32
        # Pack: 2 nibbles/byte. low = q[:, 0::2], high = q[:, 1::2].
        low  = (q[:, 0::2].astype(np.uint8) & 0x0F)
        high = (q[:, 1::2].astype(np.uint8) & 0x0F) << 4
        emb_packed[v0:v1] = low | high
        emb_scale[v0:v1] = scale
        del q, low, high
    # Pressure-relieve once at end so per-iter allocator churn doesn't
    # matter; forcing it inside the loop actually hurt peak in practice.
    return emb_packed, emb_scale


def _quantize_embedding_int8(
    emb_u16: "np.ndarray", row_block: int = 4096,
) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Per-row absmax int8 quantization of a bf16 (uint16-viewed) embedding.

    Returns (emb_int8, emb_scale) where emb_scale[v] = max|row v| / 127.

    Processed in row blocks so the transient fp32 working set is bounded
    (one block_fp32 ~ row_block * H * 4 bytes; for row_block=4096 that's
    40 MB on BitNet-2B, which macOS + numpy can absorb cleanly). The
    produced int8 array is heap-owned, so the caller can free the mmap
    backing of emb_u16 afterwards if it chooses.
    """
    V, H = emb_u16.shape
    emb_int8 = np.empty((V, H), dtype=np.int8)
    emb_scale = np.empty(V, dtype=np.float32)
    for v0 in range(0, V, row_block):
        v1 = min(v0 + row_block, V)
        block_fp32 = bf16_u16_to_fp32(emb_u16[v0:v1])
        abs_max = np.abs(block_fp32).max(axis=1)
        abs_max = np.maximum(abs_max, 1e-5)
        scale = abs_max / 127.0
        inv_scale = 1.0 / scale
        q = np.round(block_fp32 * inv_scale[:, None]).clip(-127, 127).astype(np.int8)
        emb_int8[v0:v1] = q
        emb_scale[v0:v1] = scale
    return emb_int8, emb_scale


def load_bitnet_2b(
    repo: str = _DEFAULT_HF_REPO,
    *,
    embed_dtype: str = "int8",
) -> PackedBitNetModel:
    """
    Load bitnet-2b torchless.

    embed_dtype:
        "bf16" - keep the embedding in bf16 (zero-copy mmap view, 626 MB).
                 Bit-identical logits vs the torch-backed reference.
        "int8" - quantize per-row at load time (313 MB + 0.5 MB scale).
                 ~3% relative logit drift, top-1 preserved on probed prompts.
        "int4" - quantize per-row, 2 nibbles/byte (~156 MB + 0.5 MB scale).
                 Larger drift; top-1 needs verification per prompt.
    """
    config_path, st_path = _resolve_hf_snapshot(repo)

    with open(config_path) as f:
        hf_config = json.load(f)
    config = BitNetConfig(
        hidden_size=hf_config["hidden_size"],
        num_layers=hf_config["num_hidden_layers"],
        num_heads=hf_config["num_attention_heads"],
        num_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        intermediate_size=hf_config["intermediate_size"],
        vocab_size=hf_config["vocab_size"],
        max_position_embeddings=hf_config.get("max_position_embeddings", 4096),
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
        rope_theta=hf_config.get("rope_theta", 500000.0),
    )

    # Open mmap'd; keep the SafetensorsFile alive on the model so the
    # zero-copy view on embed_tokens remains valid. The embedding is
    # served from the file mapping: no 626 MB heap allocation, and the
    # kernel can madvise(DONTNEED) on it between forwards to let the OS
    # reclaim the clean file-backed pages under pressure.
    sf = open_safetensors(str(st_path))

    emb_bf16 = None
    emb_int8 = None
    emb_int4 = None
    emb_scale = None
    if embed_dtype == "bf16":
        emb_bf16, _ = sf.view("model.embed_tokens.weight")
    elif embed_dtype == "int8":
        emb_view, _ = sf.view("model.embed_tokens.weight")
        emb_int8, emb_scale = _quantize_embedding_int8(emb_view)
        del emb_view
    elif embed_dtype == "int4":
        emb_view, _ = sf.view("model.embed_tokens.weight")
        emb_int4, emb_scale = _quantize_embedding_int4(emb_view)
        del emb_view
    else:
        raise ValueError(
            f"embed_dtype must be 'bf16', 'int8', or 'int4', got {embed_dtype!r}"
        )

    # For int8/int4 we're done with the mmap: everything else gets pread'd
    # into heap buffers. Release the mmap now so the 626 MB of faulted
    # embedding pages don't stay resident through the whole layer loop.
    # (bf16 path keeps the mmap alive since its embedding is a live view.)
    if embed_dtype in ("int8", "int4"):
        if not sf.close_mmap():
            # A stray numpy view was still anchoring the mmap; force GC.
            gc.collect()
            sf.close_mmap()
        _release_pages()

    # Small tensors: copy out (pread) so they don't anchor mmap pages
    # past the tensors' own size.
    final_norm = sf.get("model.norm.weight")[0].copy()

    layers: list[PackedLayer] = []
    for i in range(config.num_layers):
        prefix = f"model.layers.{i}"
        L = PackedLayer(
            q_proj=_load_proj(sf, prefix, ".self_attn.q_proj.weight"),
            k_proj=_load_proj(sf, prefix, ".self_attn.k_proj.weight"),
            v_proj=_load_proj(sf, prefix, ".self_attn.v_proj.weight"),
            o_proj=_load_proj(sf, prefix, ".self_attn.o_proj.weight"),
            gate_proj=_load_proj(sf, prefix, ".mlp.gate_proj.weight"),
            up_proj=_load_proj(sf, prefix, ".mlp.up_proj.weight"),
            down_proj=_load_proj(sf, prefix, ".mlp.down_proj.weight"),
            input_norm=sf.get(f"{prefix}.input_layernorm.weight")[0].copy(),
            post_attn_norm=sf.get(f"{prefix}.post_attention_layernorm.weight")[0].copy(),
            attn_sub_norm=sf.get(f"{prefix}.self_attn.attn_sub_norm.weight")[0].copy(),
            ffn_sub_norm=sf.get(f"{prefix}.mlp.ffn_sub_norm.weight")[0].copy(),
        )
        layers.append(L)
        # Ask the allocator to return any freed per-projection transient
        # pages before loading the next layer. Without this the macOS
        # libmalloc retains them and peak RSS ends up hundreds of MB
        # above the actual held tensor bytes.
        _release_pages()

    model = PackedBitNetModel(
        config=config,
        embed_tokens=emb_bf16,
        final_norm=final_norm, layers=layers,
        embed_int8=emb_int8,
        embed_int4=emb_int4,
        embed_scale=emb_scale,
    )

    if embed_dtype in ("int8", "int4"):
        # We've already copied everything we need into heap buffers; the
        # safetensors mmap is no longer referenced, so close it to let
        # the OS reclaim the 4.5 GB of mapped pages (notably the 626 MB
        # bf16 embedding pages that got faulted in during quantization).
        sf.close()
        _release_pages()
    else:
        # Keep the mmap alive so the bf16 view on embed_tokens stays valid.
        model._safetensors = sf  # type: ignore[attr-defined]
        emb_meta = sf.header["model.embed_tokens.weight"]
        emb_off0, emb_off1 = emb_meta["data_offsets"]
        model._emb_mm_offset = sf.data_start + emb_off0
        model._emb_mm_length = emb_off1 - emb_off0

    return model
