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

from .model import BitNetConfig, PackedBitNetModel, PackedLayer, PackedProjection, FalconTernaryConfig, PackedFalconLayer, PackedFalconModel
from .pack import pack_ternary_4_per_byte
from .quantize import bf16_u16_to_fp32, quantize_ternary_absmean, quantize_ternary_absmean_f32
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
    """Resolve (config.json, model.safetensors) for `repo`, downloading on cache miss.

    `hf_hub_download` is idempotent: it returns the cached blob path when the
    file is already present and downloads it on first use otherwise. It also
    honors HF_HOME / HF_HUB_OFFLINE so behaviour stays consistent with the
    rest of the HF ecosystem.
    """
    from huggingface_hub import hf_hub_download
    config_path = Path(hf_hub_download(repo, "config.json"))
    model_path = Path(hf_hub_download(repo, "model.safetensors"))
    return config_path, model_path


_PREUNPACK = os.environ.get("LITESPARK_PREUNPACK", "0") == "1"
_AMX = os.environ.get("LITESPARK_AMX", "0") == "1"


def _validate_projection_mode(mode: str) -> None:
    if mode not in ("neon", "accelerate"):
        raise ValueError(f"mode must be 'neon' or 'accelerate', got {mode!r}")
    if mode == "accelerate":
        from .kernel import has_accelerate
        if not has_accelerate():
            raise RuntimeError("torchless mode='accelerate' requires Apple Accelerate")


def _load_proj(sf, prefix: str, suffix: str, mode: str) -> PackedProjection:
    w_u16, _ = sf.get(prefix + suffix)          # bf16 (uint16 view)
    N, K = w_u16.shape
    if mode == "accelerate":
        w_float32, scale = quantize_ternary_absmean_f32(w_u16)
        del w_u16
        _release_pages()
        return PackedProjection(
            w_packed=None,
            w_sum=None,
            scale=scale,
            in_features=K,
            out_features=N,
            w_float32=w_float32,
        )

    w_int8, scale = quantize_ternary_absmean(w_u16)   # scale is scalar
    # Drop the bf16 bytes buffer as soon as possible -- gate/up weights are
    # ~35 MB each and we don't need them after quantization.
    del w_u16
    w_sum = w_int8.sum(axis=1, dtype="int32")
    w_packed = pack_ternary_4_per_byte(w_int8)
    # If LITESPARK_PREUNPACK=1, also keep an unpacked copy in [0,1,2]
    # form (4x memory, but the matmul kernel skips the whole port-5
    # unpack chain). Costs ~3 GB of RAM for BitNet-2B vs ~700 MB packed.
    w_unpacked = (w_int8 + 1).astype(np.uint8) if _PREUNPACK else None
    # If LITESPARK_AMX=1 and the kernel has AMX support, build the
    # AMX-VNNI transposed layout up front. Same 4x memory hit as
    # LITESPARK_PREUNPACK but in [-1, 0, +1] signed form and pre-
    # transposed for direct AMX TILELOADD.
    w_amx = None
    if _AMX:
        try:
            from .kernel import has_amx, transpose_packed_to_amx_vnni
            if has_amx():
                w_amx = np.empty((K // 4, N, 4), dtype=np.int8)
                transpose_packed_to_amx_vnni(w_packed, w_amx, N, K)
        except Exception:
            w_amx = None
    del w_int8
    proj = PackedProjection(
        w_packed=w_packed, w_sum=w_sum, scale=scale,
        in_features=K, out_features=N,
        w_unpacked=w_unpacked,
        w_amx=w_amx,
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
    mode: str = "neon",
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
    _validate_projection_mode(mode)
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
            q_proj=_load_proj(sf, prefix, ".self_attn.q_proj.weight", mode),
            k_proj=_load_proj(sf, prefix, ".self_attn.k_proj.weight", mode),
            v_proj=_load_proj(sf, prefix, ".self_attn.v_proj.weight", mode),
            o_proj=_load_proj(sf, prefix, ".self_attn.o_proj.weight", mode),
            gate_proj=_load_proj(sf, prefix, ".mlp.gate_proj.weight", mode),
            up_proj=_load_proj(sf, prefix, ".mlp.up_proj.weight", mode),
            down_proj=_load_proj(sf, prefix, ".mlp.down_proj.weight", mode),
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
        final_norm=final_norm,
        projection_backend=mode,
        layers=layers,
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


FALCON_TORCHLESS_REPOS = {
    "falcon-edge-1b": "tiiuae/Falcon-E-1B-Base",
    "falcon-edge-1b-instruct": "tiiuae/Falcon-E-1B-Instruct",
    "falcon-edge-3b": "tiiuae/Falcon-E-3B-Base",
    "falcon-edge-3b-instruct": "tiiuae/Falcon-E-3B-Instruct",
}


def _check_dtype(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint16:
        return bf16_u16_to_fp32(a)
    if a.dtype == np.float32:
        return a
    return a.astype(np.float32)


def _unpack_hf_packed_bitnet(packed: np.ndarray, out_features: int) -> np.ndarray:
    """Unpack HF uint8 BitNet packing into int8 {-1, 0, 1} rows."""
    if packed.ndim == 1:
        packed = packed[:, None]
    n, cols = packed.shape
    out = np.empty((n * 4, cols), dtype=np.uint8)
    for i in range(4):
        out[i * n:(i + 1) * n] = (packed >> (2 * i)) & 0x3
    return out[:out_features].astype(np.int8) - 1


def _load_falcon_proj(sf, key: str, out_features: int, mode: str) -> PackedProjection:
    raw, _dtype = sf.get(key)
    scale_key = key + "_scale"
    if raw.dtype != np.uint8:
        raise ValueError(f"Expected packed uint8 Falcon projection {key}, got {raw.dtype}")
    w_int8 = _unpack_hf_packed_bitnet(raw, out_features)
    if scale_key in sf.header:
        scale_arr, _ = sf.get(scale_key)
        scale = float(1.0 / _check_dtype(scale_arr).reshape(-1)[0])
    else:
        scale = 1.0
    N, K = w_int8.shape
    if mode == "accelerate":
        w_float32 = w_int8.astype(np.float32)
        w_float32 *= scale
        del raw, w_int8
        _release_pages()
        return PackedProjection(
            w_packed=None,
            w_sum=None,
            scale=scale,
            in_features=K,
            out_features=N,
            w_float32=w_float32,
        )

    w_sum = w_int8.sum(axis=1, dtype=np.int32)
    w_packed = pack_ternary_4_per_byte(w_int8)
    del raw, w_int8
    _release_pages()
    return PackedProjection(
        w_packed=w_packed,
        w_sum=w_sum,
        scale=scale,
        in_features=K,
        out_features=N,
    )


def _load_dense_matrix(sf, key: str, embed_dtype: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    raw, _ = sf.get(key)
    if raw.dtype != np.uint16:
        raise ValueError(f"Falcon dense tensor {key} must be BF16/uint16, got {raw.dtype}")
    if embed_dtype == "bf16":
        return raw.copy(), None, None, None
    if embed_dtype == "int8":
        q, scale = _quantize_embedding_int8(raw)
        return None, q, None, scale
    if embed_dtype == "int4":
        q, scale = _quantize_embedding_int4(raw)
        return None, None, q, scale
    raise ValueError(f"embed_dtype must be 'bf16', 'int8', or 'int4', got {embed_dtype!r}")


def load_falcon_edge(model_name: str, *, embed_dtype: str = "int4", mode: str = "neon") -> PackedFalconModel:
    _validate_projection_mode(mode)
    repo = FALCON_TORCHLESS_REPOS.get(model_name, model_name)
    config_path, model_path = _resolve_hf_snapshot(repo)

    with open(config_path) as f:
        hf_config = json.load(f)
    eos_raw = hf_config.get("eos_token_id", 2)
    eos_ids = tuple(eos_raw) if isinstance(eos_raw, list) else (int(eos_raw),)
    config = FalconTernaryConfig(
        hidden_size=hf_config["hidden_size"],
        num_layers=hf_config["num_hidden_layers"],
        num_heads=hf_config["num_attention_heads"],
        num_kv_heads=hf_config.get("num_key_value_heads", hf_config["num_attention_heads"]),
        intermediate_size=hf_config["intermediate_size"],
        vocab_size=hf_config["vocab_size"],
        max_position_embeddings=hf_config.get("max_position_embeddings", 2048),
        rms_norm_eps=hf_config.get("rms_norm_eps", 1e-5),
        rope_theta=hf_config.get("rope_theta", 10000.0),
        eos_token_ids=eos_ids,
    )

    sf = open_safetensors(str(model_path))
    try:
        emb_bf16, emb_int8, emb_int4, emb_scale = _load_dense_matrix(sf, "model.embed_tokens.weight", embed_dtype)
        _release_pages()

        if "lm_head.weight" in sf.header:
            lm_bf16, lm_int8, lm_int4, lm_scale = _load_dense_matrix(sf, "lm_head.weight", embed_dtype)
        else:
            lm_bf16, lm_int8, lm_int4, lm_scale = emb_bf16, emb_int8, emb_int4, emb_scale
        _release_pages()

        final_norm = sf.get("model.norm.weight")[0].copy()
        layers: list[PackedFalconLayer] = []
        for i in range(config.num_layers):
            prefix = f"model.layers.{i}"
            kv_out = config.num_kv_heads * (config.hidden_size // config.num_heads)
            layers.append(
                PackedFalconLayer(
                    q_proj=_load_falcon_proj(sf, f"{prefix}.self_attn.q_proj.weight", config.hidden_size, mode),
                    k_proj=_load_falcon_proj(sf, f"{prefix}.self_attn.k_proj.weight", kv_out, mode),
                    v_proj=_load_falcon_proj(sf, f"{prefix}.self_attn.v_proj.weight", kv_out, mode),
                    o_proj=_load_falcon_proj(sf, f"{prefix}.self_attn.o_proj.weight", config.hidden_size, mode),
                    gate_proj=_load_falcon_proj(sf, f"{prefix}.mlp.gate_proj.weight", config.intermediate_size, mode),
                    up_proj=_load_falcon_proj(sf, f"{prefix}.mlp.up_proj.weight", config.intermediate_size, mode),
                    down_proj=_load_falcon_proj(sf, f"{prefix}.mlp.down_proj.weight", config.hidden_size, mode),
                    input_norm=sf.get(f"{prefix}.input_layernorm.weight")[0].copy(),
                    post_attn_norm=sf.get(f"{prefix}.post_attention_layernorm.weight")[0].copy(),
                )
            )
            _release_pages()

        return PackedFalconModel(
            config=config,
            embed_tokens=emb_bf16,
            projection_backend=mode,
            embed_int8=emb_int8,
            embed_int4=emb_int4,
            embed_scale=emb_scale,
            lm_head_tokens=lm_bf16,
            lm_head_int8=lm_int8,
            lm_head_int4=lm_int4,
            lm_head_scale=lm_scale,
            final_norm=final_norm,
            layers=layers,
        )
    finally:
        sf.close()
        gc.collect()
        _release_pages()
