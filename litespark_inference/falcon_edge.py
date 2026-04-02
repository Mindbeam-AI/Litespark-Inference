"""Falcon Edge runtime built on a LLaMA-style ternary stack."""

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import BitNetLinear


def unpack_bitnet_packed(packed: torch.Tensor) -> torch.Tensor:
    """Unpack HF packed uint8 ternary weights into int8 values in {-1, 0, 1}."""
    n = packed.shape[0]
    cols = packed.shape[1] if packed.dim() > 1 else 1
    out_rows = n * 4

    if packed.dim() > 1:
        out = torch.zeros((out_rows, cols), dtype=torch.uint8)
    else:
        out = torch.zeros((out_rows,), dtype=torch.uint8)

    for i in range(4):
        mask = 3 << (2 * i)
        out[i * n : (i + 1) * n] = (packed & mask) >> (2 * i)

    return out.to(torch.int8) - 1


def is_packed_bitnet(hf_config: dict) -> bool:
    """Return whether the HF config declares packed BitNet quantization."""
    qconfig = hf_config.get("quantization_config", {})
    return qconfig.get("quant_method") == "bitnet"


@dataclass
class LlamaTernaryConfig:
    """Configuration for Falcon Edge's LLaMA-style ternary architecture."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    eos_token_ids: Tuple[int, ...] = (2,)
    packed_bitnet: bool = False


class LlamaTernaryAttention(nn.Module):
    """LLaMA-style attention with GQA, RoPE, and KV cache."""

    def __init__(self, config: LlamaTernaryConfig, num_threads: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = config.num_heads // config.num_kv_heads
        self.num_threads = num_threads

        self.q_proj = BitNetLinear(config.hidden_size, config.num_heads * self.head_dim, num_threads=num_threads)
        kv_dim = config.num_kv_heads * self.head_dim
        self.k_proj = BitNetLinear(config.hidden_size, kv_dim, num_threads=num_threads)
        self.v_proj = BitNetLinear(config.hidden_size, kv_dim, num_threads=num_threads)
        self.o_proj = BitNetLinear(config.num_heads * self.head_dim, config.hidden_size, num_threads=num_threads)

        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.use_sdpa = platform.machine() in ("x86_64", "AMD64")
        self.rope_theta = config.rope_theta
        self._init_rope(config.max_position_embeddings)

    def _init_rope(self, max_seq_len: int):
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[position_ids].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[position_ids].unsqueeze(0).unsqueeze(0)
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        return x * cos + rotated * sin

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        position_ids = torch.arange(past_len, past_len + seq_len, device=x.device)

        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_cache = (k, v) if use_cache else None

        if self.num_kv_groups == 1:
            k_expanded = k
            v_expanded = v
        else:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_sdpa and seq_len == 1:
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded, attn_mask=None, dropout_p=0.0, is_causal=False
            )
        else:
            attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
            if seq_len > 1:
                causal_mask = torch.triu(
                    torch.ones(seq_len, k_expanded.shape[2], device=q.device, dtype=torch.bool),
                    diagonal=k_expanded.shape[2] - seq_len + 1,
                )
                attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_weights, v_expanded)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(out), new_cache


class LlamaTernaryMLP(nn.Module):
    """LLaMA-style SwiGLU MLP."""

    def __init__(self, config: LlamaTernaryConfig, num_threads: int):
        super().__init__()
        self.gate_proj = BitNetLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.up_proj = BitNetLinear(config.hidden_size, config.intermediate_size, num_threads=num_threads)
        self.down_proj = BitNetLinear(config.intermediate_size, config.hidden_size, num_threads=num_threads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTernaryBlock(nn.Module):
    """Single Falcon Edge transformer block."""

    def __init__(self, config: LlamaTernaryConfig, num_threads: int):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = LlamaTernaryAttention(config, num_threads)
        self.post_attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaTernaryMLP(config, num_threads)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.input_norm(x), past_key_value, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attn_norm(x))
        return x, new_cache


class LlamaTernary(nn.Module):
    """Falcon Edge runtime using kernel-backed ternary linear layers."""

    def __init__(self, config: LlamaTernaryConfig, num_threads: int = None):
        super().__init__()
        self.config = config
        self.num_threads = num_threads or os.cpu_count()

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaTernaryBlock(config, self.num_threads) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ):
        x = self.embed_tokens(input_ids)
        new_caches = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, new_cache = layer(x, past_kv, use_cache)
            if use_cache:
                new_caches.append(new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_caches
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = None,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
        streamer=None,
    ) -> torch.Tensor:
        if do_sample is None:
            do_sample = temperature > 0

        generated = input_ids.clone()
        past_key_values = None
        eos_token_ids = (eos_token_id,) if eos_token_id is not None else self.config.eos_token_ids

        for _ in range(max_new_tokens):
            current_input = generated[:, -1:] if past_key_values is not None else generated
            logits, past_key_values = self.forward(current_input, past_key_values=past_key_values, use_cache=True)
            next_logits = logits[:, -1, :]

            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                gen_tokens = generated[0].tolist()
                ngram_len = no_repeat_ngram_size - 1
                context = tuple(gen_tokens[-ngram_len:]) if ngram_len > 0 else ()
                for i in range(len(gen_tokens) - ngram_len):
                    if tuple(gen_tokens[i : i + ngram_len]) == context:
                        next_logits[0, gen_tokens[i + ngram_len]] = float("-inf")

            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if streamer is not None:
                streamer.put(next_token)

            if next_token.item() in eos_token_ids:
                break

        if streamer is not None:
            streamer.end()

        return generated

    @classmethod
    def from_pretrained(cls, model_name: str, num_threads: int = None):
        from huggingface_hub import hf_hub_download, snapshot_download

        print(f"Loading {model_name}...")

        config_path = hf_hub_download(model_name, "config.json")
        with open(config_path) as f:
            hf_config = json.load(f)

        eos_raw = hf_config.get("eos_token_id", 2)
        eos_ids = tuple(eos_raw) if isinstance(eos_raw, list) else (eos_raw,)

        config = LlamaTernaryConfig(
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
            packed_bitnet=is_packed_bitnet(hf_config),
        )

        model = cls(config, num_threads)

        print("Loading model weights...")
        snapshot_dir = snapshot_download(
            model_name,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        model_paths = sorted(str(path) for path in Path(snapshot_dir).glob("*.safetensors"))
        if not model_paths:
            raise FileNotFoundError(f"No safetensors were found for {model_name} in {snapshot_dir}.")

        if config.packed_bitnet:
            print("Unpacking packed BitNet weights...")
            model._load_packed_safetensors_weights(model_paths)
        else:
            print("Packing ternary weights (no re-quantization)...")
            model._load_safetensors_weights(model_paths)

        return model

    def _load_safetensors_weights(self, model_paths: List[str]):
        from safetensors import safe_open

        tensors = {}
        for path in model_paths:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        self.embed_tokens.weight.data = tensors["model.embed_tokens.weight"].float()

        def _load_proj(proj, weight_key: str):
            w = tensors[weight_key].float()
            unique = w.unique()
            if unique.numel() <= 3 and w.abs().max() < 1.5:
                if all(v in (-1.0, 0.0, 1.0) for v in unique.tolist()):
                    proj.load_from_ternary_weight(w)
                    return
            proj.load_from_weight(w)

        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            _load_proj(layer.attn.q_proj, f"{prefix}.self_attn.q_proj.weight")
            _load_proj(layer.attn.k_proj, f"{prefix}.self_attn.k_proj.weight")
            _load_proj(layer.attn.v_proj, f"{prefix}.self_attn.v_proj.weight")
            _load_proj(layer.attn.o_proj, f"{prefix}.self_attn.o_proj.weight")
            _load_proj(layer.mlp.gate_proj, f"{prefix}.mlp.gate_proj.weight")
            _load_proj(layer.mlp.up_proj, f"{prefix}.mlp.up_proj.weight")
            _load_proj(layer.mlp.down_proj, f"{prefix}.mlp.down_proj.weight")
            layer.input_norm.weight.data = tensors[f"{prefix}.input_layernorm.weight"].float()
            layer.post_attn_norm.weight.data = tensors[f"{prefix}.post_attention_layernorm.weight"].float()

        self.norm.weight.data = tensors["model.norm.weight"].float()
        self.lm_head.weight.data = tensors["lm_head.weight"].float()

    def _load_packed_safetensors_weights(self, model_paths: List[str]):
        from safetensors import safe_open

        tensors = {}
        for path in model_paths:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)

        all_keys = set(tensors.keys())
        self.embed_tokens.weight.data = tensors["model.embed_tokens.weight"].float()

        def _load_proj(proj, weight_key: str):
            raw = tensors[weight_key]
            scale_key = weight_key + "_scale"

            if scale_key in all_keys:
                unpacked = unpack_bitnet_packed(raw).float()
                proj.load_from_ternary_weight(unpacked)
                proj.scale = 1.0 / tensors[scale_key].float().item()
            elif raw.dtype == torch.uint8:
                proj.load_from_ternary_weight(unpack_bitnet_packed(raw).float())
            else:
                proj.load_from_ternary_weight(raw.float())

        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            _load_proj(layer.attn.q_proj, f"{prefix}.self_attn.q_proj.weight")
            _load_proj(layer.attn.k_proj, f"{prefix}.self_attn.k_proj.weight")
            _load_proj(layer.attn.v_proj, f"{prefix}.self_attn.v_proj.weight")
            _load_proj(layer.attn.o_proj, f"{prefix}.self_attn.o_proj.weight")
            _load_proj(layer.mlp.gate_proj, f"{prefix}.mlp.gate_proj.weight")
            _load_proj(layer.mlp.up_proj, f"{prefix}.mlp.up_proj.weight")
            _load_proj(layer.mlp.down_proj, f"{prefix}.mlp.down_proj.weight")
            layer.input_norm.weight.data = tensors[f"{prefix}.input_layernorm.weight"].float()
            layer.post_attn_norm.weight.data = tensors[f"{prefix}.post_attention_layernorm.weight"].float()

        self.norm.weight.data = tensors["model.norm.weight"].float()
        if "lm_head.weight" in all_keys:
            self.lm_head.weight.data = tensors["lm_head.weight"].float()
        else:
            self.lm_head.weight.data = tensors["model.embed_tokens.weight"].float()


__all__ = [
    "LlamaTernary",
    "LlamaTernaryAttention",
    "LlamaTernaryBlock",
    "LlamaTernaryConfig",
    "LlamaTernaryMLP",
    "is_packed_bitnet",
    "unpack_bitnet_packed",
]
