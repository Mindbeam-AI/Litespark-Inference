"""
High-level torchless BitNet API.

Hides the loader/tokenizer/runtime split behind a single class:

    from litespark_inference.torchless import BitNet

    bn = BitNet.from_pretrained("bitnet-2b")
    print(bn.generate("The capital of France is", max_new_tokens=32))
    print(bn.benchmark())

Backed by the same packed-ternary weights, AMX/VNNI matmul kernels,
and prompt-lookup speculative decoder as the CLI; this just gives the
team a real Python surface instead of asking them to compose
load_bitnet_2b() + load_tokenizer() + init_state() + forward_*.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import numpy as np


@dataclass
class GenResult:
    """Returned by BitNet.generate(..., return_full=True)."""
    text: str
    token_ids: list
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    gen_seconds: float
    prefill_tok_per_s: float
    gen_tok_per_s: float
    speculative: bool = False
    spec_stats: Optional[dict] = None


class BitNet:
    """Torchless BitNet inference. Stateless w.r.t. user calls -- each
    generate() builds its own InferState internally."""

    def __init__(self, model, tokenizer, *, default_t_max: int = 2048):
        self._model = model
        self._tokenizer = tokenizer
        self._default_t_max = default_t_max

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, name: str = "bitnet-2b", *,
        embed_dtype: str = "int4",
        t_max: int = 2048,
    ) -> "BitNet":
        """Load BitNet (currently only bitnet-2b) from HF cache or auto-download."""
        from . import load_bitnet_2b, load_tokenizer
        if name not in ("bitnet-2b", "microsoft/bitnet-b1.58-2B-4T-bf16"):
            raise ValueError(f"unsupported model name: {name}")
        tok = load_tokenizer()
        model = load_bitnet_2b(embed_dtype=embed_dtype)
        return cls(model, tok, default_t_max=t_max)

    # ---------------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------------

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._model.config

    def encode(self, text: str) -> list:
        return self._tokenizer.encode(text)

    def decode(self, tokens: Iterable[int]) -> str:
        return self._tokenizer.decode(list(tokens))

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------

    def generate(
        self, prompt: Union[str, list], max_new_tokens: int = 64, *,
        speculative: bool = False,
        spec_k: int = 4,
        match_n: int = 4,
        return_full: bool = False,
        chat: bool = False,
    ) -> Union[str, GenResult]:
        """Greedy decode from `prompt`. Returns text by default; pass
        return_full=True to get a GenResult with timing + counts.

        Args:
            prompt:           string (will be encoded) or pre-tokenized list
            max_new_tokens:   cap on generated tokens
            speculative:      use prompt-lookup speculative decoding
            spec_k, match_n:  speculative-decode knobs
            chat:             apply the BitNet chat template before encoding
        """
        from .runtime import (
            init_state, forward_prefill, forward_one,
            generate as _greedy, generate_speculative,
        )

        if isinstance(prompt, str):
            text = self._tokenizer.format_chat(prompt) if chat else prompt
            prompt_ids = self._tokenizer.encode(text)
        else:
            prompt_ids = list(prompt)

        t_max = max(self._default_t_max, len(prompt_ids) + max_new_tokens + 16)
        state = init_state(self._model, t_max=t_max)

        if speculative:
            t_pre = time.perf_counter()
            # generate_speculative does its own prefill internally.
            t_gen_start = time.perf_counter()
            tokens, stats = generate_speculative(
                self._model, prompt_ids, max_new_tokens,
                spec_k=spec_k, match_n=match_n,
                state=state, return_stats=True,
            )
            gen_dt = time.perf_counter() - t_gen_start
            prefill_dt = 0.0  # rolled into total gen time above
            spec_stats = stats
        else:
            t_pre = time.perf_counter()
            forward_prefill(self._model, state, list(prompt_ids))
            prefill_dt = time.perf_counter() - t_pre

            tokens = []
            t_gen_start = time.perf_counter()
            # Re-derive next-token logits via forward_prefill's last position;
            # but our forward_prefill already consumed those, so emulate the
            # CLI loop: argmax then forward_one for the rest. Cheaper to
            # reuse runtime.generate which prefills again then loops.
            tokens = _greedy(self._model, prompt_ids, max_new_tokens,
                             state=init_state(self._model, t_max=t_max))
            gen_dt = time.perf_counter() - t_gen_start
            spec_stats = None

        text_out = self.decode(tokens)
        if not return_full:
            return text_out
        return GenResult(
            text=text_out,
            token_ids=list(tokens),
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(tokens),
            prefill_seconds=prefill_dt,
            gen_seconds=gen_dt,
            prefill_tok_per_s=(len(prompt_ids) / prefill_dt) if prefill_dt > 0 else 0.0,
            gen_tok_per_s=(len(tokens) / gen_dt) if gen_dt > 0 else 0.0,
            speculative=speculative,
            spec_stats=spec_stats,
        )

    # ---------------------------------------------------------------------
    # Benchmarking
    # ---------------------------------------------------------------------

    def benchmark(
        self,
        prompts: Optional[list] = None,
        max_new_tokens: int = 32,
        runs: int = 3,
        speculative: bool = False,
        chat: bool = False,
    ) -> dict:
        """Run a small benchmark suite and return aggregated stats.

        Returns a dict shaped like:
          {
            "config":     {"speculative": ..., "max_new_tokens": ..., "runs": ...},
            "results":    [ {prompt: ..., gen_tok_per_s: [r1, r2, ...]}, ... ],
            "summary":    {"median_gen_tok_per_s": ..., "median_prefill_tok_per_s": ...},
          }
        """
        if prompts is None:
            prompts = [
                "The capital of France is",
                "Question: What is 2 + 2? Answer:",
                "def fibonacci(n):",
            ]
        results = []
        all_gen, all_pre = [], []
        for prompt in prompts:
            gens, pres = [], []
            for _ in range(runs):
                r = self.generate(prompt, max_new_tokens,
                                  speculative=speculative, chat=chat,
                                  return_full=True)
                gens.append(r.gen_tok_per_s)
                pres.append(r.prefill_tok_per_s)
            results.append({
                "prompt": prompt,
                "gen_tok_per_s": gens,
                "prefill_tok_per_s": pres,
                "median_gen": float(np.median(gens)),
                "median_prefill": float(np.median(pres)),
            })
            all_gen.extend(gens)
            all_pre.extend(pres)
        return {
            "config": {
                "speculative": speculative,
                "max_new_tokens": max_new_tokens,
                "runs": runs,
            },
            "results": results,
            "summary": {
                "median_gen_tok_per_s": float(np.median(all_gen)),
                "median_prefill_tok_per_s": float(np.median(all_pre)),
            },
        }

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------

    def memory_bytes(self) -> dict:
        """Per-component breakdown of stored tensor bytes."""
        return self._model.tensor_bytes()

    def kernel_paths(self) -> dict:
        """Which kernel optimizations are active in the loaded shared lib."""
        from . import kernel as k
        try:
            return {
                "extension": str(k._find_built_extension()),
                "has_omp": k.has_omp(),
                "max_threads": k.max_threads(),
                "has_batched_matmul": k.has_batched_matmul(),
                "has_batched_helpers": k.has_batched_helpers(),
                "has_fused_rmsnorm_quantize": k.has_fused_rmsnorm_quantize(),
                "has_amx": k.has_amx(),
                "has_unpacked_matmul": k.has_unpacked_matmul(),
            }
        except Exception as e:
            return {"error": repr(e)}
