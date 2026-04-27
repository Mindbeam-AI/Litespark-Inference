"""
Torchless tokenizer wrapper.

Uses the `tokenizers` Rust-backed Python package directly (no transformers,
no torch). The BitNet-2B checkpoint ships a tokenizer.json in its HF
snapshot; we read it and apply the standard chat / generation prompt format
used by the torch-backed CLI.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DEFAULT_HF_REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"


def _resolve_tokenizer_path(repo: str = _DEFAULT_HF_REPO) -> tuple[Path, Path]:
    """Return (tokenizer.json, tokenizer_config.json) paths from the HF cache."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snapshots = root / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(
            f"{repo} not found under {snapshots}. Download it first with "
            f"`huggingface-cli download {repo}` or let the torch-based "
            f"loader prime the cache."
        )
    (snap,) = list(snapshots.iterdir())[:1] or (None,)
    if snap is None:
        raise FileNotFoundError(f"{repo}: no snapshots in {snapshots}")
    tok_json = snap / "tokenizer.json"
    tok_cfg = snap / "tokenizer_config.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"tokenizer.json missing at {tok_json}")
    return tok_json, tok_cfg


@dataclass
class BitNetTokenizer:
    """Thin wrapper over tokenizers.Tokenizer with BitNet chat templating."""

    _tok: "object"  # tokenizers.Tokenizer
    eos_token_id: int
    bos_token_id: Optional[int]
    pad_token_id: Optional[int]
    chat_template: Optional[str]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def format_chat(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Build the prompt string the same way cli.py:_build_prompt does when a
        chat template is available, falling back to the plain prompt otherwise.

        We don't implement Jinja; instead, for BitNet-2B we use the documented
        format:

            <|begin_of_text|>System: ...<newline>User: ...<newline>Assistant:

        The torch-backed CLI uses apply_chat_template from transformers, which
        applies the tokenizer's chat template. Without transformers we hand-
        write the minimal format. If callers need exact parity they can pass
        the already-templated string directly.
        """
        if system is None:
            system = ("You are Litespark, a helpful AI assistant running locally. "
                      "Provide accurate, concise, and practical answers.")
        # Minimal non-templated prompt, matches the fallback branch in
        # cli.py:_build_prompt.
        return f"{system}\n\nQuestion: {prompt}\nAnswer:"


def load_tokenizer(repo: str = _DEFAULT_HF_REPO) -> BitNetTokenizer:
    # Import inside function so `import litespark_inference.torchless` stays
    # cheap for callers that don't need the tokenizer.
    from tokenizers import Tokenizer  # type: ignore

    tok_json, tok_cfg_path = _resolve_tokenizer_path(repo)
    tok = Tokenizer.from_file(str(tok_json))

    eos_id = None
    bos_id = None
    pad_id = None
    chat_template = None
    if tok_cfg_path.exists():
        with open(tok_cfg_path) as f:
            cfg = json.load(f)
        eos = cfg.get("eos_token")
        bos = cfg.get("bos_token")
        pad = cfg.get("pad_token")
        # eos_token / bos_token can be a string or a dict {content, ...}
        def _content(x):
            if isinstance(x, dict):
                return x.get("content")
            return x
        eos_id = tok.token_to_id(_content(eos)) if eos else None
        bos_id = tok.token_to_id(_content(bos)) if bos else None
        pad_id = tok.token_to_id(_content(pad)) if pad else None
        chat_template = cfg.get("chat_template")

    if eos_id is None:
        # Fall back to a common default for Llama-style tokenizers.
        eos_id = tok.token_to_id("<|end_of_text|>") or 0

    return BitNetTokenizer(
        _tok=tok,
        eos_token_id=int(eos_id),
        bos_token_id=int(bos_id) if bos_id is not None else None,
        pad_token_id=int(pad_id) if pad_id is not None else None,
        chat_template=chat_template,
    )
