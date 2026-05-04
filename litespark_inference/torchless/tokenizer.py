"""
Torchless tokenizer wrapper.

Uses the `tokenizers` Rust-backed Python package directly (no transformers,
no torch). The BitNet-2B checkpoint ships a tokenizer.json in its HF
snapshot; we read it and apply the standard chat / generation prompt format
used by the torch-backed CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DEFAULT_HF_REPO = "microsoft/bitnet-b1.58-2B-4T-bf16"


def _resolve_tokenizer_path(repo: str = _DEFAULT_HF_REPO) -> tuple[Path, Path]:
    """Resolve (tokenizer.json, tokenizer_config.json), downloading on cache miss.

    tokenizer_config.json is optional — load_tokenizer below already handles
    its absence gracefully — so we don't fail the whole load if that file is
    missing from the repo.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    tok_json = Path(hf_hub_download(repo, "tokenizer.json"))
    try:
        tok_cfg = Path(hf_hub_download(repo, "tokenizer_config.json"))
    except EntryNotFoundError:
        tok_cfg = tok_json.parent / "tokenizer_config.json"
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


@dataclass
class FalconTokenizer:
    _tok: "object"
    eos_token_id: int
    bos_token_id: Optional[int]
    pad_token_id: Optional[int]
    chat_template: Optional[str]
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None
    stop_token_ids: tuple[int, ...] = ()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def format_messages(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        if self.chat_template and "<|im_start|>" in self.chat_template:
            bos = self.bos_token or ""
            parts = [bos]
            for message in messages:
                role = message["role"]
                content = message["content"]
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "".join(parts)

        system_messages = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        instruction = system_messages[-1] if system_messages else ""
        latest_user = user_messages[-1] if user_messages else ""
        return f"{instruction}\n\nQuestion: {latest_user}\nAnswer:".strip()

    def format_chat(self, prompt: str, system: Optional[str] = None) -> str:
        if system is None:
            system = ("You are Litespark, a helpful AI assistant running locally. "
                      "Provide accurate, concise, and practical answers.")
        return self.format_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )


def load_tokenizer(repo: str = _DEFAULT_HF_REPO) -> BitNetTokenizer | FalconTokenizer:
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
        eos_token = _content(eos)
        bos_token = _content(bos)
        pad_token = _content(pad)
        eos_id = tok.token_to_id(eos_token) if eos_token else None
        bos_id = tok.token_to_id(bos_token) if bos_token else None
        pad_id = tok.token_to_id(pad_token) if pad_token else None
        chat_template = cfg.get("chat_template")
    else:
        eos_token = None
        bos_token = None

    if eos_id is None:
        # Fall back to a common default for Llama-style tokenizers.
        eos_token = "<|end_of_text|>"
        eos_id = tok.token_to_id(eos_token) or 0

    stop_ids = {int(eos_id)}
    im_end_id = tok.token_to_id("<|im_end|>")
    if im_end_id is not None and chat_template and "<|im_end|>" in chat_template:
        stop_ids.add(int(im_end_id))

    tokenizer_cls = (
        FalconTokenizer
        if chat_template and "<|im_start|>" in chat_template
        else BitNetTokenizer
    )

    common = dict(
        _tok=tok,
        eos_token_id=int(eos_id),
        bos_token_id=int(bos_id) if bos_id is not None else None,
        pad_token_id=int(pad_id) if pad_id is not None else None,
        chat_template=chat_template,
    )
    if tokenizer_cls is FalconTokenizer:
        return FalconTokenizer(
            **common,
            eos_token=eos_token,
            bos_token=bos_token,
            stop_token_ids=tuple(sorted(stop_ids)),
        )
    return BitNetTokenizer(**common)
