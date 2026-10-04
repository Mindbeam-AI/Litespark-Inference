"""Distillery Nemotron text inference with the public ARM/NEON runtime.

Routed experts stay packed in file mappings. Dense decode uses the existing
BF16 NEON GEMV, prefill uses Accelerate, and Transformers supplies the hybrid
block structure and recurrent-state reference operations. CPU only.
"""
import argparse
import ctypes
import json
import inspect
from pathlib import Path
import platform
import resource
import time
from types import FunctionType, MethodType

import numpy as np
from safetensors import safe_open
import torch

from .torchless import kernel


class NativeKernels:
    def __init__(self, threads=10):
        if not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if platform.machine().lower() not in ("arm64", "aarch64"):
            raise RuntimeError("This Nemotron backend requires ARM64 NEON")
        self.lib = kernel._load()
        array = np.ctypeslib.ndpointer
        f32 = array(dtype=np.float32, flags="C_CONTIGUOUS")
        u16 = array(dtype=np.uint16, flags="C_CONTIGUOUS")
        u8 = array(dtype=np.uint8, flags="C_CONTIGUOUS")
        i64 = array(dtype=np.int64, flags="C_CONTIGUOUS")
        ptrs = array(dtype=np.uintp, flags="C_CONTIGUOUS")
        integer = ctypes.c_int
        self.lib.nemotron_set_threads_neon.argtypes = [integer]
        self.lib.nemotron_set_threads_neon.restype = None
        self.lib.nemotron_set_threads_neon(threads)
        self.lib.nemotron_dense_bf16_neon.argtypes = [u16, f32, f32, integer, integer, integer]
        self.lib.nemotron_dense_bf16_neon.restype = None
        self.lib.nemotron_experts_neon.argtypes = [f32, i64, f32, ptrs, ptrs, ptrs, ptrs,
                                                  f32, f32, f32, integer, integer, integer, integer, integer]
        self.lib.nemotron_experts_neon.restype = None
        self.lib.nemotron_causal_conv_neon.argtypes = [f32, f32, f32, f32] + [integer]*5
        self.lib.nemotron_causal_conv_neon.restype = None
        self.lib.nemotron_dense_int4_neon.argtypes = [u8, u16, f32, f32] + [integer]*4
        self.lib.nemotron_dense_int4_neon.restype = None

    def dense(self, weight, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if (weight.ndim != 2 or weight.dtype != np.uint16 or not weight.flags.c_contiguous or
                x.ndim != 2 or x.shape[1] != weight.shape[1] or min(weight.shape) < 1):
            raise ValueError("Expected contiguous BF16 weight bits [rows, cols] and input [tokens, cols]")
        result = np.empty((x.shape[0], weight.shape[0]), dtype=np.float32)
        self.lib.nemotron_dense_bf16_neon(weight, x, result, x.shape[0], weight.shape[0], weight.shape[1])
        return result

    def dense_int4(self, packed, scales, x, cols, group):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if (cols < 1 or group < 2 or group % 2 or packed.ndim != 2 or packed.shape[1] != (cols+1)//2 or
                scales.shape != (packed.shape[0], (cols+group-1)//group) or
                x.ndim != 2 or x.shape[1] != cols):
            raise ValueError("Invalid grouped INT4 matrix dimensions")
        output = np.empty((x.shape[0], packed.shape[0]), dtype=np.float32)
        self.lib.nemotron_dense_int4_neon(packed, scales, x, output, x.shape[0], packed.shape[0], cols, group)
        return output

    def causal_conv(self, x, weight, bias, output_tokens):
        values = x.float().contiguous().numpy()
        weights = weight.float().contiguous().numpy()
        if (x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 3 or
                weight.ndim != 2 or weight.shape[0] != x.shape[1] or
                not 0 < output_tokens <= x.shape[2] or weight.shape[1] < 1):
            raise ValueError("Expected CPU BF16 causal convolution input [batch, channels, tokens]")
        biases = np.zeros(x.shape[1], dtype=np.float32) if bias is None else bias.float().contiguous().numpy()
        output = np.empty((*x.shape[:2], output_tokens), dtype=np.float32)
        self.lib.nemotron_causal_conv_neon(values, weights, biases, output, *x.shape,
                                          weight.shape[1], output_tokens)
        return torch.from_numpy(output).to(x.dtype)


def _native_convolution(mixer, kernels):
    """Bind native convolution to this mixer's forward, leaving HF globals intact.

    Retain Transformers' cache and recurrent-state implementation. Binding a
    private copy of the forward's globals avoids changing any other model.
    """
    original = inspect.unwrap(type(mixer).forward)
    required = {"causal_conv1d_fn", "causal_conv1d_update"}
    if not required.issubset(original.__globals__):
        raise RuntimeError("Unsupported Nemotron forward; install litespark-inference[nemotron]")
    if mixer.activation not in ("silu", "swish"):
        raise ValueError("Native Nemotron convolution requires SiLU activation")

    def prefill(hidden_states, weight, bias=None, activation=None, **kwargs):
        return kernels.causal_conv(hidden_states, weight, bias, hidden_states.shape[-1])

    def update(hidden_states, conv_state, weight, bias=None, activation=None):
        combined = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
        conv_state.copy_(combined[:, :, -conv_state.shape[-1]:])
        return kernels.causal_conv(combined, weight, bias, hidden_states.shape[-1])

    namespace = dict(original.__globals__, causal_conv1d_fn=prefill, causal_conv1d_update=update)
    forward = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    forward.__kwdefaults__ = original.__kwdefaults__
    mixer.forward = MethodType(forward, mixer)


class Checkpoint:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.config = json.loads((self.root/"config.json").read_text(), object_hook=lambda v:
                                float(v["__float__"]) if set(v) == {"__float__"} else v)
        self.manifest = json.loads((self.root/"distillery_manifest.json").read_text())
        m = self.manifest
        q = self.config.get("quantization_config", {})
        if (m["format"] != "distillery-nemotron-grouped-ternary-v1" or
            m["scale_dtype"] != "float16" or m["scale_convention"] != "multiply" or
            m["activation_dtype"] != "bfloat16" or m["converted_tensors"] != 40960 or
            m["packing"] != "four output-row planes per uint8; codes -1,0,1 map to 0,1,2" or
            q.get("quant_method") != "distillery_ternary" or q.get("group_size") != m["group_size"] or
            q.get("linear_pre_norm") != "none" or not 0 < m["group_size"] <= 4096):
            raise ValueError("Unsupported Distillery checkpoint contract")
        self.weight_map = json.loads((self.root/"model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {}

    def raw(self, name):
        filename = self.weight_map[name]
        if filename not in self.handles:
            path = (self.root/filename).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Shard path escapes checkpoint directory")
            self.handles[filename] = safe_open(path, framework="pt", device="cpu")
        return self.handles[filename].get_tensor(name)


class NativeLinear(torch.nn.Module):
    def __init__(self, weight, kernels, bias=None):
        super().__init__()
        if weight.dtype != torch.bfloat16 or weight.device.type != "cpu":
            raise ValueError("Native dense weights must be CPU BF16")
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.bits = weight.detach().view(torch.uint16).numpy()
        self.kernels = kernels
        self.out_features, self.in_features = weight.shape

    def forward(self, x):
        shape = x.shape[:-1] + (self.out_features,)
        values = x.reshape(-1, self.in_features).float().contiguous().numpy()
        out = torch.from_numpy(self.kernels.dense(self.bits, values)).reshape(shape)
        if self.bias is not None:
            out += self.bias.float()
        return out.to(x.dtype)


class NativeExperts(torch.nn.Module):
    def __init__(self, checkpoint, layer, config, kernels):
        super().__init__()
        self.kernels = kernels
        self.dim, self.intermediate = config.moe_latent_size, config.moe_intermediate_size
        self.group = checkpoint.manifest["group_size"]
        self.num_experts = config.n_routed_experts
        self.arrays = []
        self.pointers = []
        for projection, rows, cols in [("up_proj", self.intermediate, self.dim),
                                        ("down_proj", self.dim, self.intermediate)]:
            codes, scales = [], []
            for expert in range(self.num_experts):
                key = f"language_model.backbone.layers.{layer}.mixer.experts.{expert}.{projection}.weight"
                w = checkpoint.raw(key)
                s = checkpoint.raw(key + "_scale")
                if (w.dtype != torch.uint8 or list(w.shape) != [(rows+3)//4, cols] or
                    s.dtype != torch.float16 or list(s.shape) != [rows, (cols+self.group-1)//self.group]):
                    raise ValueError(f"Invalid expert code/scale tensor: {key}")
                codes.append(w.numpy())
                scales.append(s.view(torch.uint16).numpy())
            self.arrays.extend([codes, scales])
            self.pointers.extend([np.array([a.ctypes.data for a in codes], dtype=np.uintp),
                                  np.array([a.ctypes.data for a in scales], dtype=np.uintp)])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if (hidden_states.ndim != 2 or hidden_states.shape[1] != self.dim or
                top_k_index.ndim != 2 or top_k_index.shape[0] != hidden_states.shape[0] or
                top_k_index.shape != top_k_weights.shape or top_k_index.numel() == 0):
            raise ValueError("Invalid expert input or routing shape")
        values = hidden_states.float().contiguous().numpy()
        ids = top_k_index.long().contiguous().numpy()
        routing = top_k_weights.float().contiguous().numpy()
        if ids.min() < 0 or ids.max() >= self.num_experts:
            raise ValueError("Routing index outside expert bank")
        tokens, topk = ids.shape
        hidden = np.empty((tokens*topk, self.intermediate), dtype=np.float32)
        partial = np.empty((tokens*topk, self.dim), dtype=np.float32)
        output = np.empty((tokens, self.dim), dtype=np.float32)
        self.kernels.lib.nemotron_experts_neon(values, ids, routing, *self.pointers,
            hidden, partial, output, tokens, self.dim, self.intermediate, topk, self.group)
        return torch.from_numpy(output).to(hidden_states.dtype)


class NativeInt4Linear(torch.nn.Module):
    def __init__(self, overlay, name, kernels, bias=None):
        super().__init__()
        packed, scales = overlay.tensors(name)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self.codes = packed.numpy()
        self.scale_bits = scales.view(torch.uint16).numpy()
        self.out_features, self.in_features = overlay.manifest["tensors"][name]["shape"]
        self.group = overlay.manifest["group_size"]
        self.kernels = kernels

    def forward(self, x):
        values = x.reshape(-1, self.in_features).float().contiguous().numpy()
        out = torch.from_numpy(self.kernels.dense_int4(self.codes, self.scale_bits, values,
                                                       self.in_features, self.group))
        if self.bias is not None:
            out += self.bias.float()
        return out.reshape(x.shape[:-1]+(self.out_features,)).to(x.dtype)


def _linear(weight, kernels, bias=None, overlay=None, name=""):
    if overlay is not None and name in overlay.manifest["tensors"]:
        if list(weight.shape) != overlay.manifest["tensors"][name]["shape"]:
            raise ValueError(f"INT4 overlay disagrees with model shape: {name}")
        overlay.used.add(name)
        return NativeInt4Linear(overlay, name, kernels, bias)
    return NativeLinear(weight, kernels, bias)


def _replace_linears(module, kernels, overlay=None, prefix=""):
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, _linear(child.weight, kernels, child.bias, overlay, prefix+name+".weight"))
        else:
            _replace_linears(child, kernels, overlay, prefix+name+".")


def load_nemotron(directory, threads=10, report=None, int4_overlay=None):
    try:
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM
    except ImportError as error:
        raise ImportError("Install litespark-inference[nemotron] for the Nemotron backend") from error
    torch.set_num_threads(threads)
    kernels = NativeKernels(threads)
    if report is not None:
        report["openmp"] = kernel.has_omp()
        report["kernel_threads"] = kernel.max_threads()
    checkpoint = Checkpoint(directory)
    overlay = None
    if int4_overlay is not None:
        from .nemotron_int4 import Int4Overlay
        overlay = Int4Overlay(int4_overlay, source=directory)
        overlay.used = set()
        if report is not None:
            report["dense_int4"] = {key: overlay.manifest[key] for key in
                                    ("group_size", "source_tensor_bytes", "packed_tensor_bytes")}
    config = NemotronHConfig(**checkpoint.config["llm_config"])
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    with torch.device("meta"):
        model = NemotronHForCausalLM(config)
    for i, block in enumerate(model.model.layers):
        if block.block_type == "moe":
            block.mixer.experts = NativeExperts(checkpoint, i, config, kernels)
        prefix = f"language_model.backbone.layers.{i}."
        block.load_state_dict({k: checkpoint.raw(prefix+k) for k in block.state_dict()}, strict=True, assign=True)
        _replace_linears(block, kernels, overlay, prefix)
        if block.block_type == "linear_attention":
            _native_convolution(block.mixer, kernels)
        if report is not None:
            def before(module, inputs):
                module._started = time.monotonic()
            def after(module, inputs, output, idx=i):
                seconds = time.monotonic()-module._started
                report.setdefault("layers", []).append({"layer": idx, "tokens": inputs[0].shape[1],
                                                        "type": module.block_type, "seconds": seconds})
                if idx % 8 == 0 or idx == config.num_hidden_layers-1:
                    print(f"layer={idx} tokens={inputs[0].shape[1]} seconds={seconds:.3f}", flush=True)
            block.register_forward_pre_hook(before)
            block.register_forward_hook(after)
        if i % 16 == 0:
            print(f"mapped layer {i}", flush=True)
    model.model.embeddings.weight = torch.nn.Parameter(checkpoint.raw("language_model.backbone.embeddings.weight"), requires_grad=False)
    model.model.norm_f.weight = torch.nn.Parameter(checkpoint.raw("language_model.backbone.norm_f.weight"), requires_grad=False)
    model.lm_head = _linear(checkpoint.raw("language_model.lm_head.weight"), kernels,
                            overlay=overlay, name="language_model.lm_head.weight")
    model._nemotron_checkpoint = checkpoint
    model._nemotron_int4 = overlay
    if overlay is not None and overlay.used != set(overlay.manifest["tensors"]):
        raise ValueError("INT4 overlay contains tensors unused by the text model")
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError("Uninitialized model tensor")
    return model.eval().requires_grad_(False)


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="What is 17 times 19? Reply with only the integer.")
    parser.add_argument("--expect", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument("--dense-int4", help="Path to a grouped INT4 dense-weight overlay")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {"backend": "public Litespark ARM/NEON + Accelerate; CPU recurrent reference",
              "prompt": args.prompt, "threads": args.threads, "platform": platform.platform()}
    started = time.monotonic()
    model = load_nemotron(args.model, args.threads, report, args.dense_int4)
    report["load_seconds"] = time.monotonic()-started
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    report["prompt_tokens"] = tokens.shape[1]
    config = json.loads((Path(args.model)/"generation_config.json").read_text())
    eos = config.get("eos_token_id", tokenizer.eos_token_id)
    eos = set(eos if isinstance(eos, list) else [eos])
    generated, timings, cache = [], [], None
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for i in range(args.max_new_tokens):
                step = time.monotonic()
                result = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
                if not torch.isfinite(result.logits).all():
                    raise RuntimeError("Nonfinite logits")
                token = result.logits[0,-1].argmax().item()
                cache = result.past_key_values
                generated.append(token)
                timings.append(time.monotonic()-step)
                print(json.dumps({"step": i, "text": tokenizer.decode(generated), "seconds": timings[-1]}), flush=True)
                if token in eos:
                    break
                tokens = torch.tensor([[token]])
        report.update(text=tokenizer.decode(generated, skip_special_tokens=True).strip(), token_ids=generated,
                      step_seconds=timings, seconds=time.monotonic()-started,
                      decode_tokens_per_second=None if len(timings)<2 else (len(timings)-1)/sum(timings[1:]),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report["passed"] = args.expect is None or report["text"] == args.expect
    except Exception as error:
        report.update(error=repr(error), passed=False)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k != "layers"}, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
