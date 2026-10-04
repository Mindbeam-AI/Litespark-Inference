"""Grouped INT4 overlays for the retained dense Nemotron projections.

Packed codes are q+8, two consecutive input columns per byte, low nibble
first. FP16 group scales multiply q; reconstructed weights round to BF16.
The source checkpoint is never modified. No activation calibration is used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


FORMAT = "litespark-nemotron-dense-int4-v1"
PROJECTIONS = re.compile(
    r"language_model\.backbone\.layers\.\d+\.mixer\."
    r"(in_proj|out_proj|[qkvo]_proj|fc[12]_latent_proj|shared_experts\.(up_proj|down_proj))\.weight$"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


@torch.no_grad()
def quantize(weight, group_size=128, row_chunk=64, device="cpu"):
    if weight.ndim != 2 or not 2 <= group_size <= 4096 or group_size % 2 or row_chunk < 1:
        raise ValueError("INT4 requires a matrix, positive row chunk, and even group size in [2, 4096]")
    rows, cols = weight.shape
    groups = (cols+group_size-1)//group_size
    packed = torch.empty((rows, (cols+1)//2), dtype=torch.uint8)
    scales = torch.empty((rows, groups), dtype=torch.float16)
    squared_error = squared_weight = 0.
    for start in range(0, rows, row_chunk):
        chunk = weight[start:start+row_chunk].to(device=device, dtype=torch.float32)
        if not torch.isfinite(chunk).all():
            raise ValueError("Nonfinite dense weights")
        padded = torch.nn.functional.pad(chunk, (0, groups*group_size-cols))
        x = padded.reshape(-1, groups, group_size)
        maximum = x.abs().amax(-1, keepdim=True)
        best_error = torch.full_like(maximum, float("inf"))
        best_scale = torch.ones_like(maximum)
        best_q = torch.zeros_like(x)

        def candidate(scale):
            scale = scale.clamp_min(2**-24).half().float()
            q = (x/scale).round().clamp(-7, 7)
            error = (x-(q*scale).bfloat16().float()).square().sum(-1, keepdim=True)
            return scale, q, error

        for clipping in (1., .99, .97, .95, .925, .9, .875, .85, .8, .75):
            scale, q, error = candidate(maximum*(clipping/7))
            better = error < best_error
            best_error = torch.minimum(error, best_error)
            best_scale = torch.where(better, scale, best_scale)
            best_q = torch.where(better, q, best_q)
        for _ in range(2):
            fitted = (x*best_q).sum(-1, keepdim=True)/best_q.square().sum(-1, keepdim=True).clamp_min(1)
            scale, q, error = candidate(fitted)
            better = error < best_error
            best_error = torch.minimum(error, best_error)
            best_scale = torch.where(better, scale, best_scale)
            best_q = torch.where(better, q, best_q)
        codes = (best_q.reshape(chunk.shape[0], -1)[:, :cols]+8).to(torch.uint8)
        if cols % 2:
            codes = torch.nn.functional.pad(codes, (0, 1), value=8)
        packed[start:start+chunk.shape[0]] = (codes[:, ::2] | (codes[:, 1::2] << 4)).cpu()
        scales[start:start+chunk.shape[0]] = best_scale.squeeze(-1).half().cpu()
        squared_error += best_error.sum().item()
        squared_weight += chunk.square().sum().item()
    return packed, scales, {"relative_rmse": (squared_error/max(squared_weight, 1e-30))**.5}


def expand(packed, scales, shape, group_size):
    rows, cols = shape
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, -1)[:, :cols]
    return ((codes.float()-8)*scales.float().repeat_interleave(group_size, dim=1)[:, :cols]).bfloat16()


class Int4Overlay:
    def __init__(self, directory, source=None):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root/"int4_manifest.json").read_text())
        m = self.manifest
        if (m.get("format") != FORMAT or not m.get("complete") or m.get("zero_point") != 8 or
                m.get("dequant_dtype") != "bfloat16" or m.get("scale_dtype") != "float16" or
                m.get("packing") != "two input-column nibbles per byte; low nibble first" or
                not 2 <= m["group_size"] <= 4096 or m["group_size"] % 2):
            raise ValueError("Unsupported or incomplete INT4 overlay")
        if source is not None:
            for filename, expected in m["source"].items():
                if sha256(Path(source)/filename) != expected:
                    raise ValueError(f"INT4 overlay source mismatch: {filename}")
        self.handles = {}

    def tensors(self, name):
        entry = self.manifest["tensors"][name]
        filename = entry["shard"]
        if filename not in self.handles:
            path = (self.root/filename).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("INT4 shard path escapes overlay directory")
            self.handles[filename] = safe_open(path, framework="pt", device="cpu")
        handle = self.handles[filename]
        packed, scales = handle.get_tensor(name), handle.get_tensor(name+"_int4_scale")
        rows, cols = entry["shape"]
        group = self.manifest["group_size"]
        if (packed.dtype != torch.uint8 or packed.shape != (rows, (cols+1)//2) or
                scales.dtype != torch.float16 or scales.shape != (rows, (cols+group-1)//group)):
            raise ValueError(f"Invalid INT4 tensor shape/dtype: {name}")
        return packed, scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.output.resolve() == args.model.resolve() or args.output.resolve().is_relative_to(args.model.resolve()):
        raise ValueError("Write the overlay outside the source checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    index = json.loads((args.model/"model.safetensors.index.json").read_text())["weight_map"]
    selected = [name for name in index if PROJECTIONS.fullmatch(name) or name == "language_model.lm_head.weight"]
    if not selected:
        raise ValueError("No supported dense projections found")
    source_manifest = json.loads((args.model/"distillery_manifest.json").read_text())
    manifest = {"format": FORMAT, "complete": False, "group_size": args.group_size,
        "zero_point": 8, "qmin": -7, "qmax": 7, "scale_dtype": "float16", "dequant_dtype": "bfloat16",
        "packing": "two input-column nibbles per byte; low nibble first",
        "algorithm": "weight MSE clipping search and two least-squares scale refinements",
        "source": {name: sha256(args.model/name) for name in
                   ("config.json", "model.safetensors.index.json", "distillery_manifest.json")},
        "source_identity": source_manifest["identity"],
        "source_shard_sha256": source_manifest["shard_sha256"],
        "tensors": {}, "shard_sha256": {}, "source_tensor_bytes": 0, "packed_tensor_bytes": 0}
    buckets = defaultdict(list)
    for name in selected:
        bucket = int(name.split(".")[3]) if ".layers." in name else 99999
        buckets[bucket].append(name)
    handles = {}
    for bucket, names in sorted(buckets.items()):
        start = time.monotonic()
        saved = {}
        filename = f"dense-{bucket:05d}.safetensors"
        for name in sorted(names):
            shard = index[name]
            if shard not in handles:
                handles[shard] = safe_open(args.model/shard, framework="pt", device="cpu")
            source = handles[shard].get_tensor(name)
            packed, scales, error = quantize(source, args.group_size,
                                            row_chunk=64 if args.device == "cpu" else 1024,
                                            device=args.device)
            saved[name], saved[name+"_int4_scale"] = packed, scales
            manifest["tensors"][name] = {"shape": list(source.shape), "shard": filename, **error}
            manifest["source_tensor_bytes"] += source.numel()*source.element_size()
            manifest["packed_tensor_bytes"] += packed.numel()+scales.numel()*2
        path = args.output/filename
        save_file(saved, path, metadata={"format": "pt"})
        with safe_open(path, framework="pt") as check:
            for name, value in saved.items():
                if not torch.equal(check.get_tensor(name), value):
                    raise ValueError(f"Serialized INT4 mismatch: {name}")
        manifest["shard_sha256"][filename] = sha256(path)
        atomic_json(args.output/"int4_manifest.json", manifest)
        print(json.dumps({"shard": filename, "seconds": time.monotonic()-start,
                          "packed_bytes_so_far": manifest["packed_tensor_bytes"]}), flush=True)
    manifest["complete"] = True
    atomic_json(args.output/"int4_manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in
                     ("tensors", "shard_sha256", "source", "source_identity", "source_shard_sha256")}), flush=True)


if __name__ == "__main__":
    main()
