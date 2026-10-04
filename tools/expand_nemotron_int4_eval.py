"""Expand an INT4 overlay over a verified ternary BF16 evaluation derivative.

This writes a separate, uncompressed checkpoint for matched quality tests.
Unmodified shards are hard-linked. Modified shards are written atomically;
all serialized tensors are checked. Source checkpoints are never changed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"litespark_inference"))
from nemotron_int4 import Int4Overlay, expand, sha256, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    base, output = args.base.resolve(), args.output.resolve()
    if output == base or output.is_relative_to(base) or output.is_relative_to(args.overlay.resolve()):
        raise ValueError("Evaluation output must be separate from its sources")
    complete = json.loads((base/"expansion_complete.json").read_text())
    if not complete.get("complete") or not complete.get("all_serialized_tensors_checked"):
        raise ValueError("The base expansion has not passed verification")
    overlay = Int4Overlay(args.overlay)
    base_identity = json.loads((base/"evaluation_derivative.json").read_text())
    if (base_identity["packed_identity"] != overlay.manifest["source_identity"] or
            base_identity["packed_shard_sha256"] != overlay.manifest["source_shard_sha256"]):
        raise ValueError("INT4 overlay and base expansion derive from different checkpoints")
    for filename, expected in overlay.manifest["shard_sha256"].items():
        if sha256(args.overlay/filename) != expected:
            raise ValueError(f"INT4 overlay checksum mismatch: {filename}")
    index = json.loads((base/"model.safetensors.index.json").read_text())["weight_map"]
    if not set(overlay.manifest["tensors"]).issubset(index):
        raise ValueError("Overlay tensors missing from base checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Evaluation output must be empty")
    identity = {"purpose": "quality-evaluation-only; ternary and INT4 weights reconstructed as BF16",
                "base_expansion": sha256(base/"evaluation_derivative.json"),
                "overlay_manifest": sha256(args.overlay/"int4_manifest.json"),
                "overlay_source": overlay.manifest["source"]}
    atomic_json(output/"evaluation_derivative.json", identity)
    buckets = defaultdict(list)
    for name, shard in index.items():
        buckets[shard].append(name)
    receipts = {}
    with torch.inference_mode():
        for shard, names in sorted(buckets.items()):
            start = time.monotonic()
            modified = [name for name in names if name in overlay.manifest["tensors"]]
            if not modified:
                os.link(base/shard, output/shard)
                receipts[shard] = {"hardlinked": True, "modified_tensors": 0}
            else:
                with safe_open(base/shard, framework="pt") as source:
                    tensors = {name: source.get_tensor(name) for name in names}
                    for name in modified:
                        packed, scales = overlay.tensors(name)
                        shape = overlay.manifest["tensors"][name]["shape"]
                        if list(tensors[name].shape) != shape or tensors[name].dtype != torch.bfloat16:
                            raise ValueError(f"Base/overlay shape or dtype mismatch: {name}")
                        tensors[name] = expand(packed, scales, shape, overlay.manifest["group_size"])
                    temporary = output/(shard+".tmp")
                    save_file(tensors, temporary, metadata={"format": "pt"})
                    with safe_open(temporary, framework="pt") as saved:
                        for name, tensor in tensors.items():
                            if not torch.equal(saved.get_tensor(name), tensor):
                                raise ValueError(f"Serialization mismatch: {name}")
                    temporary.replace(output/shard)
                    receipts[shard] = {"hardlinked": False, "modified_tensors": len(modified),
                                       "sha256": sha256(output/shard)}
            print(json.dumps({"shard": shard, "seconds": time.monotonic()-start,
                              "modified": len(modified)}), flush=True)
    for path in base.iterdir():
        if path.is_file() and (path.suffix in (".json", ".py", ".jinja", ".model", ".txt") or
                               path.name in ("LICENSE", "NOTICE")):
            if path.name not in ("evaluation_derivative.json", "expansion_complete.json"):
                shutil.copy2(path, output/path.name)
    atomic_json(output/"int4_expansion_receipts.json", receipts)
    atomic_json(output/"expansion_complete.json", {
        **complete, "complete": True, "all_serialized_tensors_checked": True,
        "dense_int4_tensors": len(overlay.manifest["tensors"]), "overlay_manifest": identity["overlay_manifest"]})
    print(json.dumps({"complete": True, "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
