Litespark Inference

CPU inference for ternary language models. The existing BitNet and Falcon
interfaces remain available through litespark-inference.

Distillery Nemotron text inference on Apple Silicon

Install Apple's command-line developer tools, then:

    brew install libomp
    python -m pip install '.[nemotron]'
    litespark-nemotron --model /path/to/checkpoint --threads 10 \
        --prompt 'What is 17 times 19? Reply with only the integer.' \
        --expect 323 --output result.json

The checkpoint directory must contain config.json, distillery_manifest.json,
model.safetensors.index.json, all indexed shards, tokenizer files, and
generation_config.json. Model weights are obtained separately.

This backend accepts Distillery's grouped ternary Nemotron expert format:
four output-row planes per byte, multiplicative FP16 group scales, and BF16
activations. Packed experts remain memory-mapped. Dense decode uses the
public NEON BF16 GEMV; prefill uses Accelerate. Native packed-expert and
causal-convolution kernels avoid expanded expert banks and per-channel
PyTorch convolution dispatch. Transformers supplies attention, routing,
normalization, recurrent state updates, and the model/cache structure.

Text generation is supported. Vision inputs and speculative MTP decoding
are not implemented by this loader. It requires ARM64 and runs on the CPU.
The JSON report separates initial model loading, first-token latency, and
subsequent decode throughput, and records peak process RSS.

Run numerical and recurrent-cache checks with:

    python -m unittest discover -s tests -p 'test_nemotron*.py' -v

Python API:

    from litespark_inference import load_nemotron
    model = load_nemotron('/path/to/checkpoint', threads=10)

Use torch.inference_mode() around inference. Checkpoint mappings remain
owned by the model for its lifetime; moving this CPU model to a GPU is
unsupported. The repository's LICENSE applies to the inference code.

Optional grouped INT4 dense projections

Create a separate overlay; the source checkpoint remains intact:

    litespark-nemotron-int4 --model /path/to/checkpoint \
        --output /path/to/int4-overlay --group-size 128

The exporter also accepts --device cuda to accelerate quantization on a
CUDA machine. Copy the complete overlay back to the Mac before inference:

    litespark-nemotron --model /path/to/checkpoint \
        --dense-int4 /path/to/int4-overlay --output result.json

Routed experts stay ternary. Large Mamba, attention, latent, shared-expert
projections and the output head use INT4. Routers, normalization, convolution,
embeddings and recurrent parameters retain their original precision. The
overlay binds to the source config, index, and manifest and records checksums
for every output shard. INT4 is lossy and should be evaluated for each model.

Dense INT4 kernels reconstruct the same BF16 weights used for quality
evaluation. Apple CPUs with BF16 dot-product support use that instruction
when input values are exactly BF16; other cases use the FP32 NEON path.
This does not introduce int8 activation quantization.

For a matched fixed-length throughput benchmark, run both variants in
separate processes using the same options, adding --dense-int4 for INT4:

    PYTHONPATH=. python benchmarks/bench_nemotron.py \
        --model /path/to/checkpoint --output benchmark.json

The default workload uses 128 prompt tokens and two runs of 128 decode
steps after a complete prefill warmup. EOS is ignored for this timing test.
For quality evaluation with the same BF16 engine used by the baseline:

    python tools/expand_nemotron_int4_eval.py \
        --base /path/to/verified-ternary-bf16-expansion \
        --overlay /path/to/int4-overlay --output /path/to/evaluation-copy

The evaluation copy is uncompressed; it is not the deployment artifact.
