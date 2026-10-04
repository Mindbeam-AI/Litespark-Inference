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

    python -m unittest discover -s tests -p test_nemotron_cpu.py -v

Python API:

    from litespark_inference import load_nemotron
    model = load_nemotron('/path/to/checkpoint', threads=10)

Use torch.inference_mode() around inference. Checkpoint mappings remain
owned by the model for its lifetime; moving this CPU model to a GPU is
unsupported. The repository's LICENSE applies to the inference code.
