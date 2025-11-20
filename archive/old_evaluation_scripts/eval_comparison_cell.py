# Cell: Compare Step 9000 with Baseline Model (Paper-Style Evaluation)

# IMPORTANT: Set cache BEFORE any imports
import os
import sys

# Set HuggingFace cache to current directory (SageMaker folder with more space)
cache_dir = os.path.join(os.getcwd(), 'hf_cache')
os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

print(f"Setting HuggingFace cache to: {cache_dir}")

# Now import other libraries
import torch
import torch.nn as nn
sys.path.append('.')

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import json

print("="*70)
print(f"COMPARATIVE EVALUATION: MatMul-Free vs Baseline")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70)

# ===== 1. Load MatMul-Free Model (Step 9000) =====
print("\n[1/4] Loading MatMul-Free Model (Step 9000)...")
config = HGRNBitConfig(
    vocab_size=50000,
    hidden_size=1024,
    num_hidden_layers=12,
    max_position_embeddings=2048,
    num_heads=8,
    expand_ratio=1,
    hidden_ratio=4,
    rms_norm_eps=1e-6,
    use_cache=False
)

matmul_free_model = HGRNBitForCausalLM(config)

checkpoint_path = "checkpoint_step_9000.pt"
state_dict = torch.load(checkpoint_path)

# Strip "module." prefix
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[7:] if key.startswith('module.') else key
    new_state_dict[new_key] = value

matmul_free_model.load_state_dict(new_state_dict)
matmul_free_model = matmul_free_model.cuda()
matmul_free_model.eval()

matmul_free_params = sum(p.numel() for p in matmul_free_model.parameters())
print(f"✓ MatMul-Free Model: {matmul_free_params:,} parameters")

# ===== 2. Download Baseline Model =====
print("\n[2/4] Loading Baseline Model...")

# Try multiple baseline models in order of preference
baseline_options = [
    ("EleutherAI/pythia-410m-deduped", "Pythia-410M-Deduped"),
    ("EleutherAI/pythia-160m-deduped", "Pythia-160M-Deduped"),
    ("gpt2-medium", "GPT-2 Medium (355M)"),
    ("gpt2", "GPT-2 Small (117M)")
]

baseline_model = None
baseline_tokenizer = None
baseline_model_name = None

for model_id, display_name in baseline_options:
    try:
        print(f"Trying to load {display_name}...")
        baseline_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda:1",  # Use GPU 1 to avoid memory conflict
            trust_remote_code=True
        )
        baseline_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        baseline_model_name = model_id
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"✓ Baseline Model ({display_name}): {baseline_params:,} parameters")
        break
    except Exception as e:
        print(f"  Failed to load {display_name}: {str(e)[:100]}")
        continue

if baseline_model is None:
    raise RuntimeError("Could not load any baseline model. Check internet connection or install transformers: pip install --upgrade transformers")

# ===== 3. Run Eval Harness on Both Models =====
print("\n[3/4] Running lm-evaluation-harness on both models...")
print("-" * 70)

tasks = "arc_easy,hellaswag,piqa,openbookqa,winogrande"

print("\nRunning eval on MatMul-Free Model...")
print("Note: This requires lm-eval to be installed and configured properly")
print("If this fails, you'll need to run manually (instructions below)")

# Create a wrapper to make our model compatible with lm-eval
# For now, provide manual instructions since lm-eval integration is complex

print("\n" + "="*70)
print("MANUAL EVALUATION INSTRUCTIONS")
print("="*70)
print("\nTo run the full eval-harness comparison, execute these commands:\n")

print("1. Evaluate MatMul-Free Model (Step 9000):")
print(f"   lm_eval --model hf \\")
print(f"       --model_args pretrained=checkpoint_step_9000.pt,dtype=float16 \\")
print(f"       --tasks {tasks} \\")
print(f"       --batch_size 4 \\")
print(f"       --output_path ./eval_matmul_free/\n")

print("2. Evaluate Baseline Model (Pythia-410M):")
print(f"   lm_eval --model hf \\")
print(f"       --model_args pretrained={baseline_model_name},dtype=float16 \\")
print(f"       --tasks {tasks} \\")
print(f"       --batch_size 4 \\")
print(f"       --output_path ./eval_baseline/\n")

# ===== 4. Hardware Efficiency Comparison =====
print("\n[4/4] Hardware Efficiency Comparison...")
print("="*70)

print("\n--- MEMORY EFFICIENCY ---")
results = {
    'matmul_free': {},
    'baseline': {}
}

seq_lengths = [512, 1024, 2048]

print("\nMatMul-Free Model:")
for seq_len in seq_lengths:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=0)

    input_ids = torch.randint(0, 50000, (1, seq_len)).cuda(0)

    with torch.no_grad():
        outputs = matmul_free_model(input_ids)

    peak_mem = torch.cuda.max_memory_allocated(device=0) / 1e9
    results['matmul_free'][f'mem_{seq_len}'] = peak_mem
    print(f"  Seq {seq_len}: {peak_mem:.2f} GB")

print("\nBaseline Model (Pythia-410M):")
for seq_len in seq_lengths:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=1)

    input_ids = torch.randint(0, 50000, (1, seq_len)).cuda(1)

    with torch.no_grad():
        outputs = baseline_model(input_ids)

    peak_mem = torch.cuda.max_memory_allocated(device=1) / 1e9
    results['baseline'][f'mem_{seq_len}'] = peak_mem
    print(f"  Seq {seq_len}: {peak_mem:.2f} GB")

# Memory savings
print("\nMemory Savings (MatMul-Free vs Baseline):")
for seq_len in seq_lengths:
    mf_mem = results['matmul_free'][f'mem_{seq_len}']
    bl_mem = results['baseline'][f'mem_{seq_len}']
    savings = ((bl_mem - mf_mem) / bl_mem) * 100
    print(f"  Seq {seq_len}: {savings:.1f}% reduction")

print("\n--- THROUGHPUT COMPARISON ---")
batch_size = 4
seq_len = 1024
num_iters = 20

# MatMul-Free throughput
print("\nMatMul-Free Model:")
input_ids = torch.randint(0, 50000, (batch_size, seq_len)).cuda(0)
for _ in range(3):
    with torch.no_grad():
        outputs = matmul_free_model(input_ids)

torch.cuda.synchronize(0)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
with torch.no_grad():
    for _ in range(num_iters):
        outputs = matmul_free_model(input_ids)
end.record()

torch.cuda.synchronize(0)
elapsed = start.elapsed_time(end) / 1000.0
mf_throughput = (batch_size * seq_len * num_iters) / elapsed
results['matmul_free']['throughput'] = mf_throughput
print(f"  Throughput: {int(mf_throughput)} tokens/sec")

# Baseline throughput
print("\nBaseline Model (Pythia-410M):")
input_ids = torch.randint(0, 50000, (batch_size, seq_len)).cuda(1)
for _ in range(3):
    with torch.no_grad():
        outputs = baseline_model(input_ids)

torch.cuda.synchronize(1)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
with torch.no_grad():
    for _ in range(num_iters):
        outputs = baseline_model(input_ids)
end.record()

torch.cuda.synchronize(1)
elapsed = start.elapsed_time(end) / 1000.0
bl_throughput = (batch_size * seq_len * num_iters) / elapsed
results['baseline']['throughput'] = bl_throughput
print(f"  Throughput: {int(bl_throughput)} tokens/sec")

# Speedup
speedup = (mf_throughput / bl_throughput) * 100 - 100
print(f"\nSpeedup: {speedup:+.1f}%")

# ===== Summary =====
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"\nModel Sizes:")
print(f"  MatMul-Free: {matmul_free_params:,} params")
print(f"  Baseline:    {baseline_params:,} params")

print(f"\nMemory Usage (seq=2048):")
print(f"  MatMul-Free: {results['matmul_free']['mem_2048']:.2f} GB")
print(f"  Baseline:    {results['baseline']['mem_2048']:.2f} GB")
print(f"  Savings:     {((results['baseline']['mem_2048'] - results['matmul_free']['mem_2048']) / results['baseline']['mem_2048'] * 100):.1f}%")

print(f"\nThroughput (batch=4, seq=1024):")
print(f"  MatMul-Free: {int(results['matmul_free']['throughput'])} tokens/sec")
print(f"  Baseline:    {int(results['baseline']['throughput'])} tokens/sec")
print(f"  Speedup:     {speedup:+.1f}%")

print("\nFor complete accuracy comparison, run the lm-eval commands above")
print("="*70)

# Save results
with open('comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to: comparison_results.json")
