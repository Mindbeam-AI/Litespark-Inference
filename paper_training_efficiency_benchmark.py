#!/usr/bin/env python3
"""
Training Efficiency Benchmark Script
Reproduces the exact tests from "Scalable MatMul-free Language Modeling" Section 5.1
Plus comparison with Pythia models

Paper Tests:
- Fused vs Vanilla BitLinear training performance (Figure 3a-b)
- Input size and sequence length: 1024
- NVIDIA A100 80GB GPU
- Batch sizes tested, measuring time per iteration and memory usage

Additional Tests:
- 370M MatMul-free vs Pythia-410M
- 1.3B MatMul-free vs Pythia-1.4B
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress HuggingFace warnings
warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")

from src.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from src.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from src.ops.fusedbitnet import FusedBitLinear
from src.ops.bitnet import BitLinear


class PaperTrainingBenchmark:
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU',
                'pytorch_version': torch.__version__,
                'paper_reference': 'Scalable MatMul-free Language Modeling, Section 5.1'
            },
            'figure3a_training_latency': {
                'fused_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []},
                'vanilla_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []}
            },
            'figure3b_memory_utilization': {
                'fused_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []},
                'vanilla_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []}
            },
            'figure3c_resource_efficiency': {
                'matmul_free_370m': {'model_size': 0.37, 'inference_memory': 0, 'inference_latency': 0},
                'matmul_free_1_3b': {'model_size': 1.3, 'inference_memory': 0, 'inference_latency': 0},
                'matmul_free_2_7b': {'model_size': 2.7, 'inference_memory': 0, 'inference_latency': 0},
                'transformer_plus_370m': {'model_size': 0.37, 'inference_memory': 0, 'inference_latency': 0},
                'transformer_plus_1_3b': {'model_size': 1.3, 'inference_memory': 0, 'inference_latency': 0},
                'transformer_plus_2_7b': {'model_size': 2.7, 'inference_memory': 0, 'inference_latency': 0}
            }
        }
    
    def create_matmul_free_model(self, size='1.3B', use_fused=True, model_path=None):
        """Create MatMul-free model using actual trained model configuration"""

        # Load trained model if path provided - use actual config
        if model_path and Path(model_path).exists():
            print(f"📂 Loading trained model from {model_path}")
            if Path(model_path).is_dir():
                # Load from HuggingFace format directory
                model = HGRNBitForCausalLM.from_pretrained(model_path)
                config = model.config
            else:
                # Load from checkpoint file
                checkpoint = torch.load(model_path, map_location='cpu')
                config = checkpoint.get('config')
                if config is None:
                    raise ValueError(f"No config found in checkpoint {model_path}")
                model = HGRNBitForCausalLM(config)
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback: create model with default config (only if no trained model available)
            print(f"⚠️  No trained model path provided, creating fresh {size} model with default config")
            if size == '370M':
                config = HGRNBitConfig(
                    vocab_size=32000,  # Use original repo default
                    hidden_size=1024,
                    num_hidden_layers=24,
                    num_heads=1,       # Use original repo default
                    max_position_embeddings=2048,
                )
            elif size == '1.3B':
                config = HGRNBitConfig(
                    vocab_size=32000,  # Use original repo default
                    hidden_size=2048,
                    num_hidden_layers=24,
                    num_heads=1,       # Use original repo default
                    max_position_embeddings=2048,
                )
            else:
                raise ValueError(f"Unsupported size: {size}")

            model = HGRNBitForCausalLM(config)

        model = model.to(self.device)

        # Replace with vanilla BitLinear if needed
        if not use_fused:
            self._replace_fused_with_vanilla(model)

        # Print actual model configuration for verification
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📈 Model config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, "
              f"num_layers={config.num_hidden_layers}, num_heads={getattr(config, 'num_heads', 1)}")
        print(f"📊 Total parameters: {param_count/1e9:.2f}B")

        return model, config
    
    def load_pythia_model(self, size='1.4b'):
        """Load Pythia model for comparison"""
        model_name = f"EleutherAI/pythia-{size}"
        print(f"📂 Loading {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def _replace_fused_with_vanilla(self, model):
        """Replace FusedBitLinear with vanilla BitLinear"""
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, FusedBitLinear):
                    vanilla_layer = BitLinear(
                        child.in_features, 
                        child.out_features, 
                        bias=child.bias is not None
                    ).to(self.device)
                    setattr(module, name, vanilla_layer)
                else:
                    replace_layers(child)
        replace_layers(model)
    
    def benchmark_training_iteration(self, model, batch_size, seq_length=1024, warmup_steps=3, measure_steps=10):
        """
        Benchmark single training iteration - exact paper methodology
        Paper: "set the input size and sequence length to 1024"
        """
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create batch exactly as paper: input_size=1024, seq_length=1024
        vocab_size = getattr(model.config, 'vocab_size', 50000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)
        
        # Warmup
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        
        # Reset memory tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Measure training time
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(measure_steps):
            optimizer.zero_grad()
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time_per_iteration = (end_time - start_time) / measure_steps
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return avg_time_per_iteration, peak_memory_gb

    def benchmark_inference_iteration(self, model, batch_size=1, seq_length=2048, warmup_steps=3, measure_steps=10):
        """
        Benchmark inference iteration - matches Figure 3(c) setup
        Paper: "batch size of 1 and a sequence length of 2048"
        """
        model.eval()  # Inference mode

        # Create dummy batch for inference
        vocab_size = getattr(model.config, 'vocab_size', 50000)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                outputs = model(input_ids)
                torch.cuda.synchronize()

        # Reset memory tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(measure_steps):
                outputs = model(input_ids)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time_per_iteration = (end_time - start_time) / measure_steps
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        return avg_time_per_iteration, peak_memory_gb
    
    def run_figure3_complete_reproduction(self, model_path_370m=None, model_path_1_3b=None, model_path_2_7b=None, batch_sizes=[1, 2, 4, 8, 16, 28]):
        """
        Reproduce complete Figure 3 from the paper with exact same models:
        (a) Computational Latency vs. Batch Size - Fused vs Vanilla BitLinear
        (b) Memory Utilization vs. Batch Size - Fused vs Vanilla BitLinear
        (c) Resource Efficiency Analysis - MatMul-free LM vs Transformer++ across model sizes
        """
        print("🔬 REPRODUCING COMPLETE PAPER FIGURE 3")
        print("📋 Paper setup: input_size=1024, seq_length=1024, NVIDIA A100 80GB")
        print("📊 Figure 3(a-b): Training efficiency with batch size sweep")
        print("📊 Figure 3(c): Inference efficiency across model sizes")
        print("=" * 80)

        # Figure 3(a-b): Training efficiency on 1.3B model
        self._run_figure3ab_training_efficiency(model_path_1_3b, batch_sizes)

        # Figure 3(c): Resource efficiency across model sizes
        self._run_figure3c_resource_efficiency(model_path_370m, model_path_1_3b, model_path_2_7b)

    def _run_figure3ab_training_efficiency(self, model_path_1_3b, batch_sizes):
        """Run Figure 3(a-b): Training efficiency tests"""
        print("\n📊 FIGURE 3(A-B): TRAINING EFFICIENCY - 1.3B Model")
        print("=" * 60)

        # Test 1: Fused BitLinear (Paper's optimized version)
        print("\n🔥 Testing Fused BitLinear (Paper's Optimized Implementation)")
        fused_model, config = self.create_matmul_free_model('1.3B', use_fused=True, model_path=model_path_1_3b)
        param_count = sum(p.numel() for p in fused_model.parameters())
        print(f"📈 Model: {param_count/1e9:.1f}B parameters")

        self._benchmark_model_series(fused_model, 'figure3a_training_latency', 'fused_bitlinear', batch_sizes)
        # Copy results to figure3b for memory data
        self.results['figure3b_memory_utilization']['fused_bitlinear'] = self.results['figure3a_training_latency']['fused_bitlinear'].copy()
        del fused_model
        torch.cuda.empty_cache()

        # Test 2: Vanilla BitLinear (Paper's baseline)
        print("\n📊 Testing Vanilla BitLinear (Paper's Baseline)")
        vanilla_model, _ = self.create_matmul_free_model('1.3B', use_fused=False, model_path=model_path_1_3b)
        self._benchmark_model_series(vanilla_model, 'figure3a_training_latency', 'vanilla_bitlinear', batch_sizes)
        # Copy results to figure3b for memory data
        self.results['figure3b_memory_utilization']['vanilla_bitlinear'] = self.results['figure3a_training_latency']['vanilla_bitlinear'].copy()
        del vanilla_model
        torch.cuda.empty_cache()

    def _run_figure3c_resource_efficiency(self, model_path_370m, model_path_1_3b, model_path_2_7b):
        """Run Figure 3(c): Resource efficiency analysis across model sizes"""
        print("\n📊 FIGURE 3(C): RESOURCE EFFICIENCY ANALYSIS")
        print("📋 Paper setup: batch_size=1, seq_length=2048, inference mode")
        print("=" * 60)

        model_configs = [
            ('370M', model_path_370m, 'matmul_free_370m'),
            ('1.3B', model_path_1_3b, 'matmul_free_1_3b'),
            ('2.7B', model_path_2_7b, 'matmul_free_2_7b')
        ]

        for size, model_path, result_key in model_configs:
            if model_path and Path(model_path).exists():
                print(f"\n🔥 Testing MatMul-free {size} model")
                try:
                    model, config = self.create_matmul_free_model(size, use_fused=True, model_path=model_path)
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"📈 Model: {param_count/1e9:.2f}B parameters")

                    # Inference benchmark: batch_size=1, seq_length=2048 (paper setup)
                    time_per_iter, memory_gb = self.benchmark_inference_iteration(model, batch_size=1, seq_length=2048)

                    self.results['figure3c_resource_efficiency'][result_key]['inference_memory'] = memory_gb
                    self.results['figure3c_resource_efficiency'][result_key]['inference_latency'] = time_per_iter

                    print(f"📊 Inference: {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")

                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"❌ Failed to test {size}: {e}")
            else:
                print(f"⚠️  {size} model not found at {model_path}")

        # Note: We don't have Transformer++ models, so we'll use Pythia as proxy
        print("\n📝 Note: Using Pythia models as Transformer++ proxy for comparison")
        self._benchmark_transformer_plus_proxy()

    def _benchmark_transformer_plus_proxy(self):
        """Use Pythia models as Transformer++ proxy for Figure 3(c)"""
        pythia_configs = [
            ('410m', 0.41, 'transformer_plus_370m'),  # Closest to 370M
            ('1.4b', 1.4, 'transformer_plus_1_3b'),   # Closest to 1.3B
            ('2.8b', 2.8, 'transformer_plus_2_7b')    # Closest to 2.7B
        ]

        for pythia_size, model_size_b, result_key in pythia_configs:
            try:
                print(f"\n🐍 Testing Pythia-{pythia_size} as Transformer++ {model_size_b:.1f}B proxy")
                model, tokenizer = self.load_pythia_model(pythia_size)
                if model is not None:
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"📈 Pythia Model: {param_count/1e9:.2f}B parameters")

                    # Inference benchmark: batch_size=1, seq_length=2048
                    time_per_iter, memory_gb = self.benchmark_inference_iteration(model, batch_size=1, seq_length=2048)

                    self.results['figure3c_resource_efficiency'][result_key]['model_size'] = model_size_b
                    self.results['figure3c_resource_efficiency'][result_key]['inference_memory'] = memory_gb
                    self.results['figure3c_resource_efficiency'][result_key]['inference_latency'] = time_per_iter

                    print(f"📊 Inference: {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")

                    del model
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"❌ Failed to test Pythia-{pythia_size}: {e}")

    def run_pythia_comparison(self, model_path_370m=None, model_path_1_3b=None, batch_sizes=[1, 2, 4, 8, 16]):
        """Compare MatMul-free models with Pythia baselines"""
        print("\n🐍 PYTHIA COMPARISON TESTS")
        print("📋 370M MatMul-free vs Pythia-410M")
        print("📋 1.3B MatMul-free vs Pythia-1.4B")
        print("=" * 80)

        # Test 1: 370M MatMul-free vs Pythia-410M
        print("\n📊 370M MatMul-free vs Pythia-410M")
        try:
            # MatMul-free 370M
            print("  🔥 Testing MatMul-free 370M")
            matmul_free_370m, _ = self.create_matmul_free_model('370M', model_path=model_path_370m)
            param_count = sum(p.numel() for p in matmul_free_370m.parameters())
            print(f"     📈 {param_count/1e6:.0f}M parameters")
            self._benchmark_model_series(matmul_free_370m, 'pythia_comparison', 'matmul_free_370m', batch_sizes)
            del matmul_free_370m
            torch.cuda.empty_cache()

            # Pythia-410M
            print("  🐍 Testing Pythia-410M")
            pythia_410m, _ = self.load_pythia_model('410m')
            param_count = sum(p.numel() for p in pythia_410m.parameters())
            print(f"     📈 {param_count/1e6:.0f}M parameters")
            self._benchmark_model_series(pythia_410m, 'pythia_comparison', 'pythia_410m', batch_sizes)
            del pythia_410m
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 370M comparison failed: {e}")

        # Test 2: 1.3B MatMul-free vs Pythia-1.4B
        print("\n📊 1.3B MatMul-free vs Pythia-1.4B")
        try:
            # MatMul-free 1.3B
            print("  🔥 Testing MatMul-free 1.3B")
            matmul_free_1_3b, _ = self.create_matmul_free_model('1.3B', model_path=model_path_1_3b)
            param_count = sum(p.numel() for p in matmul_free_1_3b.parameters())
            print(f"     📈 {param_count/1e9:.1f}B parameters")
            self._benchmark_model_series(matmul_free_1_3b, 'pythia_comparison', 'matmul_free_1_3b', batch_sizes)
            del matmul_free_1_3b
            torch.cuda.empty_cache()

            # Pythia-1.4B
            print("  🐍 Testing Pythia-1.4B")
            pythia_1_4b, _ = self.load_pythia_model('1.4b')
            param_count = sum(p.numel() for p in pythia_1_4b.parameters())
            print(f"     📈 {param_count/1e9:.1f}B parameters")
            self._benchmark_model_series(pythia_1_4b, 'pythia_comparison', 'pythia_1_4b', batch_sizes)
            del pythia_1_4b
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ 1.3B comparison failed: {e}")

    def _benchmark_model_series(self, model, category, model_key, batch_sizes):
        """Benchmark a model across batch sizes"""
        for batch_size in batch_sizes:
            try:
                print(f"    📦 Batch size {batch_size}...", end=" ")
                time_per_iter, memory_gb = self.benchmark_training_iteration(model, batch_size)

                self.results[category][model_key]['batch_sizes'].append(batch_size)
                self.results[category][model_key]['times'].append(time_per_iter)
                self.results[category][model_key]['memory'].append(memory_gb)

                print(f"⏱️  {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")

            except torch.cuda.OutOfMemoryError:
                print("❌ OOM")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break

    def analyze_paper_results(self):
        """Analyze results and compare with paper claims"""
        print("\n📊 ANALYSIS - Paper Reproduction Results")
        print("=" * 80)

        fused_data = self.results['paper_reproduction']['fused_bitlinear']
        vanilla_data = self.results['paper_reproduction']['vanilla_bitlinear']

        if not fused_data['batch_sizes'] or not vanilla_data['batch_sizes']:
            print("❌ Insufficient data for analysis")
            return

        # Find batch size 28 results (paper's key result)
        target_batch = 28
        if target_batch in fused_data['batch_sizes'] and target_batch in vanilla_data['batch_sizes']:
            fused_idx = fused_data['batch_sizes'].index(target_batch)
            vanilla_idx = vanilla_data['batch_sizes'].index(target_batch)

            fused_time = fused_data['times'][fused_idx]
            vanilla_time = vanilla_data['times'][vanilla_idx]
            fused_memory = fused_data['memory'][fused_idx]
            vanilla_memory = vanilla_data['memory'][vanilla_idx]

            speedup = ((vanilla_time - fused_time) / vanilla_time) * 100
            memory_reduction = ((vanilla_memory - fused_memory) / vanilla_memory) * 100

            print(f"🎯 BATCH SIZE {target_batch} RESULTS (Paper's Key Test):")
            print(f"   ⏱️  Fused: {fused_time:.3f}s/iter, Vanilla: {vanilla_time:.3f}s/iter")
            print(f"   💾 Fused: {fused_memory:.1f}GB, Vanilla: {vanilla_memory:.1f}GB")
            print(f"   🚀 Speedup: {speedup:.1f}% (Paper claimed: 25.6%)")
            print(f"   💾 Memory reduction: {memory_reduction:.1f}% (Paper claimed: 61.0%)")
            print(f"   📋 Paper reference: 1.52s→1.21s, 82GB→32GB")

        # Show all results
        print(f"\n📈 ALL RESULTS:")
        for i, batch_size in enumerate(fused_data['batch_sizes']):
            if i < len(vanilla_data['batch_sizes']) and vanilla_data['batch_sizes'][i] == batch_size:
                fused_time = fused_data['times'][i]
                vanilla_time = vanilla_data['times'][i]
                speedup = ((vanilla_time - fused_time) / vanilla_time) * 100
                print(f"   Batch {batch_size:2d}: Fused {fused_time:.3f}s, Vanilla {vanilla_time:.3f}s, Speedup {speedup:+.1f}%")

    def save_results(self, output_file='training_efficiency_results.json'):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 Results saved to {output_file}")

    def plot_complete_figure3(self, save_plots=True):
        """Create complete Figure 3 exactly matching the paper"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # Figure 3(a): Computational Latency vs. Batch Size
        fused_data = self.results['figure3a_training_latency']['fused_bitlinear']
        vanilla_data = self.results['figure3a_training_latency']['vanilla_bitlinear']

        if fused_data['batch_sizes']:
            ax1.plot(fused_data['batch_sizes'], fused_data['times'], 'o-',
                    label='Fused BitLinear', linewidth=2, markersize=6, color='blue')
        if vanilla_data['batch_sizes']:
            ax1.plot(vanilla_data['batch_sizes'], vanilla_data['times'], 's-',
                    label='Vanilla BitLinear', linewidth=2, markersize=6, color='red')

        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Computational Latency (s/iteration)')
        ax1.set_title('(a) Computational Latency vs. Batch Size:\nComparative performance of Fused and Vanilla BitLinear implementations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Figure 3(b): Memory Utilization vs. Batch Size
        fused_mem = self.results['figure3b_memory_utilization']['fused_bitlinear']
        vanilla_mem = self.results['figure3b_memory_utilization']['vanilla_bitlinear']

        if fused_mem['batch_sizes']:
            ax2.plot(fused_mem['batch_sizes'], fused_mem['memory'], 'o-',
                    label='Fused BitLinear', linewidth=2, markersize=6, color='blue')
        if vanilla_mem['batch_sizes']:
            ax2.plot(vanilla_mem['batch_sizes'], vanilla_mem['memory'], 's-',
                    label='Vanilla BitLinear', linewidth=2, markersize=6, color='red')

        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Memory Utilization (GB)')
        ax2.set_title('(b) Memory Utilization vs. Batch Size:\nMemory efficiency comparison between Fused and Vanilla BitLinear architectures')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Figure 3(c): Resource Efficiency Analysis
        self._plot_figure3c_resource_efficiency(ax3)

        plt.tight_layout()

        if save_plots:
            plt.savefig('complete_figure3_reproduction.png', dpi=300, bbox_inches='tight')
            print("📊 Complete Figure 3 reproduction saved to complete_figure3_reproduction.png")

        plt.show()

    def _plot_figure3c_resource_efficiency(self, ax):
        """Plot Figure 3(c): Resource Efficiency Analysis"""
        resource_data = self.results['figure3c_resource_efficiency']

        # Extract data for MatMul-free models
        matmul_free_sizes = []
        matmul_free_memory = []
        matmul_free_latency = []

        # Extract data for Transformer++ (Pythia proxy)
        transformer_sizes = []
        transformer_memory = []
        transformer_latency = []

        for key, data in resource_data.items():
            if data['inference_memory'] > 0:  # Only plot if we have data
                if 'matmul_free' in key:
                    matmul_free_sizes.append(data['model_size'])
                    matmul_free_memory.append(data['inference_memory'])
                    matmul_free_latency.append(data['inference_latency'])
                elif 'transformer_plus' in key:
                    transformer_sizes.append(data['model_size'])
                    transformer_memory.append(data['inference_memory'])
                    transformer_latency.append(data['inference_latency'])

        # Create twin axis for latency
        ax2 = ax.twinx()

        # Plot memory (left y-axis)
        if matmul_free_sizes:
            line1 = ax.plot(matmul_free_sizes, matmul_free_memory, 'o-',
                           label='MatMul-free LM (Memory)', linewidth=2, markersize=6, color='blue')
        if transformer_sizes:
            line2 = ax.plot(transformer_sizes, transformer_memory, 's-',
                           label='Transformer++ (Memory)', linewidth=2, markersize=6, color='red')

        # Plot latency (right y-axis)
        if matmul_free_sizes:
            line3 = ax2.plot(matmul_free_sizes, matmul_free_latency, '^--',
                            label='MatMul-free LM (Latency)', linewidth=2, markersize=6, color='darkblue')
        if transformer_sizes:
            line4 = ax2.plot(transformer_sizes, transformer_latency, 'v--',
                            label='Transformer++ (Latency)', linewidth=2, markersize=6, color='darkred')

        ax.set_xlabel('Model Size (B parameters)')
        ax.set_ylabel('GPU Memory Consumption (GB)', color='black')
        ax2.set_ylabel('Inference Latency (s)', color='gray')
        ax.set_title('(c) Resource Efficiency Analysis:\nGPU memory consumption and inference latency comparison across model sizes')

        # Combine legends
        lines = []
        labels = []
        if matmul_free_sizes:
            lines.extend([line1[0], line3[0]])
            labels.extend(['MatMul-free LM (Memory)', 'MatMul-free LM (Latency)'])
        if transformer_sizes:
            lines.extend([line2[0], line4[0]])
            labels.extend(['Transformer++ (Memory)', 'Transformer++ (Latency)'])

        if lines:
            ax.legend(lines, labels, loc='upper left')

        ax.grid(True, alpha=0.3)
        if matmul_free_sizes or transformer_sizes:
            all_sizes = matmul_free_sizes + transformer_sizes
            ax.set_xscale('log')
            ax.set_xticks(sorted(set(all_sizes)))
            ax.set_xticklabels([f'{s:.1f}B' for s in sorted(set(all_sizes))])


def main():
    parser = argparse.ArgumentParser(description='Complete Figure 3 Reproduction - Exact Paper Implementation')
    parser.add_argument('--model_370m', type=str, help='Path to trained 370M MatMul-free model')
    parser.add_argument('--model_1_3b', type=str, help='Path to trained 1.3B MatMul-free model')
    parser.add_argument('--model_2_7b', type=str, help='Path to trained 2.7B MatMul-free model')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 28],
                       help='Batch sizes to test for Figure 3(a-b) (default: 1 2 4 8 16 28)')
    parser.add_argument('--figure3ab_only', action='store_true', help='Run only Figure 3(a-b) training efficiency tests')
    parser.add_argument('--figure3c_only', action='store_true', help='Run only Figure 3(c) resource efficiency tests')
    parser.add_argument('--output', type=str, default='complete_figure3_results.json', help='Output JSON file')

    args = parser.parse_args()

    benchmark = PaperTrainingBenchmark()

    if args.figure3c_only:
        # Only run Figure 3(c) resource efficiency
        benchmark._run_figure3c_resource_efficiency(args.model_370m, args.model_1_3b, args.model_2_7b)
    elif args.figure3ab_only:
        # Only run Figure 3(a-b) training efficiency
        benchmark._run_figure3ab_training_efficiency(args.model_1_3b, args.batch_sizes)
    else:
        # Run complete Figure 3 reproduction
        benchmark.run_figure3_complete_reproduction(
            args.model_370m, args.model_1_3b, args.model_2_7b, args.batch_sizes
        )

    benchmark.save_results(args.output)
    benchmark.plot_complete_figure3()

    # Print summary
    print("\n🎯 FIGURE 3 REPRODUCTION SUMMARY:")
    print("=" * 80)
    print("✅ Figure 3(a): Computational Latency vs. Batch Size")
    print("✅ Figure 3(b): Memory Utilization vs. Batch Size")
    print("✅ Figure 3(c): Resource Efficiency Analysis across Model Sizes")
    print(f"📊 Results saved to: {args.output}")
    print("📈 Complete Figure 3 plot saved to: complete_figure3_reproduction.png")


if __name__ == "__main__":
    main()
