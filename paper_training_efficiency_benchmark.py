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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            'paper_reproduction': {
                'fused_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []},
                'vanilla_bitlinear': {'batch_sizes': [], 'times': [], 'memory': []}
            },
            'pythia_comparison': {
                'matmul_free_370m': {'batch_sizes': [], 'times': [], 'memory': []},
                'pythia_410m': {'batch_sizes': [], 'times': [], 'memory': []},
                'matmul_free_1_3b': {'batch_sizes': [], 'times': [], 'memory': []},
                'pythia_1_4b': {'batch_sizes': [], 'times': [], 'memory': []}
            }
        }
    
    def create_matmul_free_model(self, size='1.3B', use_fused=True, model_path=None):
        """Create MatMul-free model matching paper specs"""
        if size == '370M':
            config = HGRNBitConfig(
                vocab_size=50000,
                hidden_size=1024,
                num_hidden_layers=24,
                num_heads=8,
                max_position_embeddings=2048,
            )
        elif size == '1.3B':
            config = HGRNBitConfig(
                vocab_size=50000,
                hidden_size=2048,
                num_hidden_layers=24,
                num_heads=16,
                max_position_embeddings=2048,
            )
        else:
            raise ValueError(f"Unsupported size: {size}")
        
        # Load trained model if path provided
        if model_path and Path(model_path).exists():
            if Path(model_path).is_dir():
                model = HGRNBitForCausalLM.from_pretrained(model_path)
            else:
                checkpoint = torch.load(model_path, map_location='cpu')
                model = HGRNBitForCausalLM(config)
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = HGRNBitForCausalLM(config)
        
        model = model.to(self.device)
        
        # Replace with vanilla BitLinear if needed
        if not use_fused:
            self._replace_fused_with_vanilla(model)
        
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
    
    def run_paper_reproduction(self, model_path_1_3b=None, batch_sizes=[1, 2, 4, 8, 16, 28]):
        """
        Reproduce exact paper test: Fused vs Vanilla BitLinear on 1.3B model
        Paper result: batch_size=28, 1.52s->1.21s (25.6% speedup), 82GB->32GB (61% reduction)
        """
        print("🔬 REPRODUCING PAPER SECTION 5.1 - Training Efficiency on GPU")
        print("📋 Paper setup: input_size=1024, seq_length=1024, NVIDIA A100 80GB")
        print(f"📊 Testing batch sizes: {batch_sizes}")
        print("=" * 80)
        
        # Test 1: Fused BitLinear (Paper's optimized version)
        print("\n🔥 Testing Fused BitLinear (Paper's Optimized Implementation)")
        fused_model, config = self.create_matmul_free_model('1.3B', use_fused=True, model_path=model_path_1_3b)
        param_count = sum(p.numel() for p in fused_model.parameters())
        print(f"📈 Model: {param_count/1e9:.1f}B parameters")
        
        self._benchmark_model_series(fused_model, 'paper_reproduction', 'fused_bitlinear', batch_sizes)
        del fused_model
        torch.cuda.empty_cache()
        
        # Test 2: Vanilla BitLinear (Paper's baseline)
        print("\n📊 Testing Vanilla BitLinear (Paper's Baseline)")
        vanilla_model, _ = self.create_matmul_free_model('1.3B', use_fused=False, model_path=model_path_1_3b)
        self._benchmark_model_series(vanilla_model, 'paper_reproduction', 'vanilla_bitlinear', batch_sizes)
        del vanilla_model
        torch.cuda.empty_cache()

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

    def plot_results(self, save_plots=True):
        """Create plots matching paper's Figure 3(a-b)"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Training Time vs Batch Size (Figure 3a)
        fused_data = self.results['paper_reproduction']['fused_bitlinear']
        vanilla_data = self.results['paper_reproduction']['vanilla_bitlinear']

        if fused_data['batch_sizes']:
            ax1.plot(fused_data['batch_sizes'], fused_data['times'], 'o-', label='Fused BitLinear', linewidth=2)
        if vanilla_data['batch_sizes']:
            ax1.plot(vanilla_data['batch_sizes'], vanilla_data['times'], 's-', label='Vanilla BitLinear', linewidth=2)

        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Training Time (s/iteration)')
        ax1.set_title('Training Time vs Batch Size\n(Reproducing Paper Figure 3a)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Memory Usage vs Batch Size (Figure 3b)
        if fused_data['batch_sizes']:
            ax2.plot(fused_data['batch_sizes'], fused_data['memory'], 'o-', label='Fused BitLinear', linewidth=2)
        if vanilla_data['batch_sizes']:
            ax2.plot(vanilla_data['batch_sizes'], vanilla_data['memory'], 's-', label='Vanilla BitLinear', linewidth=2)

        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Peak Memory (GB)')
        ax2.set_title('Memory Usage vs Batch Size\n(Reproducing Paper Figure 3b)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig('paper_reproduction_plots.png', dpi=300, bbox_inches='tight')
            print("📊 Plots saved to paper_reproduction_plots.png")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Training Efficiency Benchmark - Paper Reproduction + Pythia Comparison')
    parser.add_argument('--model_370m', type=str, help='Path to trained 370M MatMul-free model')
    parser.add_argument('--model_1_3b', type=str, help='Path to trained 1.3B MatMul-free model')
    parser.add_argument('--batch_sizes', nargs='+', type=int, default=[1, 2, 4, 8, 16, 28],
                       help='Batch sizes to test (default: 1 2 4 8 16 28)')
    parser.add_argument('--paper_only', action='store_true', help='Run only paper reproduction tests')
    parser.add_argument('--pythia_only', action='store_true', help='Run only Pythia comparison tests')
    parser.add_argument('--output', type=str, default='training_efficiency_results.json', help='Output JSON file')

    args = parser.parse_args()

    benchmark = PaperTrainingBenchmark()

    if not args.pythia_only:
        benchmark.run_paper_reproduction(args.model_1_3b, args.batch_sizes)
        benchmark.analyze_paper_results()

    if not args.paper_only:
        benchmark.run_pythia_comparison(args.model_370m, args.model_1_3b, args.batch_sizes)

    benchmark.save_results(args.output)
    benchmark.plot_results()


if __name__ == "__main__":
    main()
