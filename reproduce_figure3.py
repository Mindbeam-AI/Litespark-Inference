#!/usr/bin/env python3
"""
Reproduce Figure 3 from the paper exactly
Focus on batch size 28 and higher batch sizes to match paper results
"""

import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from paper_training_efficiency_benchmark import PaperTrainingBenchmark

def analyze_current_results():
    """Analyze the current results and identify issues"""
    with open('training_efficiency_results.json', 'r') as f:
        results = json.load(f)
    
    print("🔍 ANALYSIS OF CURRENT RESULTS")
    print("=" * 80)
    
    # Paper reproduction analysis
    fused = results['paper_reproduction']['fused_bitlinear']
    vanilla = results['paper_reproduction']['vanilla_bitlinear']
    
    print("\n📊 PAPER REPRODUCTION (Fused vs Vanilla BitLinear)")
    print("Batch Size | Fused Time | Vanilla Time | Speedup | Fused Mem | Vanilla Mem | Mem Reduction")
    print("-" * 90)
    
    for i, batch_size in enumerate(fused['batch_sizes']):
        if i < len(vanilla['batch_sizes']) and vanilla['batch_sizes'][i] == batch_size:
            fused_time = fused['times'][i]
            vanilla_time = vanilla['times'][i]
            fused_mem = fused['memory'][i]
            vanilla_mem = vanilla['memory'][i]
            
            speedup = ((vanilla_time - fused_time) / vanilla_time) * 100
            mem_reduction = ((vanilla_mem - fused_mem) / vanilla_mem) * 100
            
            print(f"{batch_size:10d} | {fused_time:10.3f}s | {vanilla_time:11.3f}s | {speedup:6.1f}% | {fused_mem:8.1f}GB | {vanilla_mem:10.1f}GB | {mem_reduction:10.1f}%")
    
    print(f"\n🎯 PAPER CLAIMS (at batch size 28):")
    print(f"   Time: 1.52s → 1.21s (25.6% speedup)")
    print(f"   Memory: 82GB → 32GB (61.0% reduction)")
    print(f"\n❌ ISSUES IDENTIFIED:")
    print(f"   1. Missing batch size 28 test (OOM)")
    print(f"   2. Lower speedup than paper (13% vs 25.6%)")
    print(f"   3. Lower memory reduction than paper (11-20% vs 61%)")
    
    # Pythia comparison analysis
    print(f"\n🐍 PYTHIA COMPARISON ANALYSIS")
    pythia = results['pythia_comparison']
    
    print(f"\n370M MatMul-free vs Pythia-410M (batch size 1):")
    mm_time = pythia['matmul_free_370m']['times'][0]
    pythia_time = pythia['pythia_410m']['times'][0]
    print(f"   Time: {mm_time:.3f}s vs {pythia_time:.3f}s ({mm_time/pythia_time:.1f}x slower)")
    
    print(f"\n1.3B MatMul-free vs Pythia-1.4B (batch size 1):")
    mm_time = pythia['matmul_free_1_3b']['times'][0]
    pythia_time = pythia['pythia_1_4b']['times'][0]
    print(f"   Time: {mm_time:.3f}s vs {pythia_time:.3f}s ({mm_time/pythia_time:.1f}x slower)")
    
    print(f"\n💡 NOTE: MatMul-free being slower than Pythia is expected.")
    print(f"   The paper's advantage is in memory efficiency and scaling, not raw speed.")

def test_higher_batch_sizes():
    """Test higher batch sizes to reach batch size 28"""
    print(f"\n🚀 TESTING HIGHER BATCH SIZES TO REPRODUCE FIGURE 3")
    print("=" * 80)
    
    benchmark = PaperTrainingBenchmark()
    
    # Try to test batch size 28 specifically
    batch_sizes_to_test = [16, 20, 24, 28, 32]
    
    print(f"🔥 Testing Fused BitLinear at higher batch sizes...")
    
    try:
        fused_model, config = benchmark.create_matmul_free_model('1.3B', use_fused=True)
        
        for batch_size in batch_sizes_to_test:
            try:
                print(f"  📦 Testing batch size {batch_size}...", end=" ")
                time_per_iter, memory_gb = benchmark.benchmark_training_iteration(fused_model, batch_size)
                print(f"⏱️  {time_per_iter:.3f}s/iter, 💾 {memory_gb:.1f}GB")
                
                if batch_size == 28:
                    print(f"  🎯 BATCH SIZE 28 RESULT:")
                    print(f"     Time: {time_per_iter:.3f}s (Paper: 1.21s)")
                    print(f"     Memory: {memory_gb:.1f}GB (Paper: 32GB)")
                
            except torch.cuda.OutOfMemoryError:
                print("❌ OOM")
                print(f"  💡 Try reducing model size or using gradient checkpointing")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        del fused_model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Failed to test higher batch sizes: {e}")

def create_figure3_plots():
    """Create plots matching Figure 3 from the paper"""
    with open('training_efficiency_results.json', 'r') as f:
        results = json.load(f)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Figure 3(a): Training Time vs Batch Size
    fused = results['paper_reproduction']['fused_bitlinear']
    vanilla = results['paper_reproduction']['vanilla_bitlinear']
    
    ax1.plot(fused['batch_sizes'], fused['times'], 'o-', label='Fused BitLinear', linewidth=2, markersize=8)
    ax1.plot(vanilla['batch_sizes'], vanilla['times'], 's-', label='Vanilla BitLinear', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Training Time (s/iteration)')
    ax1.set_title('Figure 3(a): Computational Latency vs. Batch Size\n(Your Results)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 3(b): Memory Usage vs Batch Size
    ax2.plot(fused['batch_sizes'], fused['memory'], 'o-', label='Fused BitLinear', linewidth=2, markersize=8)
    ax2.plot(vanilla['batch_sizes'], vanilla['memory'], 's-', label='Vanilla BitLinear', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Peak Memory (GB)')
    ax2.set_title('Figure 3(b): Memory Utilization vs. Batch Size\n(Your Results)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pythia Comparison - 370M
    pythia = results['pythia_comparison']
    mm_370m = pythia['matmul_free_370m']
    pythia_410m = pythia['pythia_410m']
    
    # Find common batch sizes
    common_batches_370 = set(mm_370m['batch_sizes']).intersection(set(pythia_410m['batch_sizes']))
    common_batches_370 = sorted(list(common_batches_370))
    
    mm_times_370 = [mm_370m['times'][mm_370m['batch_sizes'].index(b)] for b in common_batches_370]
    pythia_times_410 = [pythia_410m['times'][pythia_410m['batch_sizes'].index(b)] for b in common_batches_370]
    
    ax3.plot(common_batches_370, mm_times_370, 'o-', label='MatMul-free 370M', linewidth=2, markersize=8)
    ax3.plot(common_batches_370, pythia_times_410, 's-', label='Pythia-410M', linewidth=2, markersize=8)
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Training Time (s/iteration)')
    ax3.set_title('370M MatMul-free vs Pythia-410M')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Pythia Comparison - 1.3B
    mm_1_3b = pythia['matmul_free_1_3b']
    pythia_1_4b = pythia['pythia_1_4b']
    
    common_batches_1_3 = set(mm_1_3b['batch_sizes']).intersection(set(pythia_1_4b['batch_sizes']))
    common_batches_1_3 = sorted(list(common_batches_1_3))
    
    mm_times_1_3 = [mm_1_3b['times'][mm_1_3b['batch_sizes'].index(b)] for b in common_batches_1_3]
    pythia_times_1_4 = [pythia_1_4b['times'][pythia_1_4b['batch_sizes'].index(b)] for b in common_batches_1_3]
    
    ax4.plot(common_batches_1_3, mm_times_1_3, 'o-', label='MatMul-free 1.3B', linewidth=2, markersize=8)
    ax4.plot(common_batches_1_3, pythia_times_1_4, 's-', label='Pythia-1.4B', linewidth=2, markersize=8)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Training Time (s/iteration)')
    ax4.set_title('1.3B MatMul-free vs Pythia-1.4B')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figure3_reproduction_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Plots saved to figure3_reproduction_analysis.png")
    plt.show()

def main():
    print("🔬 FIGURE 3 REPRODUCTION ANALYSIS")
    print("=" * 80)
    
    # Analyze current results
    analyze_current_results()
    
    # Test higher batch sizes
    test_higher_batch_sizes()
    
    # Create plots
    create_figure3_plots()
    
    print(f"\n🎯 SUMMARY:")
    print(f"   1. Your fused implementation works but shows smaller improvements than paper")
    print(f"   2. Need to test batch size 28 to match paper exactly")
    print(f"   3. MatMul-free being slower than Pythia is expected - paper focuses on memory efficiency")
    print(f"   4. Consider using gradient checkpointing or model parallelism for larger batch sizes")

if __name__ == "__main__":
    main()
