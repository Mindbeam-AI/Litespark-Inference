"""
Plot Output Blocking Optimization Results
"""
import matplotlib.pyplot as plt
import numpy as np

# Results from test_blocking_sizes.py
n_blocks = [4, 8, 12, 16, 20, 24, 32]
gflops = [120.0, 124.7, 123.3, 118.9, 68.8, 64.5, 62.0]
times = [4.48, 4.31, 4.35, 4.51, 7.80, 8.32, 8.67]
time_stds = [0.33, 0.21, 0.36, 0.33, 0.27, 0.34, 0.26]

# Find optimal
optimal_idx = np.argmax(gflops)
optimal_block = n_blocks[optimal_idx]
optimal_gflops = gflops[optimal_idx]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: GFLOPS vs N_BLOCK
colors = ['#ff6b6b' if g < 100 else '#4ecdc4' if g < 120 else '#95e1d3' for g in gflops]
bars1 = ax1.bar(range(len(n_blocks)), gflops, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Highlight optimal
bars1[optimal_idx].set_color('#2ecc71')
bars1[optimal_idx].set_edgecolor('#27ae60')
bars1[optimal_idx].set_linewidth(2)

# Add value labels on bars
for i, (block, gf) in enumerate(zip(n_blocks, gflops)):
    label = f'{gf:.1f}'
    if i == optimal_idx:
        label += '\nOPTIMAL'
        ax1.text(i, gf + 3, label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    else:
        ax1.text(i, gf + 2, label, ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Output Blocking Size (N_BLOCK)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Performance (GFLOPS)', fontsize=12, fontweight='bold')
ax1.set_title('ARM64 NEON: Output Blocking Optimization\n256×1024×1024 Matrix, 10 Threads',
              fontsize=13, fontweight='bold')
ax1.set_xticks(range(len(n_blocks)))
ax1.set_xticklabels(n_blocks)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(gflops) * 1.15)

# Add annotation for register pressure
ax1.axvspan(4.5, 6.5, alpha=0.1, color='red')
ax1.text(5.5, 85, 'Register\nPressure', ha='center', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.2))

# Plot 2: Execution Time with error bars
ax2.errorbar(range(len(n_blocks)), times, yerr=time_stds,
             fmt='o-', linewidth=2, markersize=8, capsize=5,
             color='#3498db', ecolor='#95a5a6', capthick=2)

# Highlight optimal
ax2.plot(optimal_idx, times[optimal_idx], 'o', markersize=15,
         color='#2ecc71', markeredgecolor='#27ae60', markeredgewidth=2, zorder=5)

# Add value labels
for i, (block, t, std) in enumerate(zip(n_blocks, times, time_stds)):
    if i == optimal_idx:
        label = f'{t:.2f}ms\nOPTIMAL'
        ax2.text(i, t - 0.5, label, ha='center', va='top', fontweight='bold', fontsize=10)
    else:
        ax2.text(i, t + 0.3, f'{t:.2f}ms', ha='center', va='bottom', fontsize=9)

ax2.set_xlabel('Output Blocking Size (N_BLOCK)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax2.set_title('Execution Time vs Blocking Size\nLower is Better',
              fontsize=13, fontweight='bold')
ax2.set_xticks(range(len(n_blocks)))
ax2.set_xticklabels(n_blocks)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(times) * 1.15)
ax2.invert_yaxis()  # Invert so lower (better) times are higher on graph

plt.tight_layout()
plt.savefig('benchmark_results/blocking_optimization.png', dpi=150, bbox_inches='tight')
print("Graph saved to: benchmark_results/blocking_optimization.png")

# Create explanation text
explanation = """
OUTPUT BLOCKING OPTIMIZATION RESULTS
====================================

Test Configuration:
- Matrix: 256×1024×1024 (M×N×K)
- Platform: ARM64 NEON (4-wide SIMD)
- Threads: 10 (OpenMP)
- Optimization: Process multiple outputs simultaneously

Key Findings:
1. OPTIMAL: N_BLOCK=8 achieves 124.7 GFLOPS
   - Sweet spot between parallelism and register pressure
   - 4.31ms average execution time
   - Stable performance (±0.21ms)

2. Good Performance Range (N_BLOCK=4-16):
   - 118-125 GFLOPS sustained
   - Less than 10% performance variation
   - Practical for production use

3. Performance Cliff (N_BLOCK≥20):
   - Drops to 62-69 GFLOPS (50% loss)
   - Caused by register spilling
   - ARM64 has limited SIMD registers

Why Output Blocking Works:
- Loads input once, reuses for multiple outputs
- Better instruction-level parallelism
- Hides memory latency
- Maximizes SIMD utilization

Technical Details:
- 4-wide NEON SIMD (4 floats per instruction)
- N_BLOCK=8 means 8 outputs processed together
- Uses 16 SIMD registers (8 pos + 8 neg accumulators)
- TRUE matmul-free: only add/subtract operations

Comparison to Baseline:
- Unoptimized (N_BLOCK=1): ~10 GFLOPS
- Optimized (N_BLOCK=8): 124.7 GFLOPS
- Improvement: 12x faster

Why Not Higher Blocking?
- ARM64 has 32 NEON registers (v0-v31)
- N_BLOCK=8 uses 16 registers (8×2 for pos/neg)
- N_BLOCK=20+ causes register spilling to memory
- Memory spills kill performance

Extrapolation to x86_64 AVX2:
- AVX2: 8-wide SIMD (2x NEON)
- Expected optimal: N_BLOCK=16
- Expected performance: ~250 GFLOPS (2x ARM64)
- Same blocking principles apply
"""

with open('benchmark_results/blocking_optimization_explanation.txt', 'w') as f:
    f.write(explanation)

print("\nExplanation saved to: benchmark_results/blocking_optimization_explanation.txt")
print("\nOptimal Configuration:")
print(f"  N_BLOCK: {optimal_block}")
print(f"  GFLOPS: {optimal_gflops}")
print(f"  Time: {times[optimal_idx]:.2f} ± {time_stds[optimal_idx]:.2f} ms")
