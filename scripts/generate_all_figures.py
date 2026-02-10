#!/usr/bin/env python3
"""Generate all figures for the Litespark-Inf paper.

Outputs:
Main text figures:
- apple_silicon_summary.png
- avx512_vnni_summary.png
- avx_vnni_summary.png
- cross_platform_comparison.png

Appendix figures:
- performance_comparison_amd_epyc_9r14_user.png
- performance_comparison_intel_xeon_8488c_user.png
- performance_comparison_apple_m4_user.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "final"
FINAL_DIR.mkdir(exist_ok=True)

# Colors matching paper style
COLOR_PYTORCH = "#E74C3C"
COLOR_OPT = "#37C36B"
COLOR_ALT = "#3F93CC"
COLOR_MS_ORIGINAL = "#E74C3C"
COLOR_MS_V2 = "#3498DB"
COLOR_LITESPARK = "#2ECC71"


def setup_style() -> None:
    """Configure plotting style to match paper (Times New Roman font)."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": "#f2f2f2",
            "axes.facecolor": "#f2f2f2",
            "axes.edgecolor": "#bcbcbc",
            "axes.titleweight": "bold",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.titlepad": 8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "grid.color": "#b5b5b5",
            "grid.alpha": 0.7,
            "grid.linewidth": 1.0,
            "legend.fontsize": 10,
        }
    )


def _annotate_bars(ax, bars, values, fmt: str, extra: float = 0.0) -> None:
    ymax = max(values) if values else 1.0
    offset = max(ymax * 0.015, extra)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def _save_to_targets(fig, filename: str) -> None:
    root_out = ROOT / filename
    final_out = FINAL_DIR / filename
    fig.savefig(root_out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(final_out, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {root_out}")
    print(f"Saved: {final_out}")


# =============================================================================
# MAIN TEXT FIGURES
# =============================================================================

def plot_platform_summary(
    title: str,
    labels: list[str],
    memory: list[float],
    ttft: list[float],
    throughput: list[float],
    colors: list[str],
    filename: str,
) -> None:
    """Generate summary bar charts for a platform (Memory, TTFT, Throughput)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("#f2f2f2")

    charts = [
        ("Memory Usage", "MB", memory, "{:.0f}"),
        ("Time to First Token", "ms", ttft, "{:.1f}"),
        ("Throughput", "tokens/sec", throughput, "{:.1f}"),
    ]

    for ax, (chart_title, ylabel, values, label_fmt) in zip(axes, charts):
        bars = ax.bar(
            labels,
            values,
            color=colors,
            edgecolor="black",
            linewidth=1.7,
        )
        ax.set_title(chart_title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="both", alpha=0.7)

        ymax = max(values) if values else 1.0
        ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
        _annotate_bars(ax, bars, values, label_fmt)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save_to_targets(fig, filename)
    plt.close(fig)


def plot_cross_platform() -> None:
    """Generate cross-platform comparison figure."""
    platforms = ["Apple M4\n(NEON)", "Ice Lake\n(AVX-512)", "Core Ultra\n(AVX-VNNI)"]
    colors = ["#2ca02c", "#1f77b4", "#9467bd"]

    # Speedup values from benchmark results
    ttft_speedup = [9.2, 12.6, 8.3]
    throughput_speedup = [52.3, 26.7, 21.3]
    memory_eff = [14.0, 14.0, 13.9]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.patch.set_facecolor("#f2f2f2")

    metrics = [
        (ttft_speedup, "TTFT Speedup", "Speedup vs PyTorch"),
        (throughput_speedup, "Throughput Speedup", "Speedup vs PyTorch"),
        (memory_eff, "Memory Efficiency", "Reduction vs PyTorch"),
    ]

    for ax, (values, title, ylabel) in zip(axes, metrics):
        bars = ax.bar(platforms, values, color=colors, edgecolor="black", linewidth=1.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(True, axis="both", alpha=0.7)

        ymax = max(values) if values else 1.0
        ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
        _annotate_bars(ax, bars, values, "{:.1f}x")

    fig.suptitle("Litespark-Inf — Cross-Platform Performance", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save_to_targets(fig, "cross_platform_comparison.png")
    plt.close(fig)


# =============================================================================
# APPENDIX FIGURES (BitNet.cpp v2 comparison)
# =============================================================================

def plot_appendix_comparison_amd() -> None:
    """Generate AMD EPYC 9R14 comparison figure for appendix."""
    threads = [1, 2, 4, 8]

    # Data from tables in paper
    original_pp128 = [35.0, 70.0, 140.0, 210.0]
    v2_pp128 = [43.4, 81.2, 156.8, 291.8]
    litespark_pp128 = [38.2, 74.7, 140.7, 230.7]

    original_tg128 = [10.0, 18.0, 30.0, 42.0]
    v2_tg128 = [15.6, 28.7, 49.2, 66.2]
    litespark_tg128 = [15.9, 28.1, 48.2, 67.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # pp128
    ax1.plot(threads, original_pp128, 'o-', color=COLOR_MS_ORIGINAL, linewidth=2.5,
             markersize=10, label='BitNet.cpp Original', markeredgecolor='black', markeredgewidth=1)
    ax1.plot(threads, v2_pp128, 's-', color=COLOR_MS_V2, linewidth=2.5,
             markersize=10, label='BitNet.cpp V2', markeredgecolor='black', markeredgewidth=1)
    ax1.plot(threads, litespark_pp128, '^-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax1.set_xlabel('Number of Threads', fontsize=13)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax1.set_title('Prefill (pp128)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(threads)
    ax1.set_ylim(0, max(v2_pp128) * 1.15)

    # tg128
    ax2.plot(threads, original_tg128, 'o-', color=COLOR_MS_ORIGINAL, linewidth=2.5,
             markersize=10, label='BitNet.cpp Original', markeredgecolor='black', markeredgewidth=1)
    ax2.plot(threads, v2_tg128, 's-', color=COLOR_MS_V2, linewidth=2.5,
             markersize=10, label='BitNet.cpp V2', markeredgecolor='black', markeredgewidth=1)
    ax2.plot(threads, litespark_tg128, '^-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Number of Threads', fontsize=13)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax2.set_title('Token Generation (tg128)', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.4)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, max(litespark_tg128) * 1.15)

    fig.suptitle('Performance Comparison: AMD EPYC 9R14', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_to_targets(fig, "performance_comparison_amd_epyc_9r14_user.png")
    plt.close(fig)


def plot_appendix_comparison_intel() -> None:
    """Generate Intel Xeon 8488C comparison figure for appendix."""
    threads = [1, 2, 4, 6]

    # Data from tables in paper
    original_pp128 = [27.0, 40.0, 55.0, 79.0]
    v2_pp128 = [43.4, 65.8, 77.9, 101.3]
    litespark_pp128 = [59.7, 85.9, 110.2, 120.7]

    original_tg128 = [10.0, 13.0, 16.0, 20.0]
    v2_tg128 = [13.3, 19.1, 24.3, 29.5]
    litespark_tg128 = [13.6, 19.5, 25.0, 28.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # pp128
    ax1.plot(threads, original_pp128, 'o-', color=COLOR_MS_ORIGINAL, linewidth=2.5,
             markersize=10, label='BitNet.cpp Original', markeredgecolor='black', markeredgewidth=1)
    ax1.plot(threads, v2_pp128, 's-', color=COLOR_MS_V2, linewidth=2.5,
             markersize=10, label='BitNet.cpp V2', markeredgecolor='black', markeredgewidth=1)
    ax1.plot(threads, litespark_pp128, '^-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax1.set_xlabel('Number of Threads', fontsize=13)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax1.set_title('Prefill (pp128)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(threads)
    ax1.set_ylim(0, max(litespark_pp128) * 1.15)

    # tg128
    ax2.plot(threads, original_tg128, 'o-', color=COLOR_MS_ORIGINAL, linewidth=2.5,
             markersize=10, label='BitNet.cpp Original', markeredgecolor='black', markeredgewidth=1)
    ax2.plot(threads, v2_tg128, 's-', color=COLOR_MS_V2, linewidth=2.5,
             markersize=10, label='BitNet.cpp V2', markeredgecolor='black', markeredgewidth=1)
    ax2.plot(threads, litespark_tg128, '^-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Number of Threads', fontsize=13)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax2.set_title('Token Generation (tg128)', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.4)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, max(v2_tg128) * 1.15)

    fig.suptitle('Performance Comparison: Intel Xeon Platinum 8488C', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_to_targets(fig, "performance_comparison_intel_xeon_8488c_user.png")
    plt.close(fig)


def plot_appendix_apple_m4() -> None:
    """Generate Apple M4 scaling figure for appendix."""
    threads = [1, 2, 4, 8, 10]
    pp128 = [26.1, 43.1, 81.9, 101.2, 108.8]
    tg128 = [6.5, 11.0, 15.4, 14.0, 19.6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    # pp128
    ax1.plot(threads, pp128, 's-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax1.set_xlabel('Number of Threads', fontsize=13)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax1.set_title('Prefill (pp128)', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.4)
    ax1.set_xticks(threads)
    ax1.set_ylim(0, max(pp128) * 1.15)

    # Add value annotations
    for x, y in zip(threads, pp128):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

    # tg128
    ax2.plot(threads, tg128, 's-', color=COLOR_LITESPARK, linewidth=2.5,
             markersize=10, label='Litespark-Inf', markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Number of Threads', fontsize=13)
    ax2.set_ylabel('Throughput (tokens/s)', fontsize=13)
    ax2.set_title('Token Generation (tg128)', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.4)
    ax2.set_xticks(threads)
    ax2.set_ylim(0, max(tg128) * 1.15)

    # Add value annotations
    for x, y in zip(threads, tg128):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

    fig.suptitle('Litespark-Inf Scaling: Apple M4', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_to_targets(fig, "performance_comparison_apple_m4_user.png")
    plt.close(fig)


def main() -> None:
    setup_style()

    print("=" * 60)
    print("Generating all figures for Litespark-Inf paper")
    print("=" * 60)

    # Main text figures
    print("\n--- Main Text Figures ---")

    # Apple Silicon summary (PyTorch vs NEON vs Accelerate)
    plot_platform_summary(
        title="Litespark-Inf Performance: Apple Silicon (M4)",
        labels=["PyTorch", "NEON", "Accelerate"],
        memory=[7673, 556, 6949],
        ttft=[2632, 288, 373],
        throughput=[0.39, 20.4, 5.52],
        colors=[COLOR_PYTORCH, COLOR_OPT, COLOR_ALT],
        filename="apple_silicon_summary.png",
    )

    # AVX-512 VNNI summary
    plot_platform_summary(
        title="Litespark-Inf Performance: Intel/AMD (AVX-512 VNNI)",
        labels=["PyTorch", "AVX-512 VNNI"],
        memory=[7800, 556],
        ttft=[2450, 195],
        throughput=[0.42, 11.2],
        colors=[COLOR_PYTORCH, COLOR_OPT],
        filename="avx512_vnni_summary.png",
    )

    # AVX-VNNI summary
    plot_platform_summary(
        title="Litespark-Inf Performance: Intel Core Ultra (AVX-VNNI)",
        labels=["PyTorch", "AVX-VNNI"],
        memory=[7750, 556],
        ttft=[2580, 310],
        throughput=[0.40, 8.5],
        colors=[COLOR_PYTORCH, COLOR_OPT],
        filename="avx_vnni_summary.png",
    )

    # Cross-platform comparison
    plot_cross_platform()

    # Appendix figures
    print("\n--- Appendix Figures ---")
    plot_appendix_comparison_amd()
    plot_appendix_comparison_intel()
    plot_appendix_apple_m4()

    print("\n" + "=" * 60)
    print("Done! All figures generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
