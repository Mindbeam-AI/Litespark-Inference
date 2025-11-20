# Cell: Export All Results to PDF
import json
import os
from datetime import datetime
import textwrap

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf_backend
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not available. Installing...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf_backend
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True

print("="*70)
print("EXPORTING RESULTS TO PDF")
print("="*70)

# --- STYLING AND CONFIGURATION ---
STYLE = {
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 16,
}

COLORS = {
    'primary': '#005f73',
    'secondary': '#0a9396',
    'accent': '#ee9b00',
    'neutral': '#7A7A7A',
    'bg_light': '#F0F0F0',
    'bg_highlight': '#E8F4F8',
    'danger': '#ae2012',
    'text_light': '#FFFFFF',
    'text_dark': '#333333',
}

plt.rcParams.update(STYLE)

# --- HELPER FUNCTIONS ---
def create_styled_table(ax, data, col_widths, loc='center', bbox=None):
    """Creates a styled table in the given axes."""
    table = ax.table(cellText=data, cellLoc='left', loc=loc, colWidths=col_widths, bbox=bbox)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('#FFFFFF')
        cell.set_linewidth(0)
        if i == 0:
            cell.set_text_props(weight='bold', color=COLORS['text_light'])
            cell.set_facecolor(COLORS['primary'])
        else:
            cell.set_facecolor(COLORS['bg_light'] if i % 2 != 0 else '#FFFFFF')
        if j == 0:
            cell.set_text_props(weight='bold')
    return table

def add_bar_labels(ax, bars, is_int=False):
    """Adds labels to the top of bar charts."""
    for bar in bars:
        height = bar.get_height()
        label = f'{int(height)}' if is_int else f'{height:.2f}'
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

# --- DATA LOADING ---
try:
    with open('comparison_results.json', 'r') as f:
        results = json.load(f)
except FileNotFoundError:
    print("ERROR: comparison_results.json not found. Please run the comparison cell first.")
    exit()

# --- PDF CREATION ---
pdf_filename = f"training_results_step9000_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
pdf = pdf_backend.PdfPages(pdf_filename)

# ======================================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ======================================================================================
fig = plt.figure(figsize=(8.27, 11.69)) # A4 Size
fig.suptitle('MatMul-Free Model: Performance Analysis', fontweight='bold', color=COLORS['primary'])
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08, hspace=0.5, wspace=0.4)

gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 1, 1])

# --- Training Info ---
ax_info = fig.add_subplot(gs[0, 0])
ax_info.axis('off')
ax_info.set_title('Training Snapshot', fontweight='bold', loc='left')
training_info = [
    ["Checkpoint", "Step 9,000 / 10,000"],
    ["Model Size", "~257M parameters"],
    ["Dataset", "SlimPajama-6B"],
    ["Hardware", "4x NVIDIA A10G"],
]
table = create_styled_table(ax_info, training_info, [0.4, 0.6], bbox=[0, 0, 1, 0.9])

# --- Key Findings ---
ax_findings = fig.add_subplot(gs[0, 1])
ax_findings.axis('off')
ax_findings.set_title('Key Performance Metrics', fontweight='bold', loc='left')
speedup = (results['matmul_free']['throughput'] / results['baseline']['throughput'] - 1) * 100
mem_diff_2048 = (results['matmul_free']['mem_2048'] / results['baseline']['mem_2048'] - 1) * 100
findings_text = [
    (f"{speedup:+.1f}%", "Throughput vs. Baseline", "Higher is better"),
    (f"{mem_diff_2048:+.1f}%", "Memory Overhead (seq=2048)", "Lower is better"),
]
for i, (value, metric, subtext) in enumerate(findings_text):
    ax_findings.text(0, 0.7 - i*0.4, value, fontsize=18, fontweight='bold', color=COLORS['accent'])
    ax_findings.text(0, 0.6 - i*0.4, metric, fontsize=10, fontweight='bold')
    ax_findings.text(0, 0.52 - i*0.4, subtext, fontsize=8, color=COLORS['neutral'])

# --- Memory Comparison ---
ax_mem = fig.add_subplot(gs[1, :])
seq_lengths = [512, 1024, 2048]
matmul_free_mem = [results['matmul_free'][f'mem_{s}'] for s in seq_lengths]
baseline_mem = [results['baseline'][f'mem_{s}'] for s in seq_lengths]
x = range(len(seq_lengths))
width = 0.35
bars1 = ax_mem.bar([i - width/2 for i in x], matmul_free_mem, width, label='MatMul-Free', color=COLORS['secondary'], alpha=0.9)
bars2 = ax_mem.bar([i + width/2 for i in x], baseline_mem, width, label='Baseline (Pythia-410M)', color=COLORS['primary'], alpha=0.9)
ax_mem.set_ylabel('Peak Memory (GB)')
ax_mem.set_title('Memory Usage Comparison', fontweight='bold')
ax_mem.set_xticks(x)
ax_mem.set_xticklabels([str(s) for s in seq_lengths])
ax_mem.legend(loc='best')
ax_mem.grid(True, axis='y', linestyle='--', alpha=0.6)
add_bar_labels(ax_mem, bars1)
add_bar_labels(ax_mem, bars2)

# --- Throughput Comparison ---
ax_tp = fig.add_subplot(gs[2, :])
matmul_free_tp = results['matmul_free']['throughput']
baseline_tp = results['baseline']['throughput']
bars = ax_tp.bar(['MatMul-Free', 'Baseline'], [matmul_free_tp, baseline_tp], color=[COLORS['secondary'], COLORS['primary']], alpha=0.9, width=0.5)
ax_tp.set_ylabel('Tokens/Second')
ax_tp.set_title('Throughput Comparison (seq=1024, batch=4)', fontweight='bold')
ax_tp.grid(True, axis='y', linestyle='--', alpha=0.6)
add_bar_labels(ax_tp, bars, is_int=True)

pdf.savefig(fig, bbox_inches='tight')
plt.close()

# ======================================================================================
# PAGE 2: TRAINING AND MEMORY DEEP DIVE
# ======================================================================================
fig2 = plt.figure(figsize=(8.27, 11.69)) # A4 Size
fig2.suptitle('Training Trajectory & Memory Analysis', fontweight='bold', color=COLORS['primary'])
fig2.subplots_adjust(left=0.12, right=0.9, top=0.9, bottom=0.08, hspace=0.6, wspace=0.4)

gs2 = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1, 1])

# --- Training Loss ---
ax_loss = fig2.add_subplot(gs2[0, 0])
loss_data = [
    (0, 10.9177), (100, 7.9209), (200, 8.2699), (300, 7.7851), (400, 7.6438),
    (500, 8.3004), (600, 8.0159), (700, 7.7430), (1000, 7.3903), (1200, 7.7000),
    (1600, 10.1087), (1700, 8.0061), (1900, 7.7128), (9000, 5.5512), (9100, 7.0496),
    (9300, 7.0958), (9400, 6.2050), (9500, 6.0818), (9600, 6.3308), (9700, 7.5283),
    (9800, 4.4379), (9900, 6.6196),
]
steps, losses = zip(*loss_data)
ax_loss.plot(steps, losses, 'o-', color=COLORS['primary'], markersize=4, linewidth=1.5, label='Training Loss')
ax_loss.axvline(x=9000, color=COLORS['accent'], linestyle='--', linewidth=1.5, label='Checkpoint (Step 9000)')
ax_loss.set_xlabel('Training Step')
ax_loss.set_ylabel('Cross-Entropy Loss')
ax_loss.set_title('Training Loss Trajectory', fontweight='bold')
ax_loss.grid(True, linestyle='--', alpha=0.6)
ax_loss.legend()
ax_loss.annotate('Anomalous Spike', xy=(1600, 10.1087), xytext=(2500, 10.5),
                 arrowprops=dict(facecolor=COLORS['danger'], shrink=0.05, width=1.5, headwidth=6),
                 fontsize=8, color=COLORS['danger'], fontweight='bold')

# --- Memory Analysis ---
gs_mem = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs2[1], wspace=0.4)
ax_mem_scale = fig2.add_subplot(gs_mem[0])
ax_mem_scale.plot(seq_lengths, matmul_free_mem, 'o-', color=COLORS['secondary'], markersize=5, label='MatMul-Free')
ax_mem_scale.plot(seq_lengths, baseline_mem, 's-', color=COLORS['primary'], markersize=5, label='Baseline')
ax_mem_scale.set_xlabel('Sequence Length')
ax_mem_scale.set_ylabel('Peak Memory (GB)')
ax_mem_scale.set_title('Memory Scaling', fontweight='bold')
ax_mem_scale.grid(True, linestyle='--', alpha=0.6)
ax_mem_scale.legend()

ax_mem_overhead = fig2.add_subplot(gs_mem[1])
mem_diff = [(m/b - 1) * 100 for m, b in zip(matmul_free_mem, baseline_mem)]
bars = ax_mem_overhead.bar([str(s) for s in seq_lengths], mem_diff, color=COLORS['accent'], width=0.6)
ax_mem_overhead.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=1)
ax_mem_overhead.set_xlabel('Sequence Length')
ax_mem_overhead.set_ylabel('Memory Overhead (%)')
ax_mem_overhead.set_title('Overhead vs. Baseline', fontweight='bold')
ax_mem_overhead.grid(True, axis='y', linestyle='--', alpha=0.6)
for bar in bars:
    height = bar.get_height()
    ax_mem_overhead.annotate(f'{height:+.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 2 if height > 0 else -10),
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=7, fontweight='bold')

# --- Implementation Details ---
ax_details = fig2.add_subplot(gs2[2, 0])
ax_details.axis('off')
ax_details.set_title('Implementation Notes', fontweight='bold', loc='left')
details_text = """
- **Quantization:** Weights are 1.58-bit (Ternary: -1, 0, 1), Activations are 8-bit.
- **Attention:** Standard Softmax attention is replaced with a Hierarchical Gated Recurrent Network (HGRN).
- **Fused Kernels:** Triton is used to fuse LayerNorm, Linear, and Quantization operations, reducing overhead.
- **Triton Autotuning:** Disabled in this fork due to compatibility issues, using fixed default kernel configurations.
"""
wrapped_text = '\n'.join(textwrap.wrap(details_text, width=100))
ax_details.text(0, 0.4, wrapped_text, transform=ax_details.transAxes, va='center', ha='left', fontsize=9)


pdf.savefig(fig2, bbox_inches='tight')
plt.close()

# --- PDF FINALIZATION ---
pdf.close()

print(f"\n✓ PDF exported successfully: {pdf_filename}")
print(f"✓ Total pages: 2")
print(f"\nContents:")
print(f"  - Page 1: Executive Summary")
print(f"  - Page 2: Training & Memory Deep Dive")
print("\nLayout adjusted to prevent elements from going out of bounds.")
print("="*70)
