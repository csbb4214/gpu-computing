#!/usr/bin/env python3
"""
Plot GPU matrix multiplication benchmark results with clear legends.

Input CSV files (same directory):
- matrix_mul_results_amd.csv
- matrix_mul_results_nvidia.csv
- matrix_mul_results_amd_opt.csv
- matrix_mul_results_nvidia_opt.csv

CSV format:
precision,N,time_ms

Generates:
- Boxplots for Original vs Optimized (with platform & precision)
- Improvement factors
- GFLOPS vs HW Peak (separate plots for float/double & AMD/NVIDIA)
"""

import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

SHOW_POINTS = True

# Input files
FILES_INFO = [
    ("AMD", "matrix_mul_results_amd.csv", "Original"),
    ("NVIDIA", "matrix_mul_results_nvidia.csv", "Original"),
    ("AMD", "matrix_mul_results_amd_opt.csv", "Optimized"),
    ("NVIDIA", "matrix_mul_results_nvidia_opt.csv", "Optimized"),
]

# HW peak GFLOPS (float / double)
HW_PEAK_GFLOPS = {
    "AMD": {"float": 14850, "double": 464},     # GFLOPS
    "NVIDIA": {"float": 7460, "double": 233},
}

# ---------- Data loading ----------

def load_rows(path, platform_label, version):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "platform": platform_label,
                "version": version,
                "time_ms": float(row["time_ms"]),
                "N": int(row["N"]),
                "precision": row.get("precision", "float").lower()
            })
    return rows

def all_Ns(rows):
    return sorted({r["N"] for r in rows})

# ---------- Aggregations ----------

def collect_buckets(rows):
    """
    buckets[(platform, version, N, precision)] = [time_ms, ...]
    """
    buckets = defaultdict(list)
    for r in rows:
        key = (r["platform"], r["version"], r["N"], r["precision"])
        buckets[key].append(r["time_ms"])
    return buckets

def collect_improvements(rows):
    """
    improvements[(platform, N, precision)] = [Original/Optimized,...]
    """
    orig_map = {}
    opt_map = {}
    for r in rows:
        key = (r["platform"], r["N"], r.get("run",0), r["precision"])
        if r["version"] == "Original":
            orig_map[key] = r["time_ms"]
        elif r["version"] == "Optimized":
            opt_map[key] = r["time_ms"]

    improvements = defaultdict(list)
    for key, orig_ms in orig_map.items():
        if key in opt_map and opt_map[key] > 0:
            platform, N, _run, prec = key
            improvements[(platform, N, prec)].append(orig_ms / opt_map[key])
    return improvements

# ---------- Plot helpers ----------

def _apply_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

def _add_jitter_points(ax, ys_list, positions, jitter=0.06, alpha=0.35, markersize=3):
    import random
    rnd = random.Random(42)
    for vals, pos in zip(ys_list, positions):
        for v in vals:
            x = pos + (rnd.random() * 2 - 1) * jitter
            ax.plot(x, v, marker="o", linestyle="None", alpha=alpha, markersize=markersize)

# ---------- Plot 1: Boxplots Original vs Optimized ----------
def plot_matrix_mul_simple(buckets, Ns):
    """
    Original-style boxplots: Log scale, AMD & NVIDIA separate, Original vs Optimized,
    mit fein gestrichelten Linien für log-scale Abschnitte (1,2,3,...)
    """
    import matplotlib.ticker as mticker

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14,6), sharey=True)
    colors = {"Original":"#3498db", "Optimized":"#e74c3c"}

    platforms = ["AMD","NVIDIA"]
    for ax, platform in zip(axes, platforms):
        positions, data, box_colors = [], [], []
        for i, N in enumerate(Ns):
            base = i*1.5
            for j, version in enumerate(["Original","Optimized"]):
                # Alle Präzisionen zusammenfassen
                vals = []
                for prec in ["float","double"]:
                    vals.extend(buckets.get((platform, version, N, prec), []))
                if not vals:
                    continue
                pos = base + j*0.4
                positions.append(pos)
                data.append(vals)
                box_colors.append(colors[version])

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.35,
                            showfliers=True, showmeans=True, meanline=True, patch_artist=True)
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            if SHOW_POINTS:
                _add_jitter_points(ax, data, positions, jitter=0.05)

        # X-axis ticks
        tick_positions = [i*1.5 + 0.2 for i in range(len(Ns))]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("Matrix size N")
        ax.set_title(f"{platform}")
        ax.set_yscale("log")
        ax.grid(True, which='major', alpha=0.3)  # normale Gridlinien
        # Feine gestrichelte Linien für Zwischenwerte
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(1.0,10.0)*0.1, numticks=100))
        ax.grid(which='minor', linestyle=':', alpha=0.3)

    axes[0].set_ylabel("Time (ms)")
    # Legend
    legend_elements = [Patch(facecolor=colors[v], alpha=0.6, label=v) for v in ["Original","Optimized"]]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Matrix Multiplication Benchmark: Original vs Optimized", y=1.08, fontsize=14, fontweight='bold')
    fig.tight_layout()
    out_file = os.path.join(PLOTS_DIR,"boxplots_matrix_mul_simple.png")
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"[DONE] Wrote {out_file}")

# ---------- Plot 3: GFLOPS vs HW Peak (separate plots) ----------

def plot_gflops_vs_hw_separate(buckets, Ns, platforms, hw_peak_gflops):
    _apply_style()
    colors = {"Original":"#3498db", "Optimized":"#e74c3c"}
    for prec in ["float","double"]:
        for platform in platforms:
            fig, ax = plt.subplots(figsize=(10,5))
            positions, data, box_colors, labels = [], [], [], []

            for i, N in enumerate(Ns):
                base = i*2.0
                for j, version in enumerate(["Original","Optimized"]):
                    times = buckets.get((platform, version, N, prec), [])
                    if not times: continue
                    gflops_vals = [2*N**3/(t*1e6) for t in times]  # GFLOPS
                    pos = base + j*0.5
                    positions.append(pos)
                    data.append(gflops_vals)
                    box_colors.append(colors[version])
                    labels.append(f"{version}")

            if data:
                bp = ax.boxplot(data, positions=positions, widths=0.45,
                                patch_artist=True, showfliers=True,
                                showmeans=True, meanline=True)
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                if SHOW_POINTS:
                    _add_jitter_points(ax, data, positions, jitter=0.05)

            tick_positions = [i*2.0+0.25 for i in range(len(Ns))]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(n) for n in Ns])
            ax.set_xlabel("Matrix size N")
            ax.set_ylabel("GFLOPS")
            ax.set_title(f"{platform} {prec.capitalize()} GFLOPS vs HW Peak")
            ax.axhline(y=hw_peak_gflops[platform][prec], color='red', linestyle='--', label="HW Peak")
            # Legend: Version + HW Peak
            seen = set()
            legend_elements = [Patch(facecolor=colors[v], alpha=0.6, label=v) for v in ["Original","Optimized"]]
            legend_elements.append(Patch(facecolor='red', alpha=0.6, label='HW Peak'))
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            out_file = os.path.join(PLOTS_DIR,f"gflops_{platform}_{prec}.png")
            fig.savefig(out_file, dpi=200)
            print(f"[DONE] Wrote {out_file}")

# ---------- Main ----------

def main():
    rows = []
    platforms_set = set()
    for platform, fname, version in FILES_INFO:
        if not os.path.exists(fname):
            print(f"[ERROR] Missing file: {fname}")
            continue
        rows.extend(load_rows(fname, platform, version))
        platforms_set.add(platform)

    if not rows:
        print("[ERROR] No data found")
        return 1

    Ns = all_Ns(rows)
    platforms = sorted(platforms_set)

    buckets = collect_buckets(rows)
    improvements = collect_improvements(rows)

    # Generate plots
    plot_matrix_mul_simple(buckets, Ns)
    plot_improvement(improvements, Ns, platforms)
    plot_gflops_vs_hw_separate(buckets, Ns, platforms, HW_PEAK_GFLOPS)

    print("\n[DONE] All plots generated successfully!")

if __name__ == "__main__":
    raise SystemExit(main())

