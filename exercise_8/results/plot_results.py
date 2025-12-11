#!/usr/bin/env python3
"""
Compare original vs optimized OpenCL scan benchmarks with grouped boxplots and bar charts.

Expected input files (same directory as this script):
  - scan_benchmark_int_amd.csv
  - scan_benchmark_int_rtx.csv

CSV format:
  run,impl,elapsed_ms,N,type,host

Where impl can be: sequential, opencl, opencl_optimized

Outputs (written into ./plots/):
  - compare_opencl_implementations.png  (opencl vs opencl_optimized)
  - compare_speedup.png                 (speedup for both implementations)
  - compare_improvement.png             (opencl_optimized vs opencl)
  - compare_bar_times.png               (bar chart of all implementations)
  - compare_all_overview.png            (complete overview)

Run:
  python3 plot.py
"""

import csv
import os
from collections import defaultdict
import numpy as np

FILES = [
    ("AMD", "scan_benchmark_int_amd.csv"),
    ("RTX", "scan_benchmark_int_rtx.csv"),
]

PLOTS_DIR = "plots"

OUT_OPENCL_COMPARE = os.path.join(PLOTS_DIR, "compare_opencl_implementations.png")
OUT_SPEEDUP = os.path.join(PLOTS_DIR, "compare_speedup.png")
OUT_IMPROVEMENT = os.path.join(PLOTS_DIR, "compare_improvement.png")
OUT_BAR_TIMES = os.path.join(PLOTS_DIR, "compare_bar_times.png")
OUT_BAR_OPENCL_ONLY = os.path.join(PLOTS_DIR, "compare_bar_opencl_only.png")
OUT_ALL = os.path.join(PLOTS_DIR, "compare_all_overview.png")

SHOW_POINTS = True


# ---------- Data loading ----------

def load_rows(path, platform_label):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            typ = (row.get("type") or "int").strip()
            if typ != "int":
                continue

            rows.append({
                "platform": platform_label,
                "run": int(row["run"]),
                "impl": row["impl"].strip().lower(),
                "elapsed_ms": float(row["elapsed_ms"]),
                "N": int(row["N"]),
                "host": (row.get("host") or platform_label).strip(),
            })
    return rows


def all_Ns(rows):
    return sorted({r["N"] for r in rows})


# ---------- Aggregations ----------

def collect_buckets(rows):
    """
    buckets[(platform, impl, N)] = [elapsed_ms, ...]
    """
    buckets = defaultdict(list)
    for r in rows:
        key = (r["platform"], r["impl"], r["N"])
        buckets[key].append(r["elapsed_ms"])
    return buckets


def collect_speedups(rows):
    """
    speedups[(platform, impl, N)] = [seq/opencl, ...] for matched runs
    """
    seq_map = {}
    cl_map = defaultdict(dict)

    for r in rows:
        key = (r["platform"], r["N"], r["run"])
        if r["impl"] == "sequential":
            seq_map[key] = r["elapsed_ms"]
        elif r["impl"] in ["opencl", "opencl_optimized"]:
            cl_map[r["impl"]][key] = r["elapsed_ms"]

    speedups = defaultdict(list)
    for impl in ["opencl", "opencl_optimized"]:
        for key, seq_ms in seq_map.items():
            if key in cl_map[impl] and cl_map[impl][key] > 0:
                platform, N, _run = key
                speedups[(platform, impl, N)].append(seq_ms / cl_map[impl][key])

    return speedups


def collect_improvements(rows):
    """
    improvements[(platform, N)] = [opencl/opencl_optimized, ...] for matched runs
    """
    orig_map = {}
    opt_map = {}

    for r in rows:
        key = (r["platform"], r["N"], r["run"])
        if r["impl"] == "opencl":
            orig_map[key] = r["elapsed_ms"]
        elif r["impl"] == "opencl_optimized":
            opt_map[key] = r["elapsed_ms"]

    improvements = defaultdict(list)
    for key, orig_ms in orig_map.items():
        if key in opt_map and opt_map[key] > 0:
            platform, N, _run = key
            improvements[(platform, N)].append(orig_ms / opt_map[key])

    return improvements


# ---------- Plot helpers ----------

def _apply_style():
    import matplotlib.pyplot as plt
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


# ---------- Plot 1: OpenCL vs OpenCL Optimized ----------

def plot_opencl_comparison(buckets, Ns, platforms):
    """
    Compare opencl vs opencl_optimized for each platform
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    impls = ["opencl", "opencl_optimized"]
    impl_labels = ["Original", "Optimized"]
    colors = ["#3498db", "#e74c3c"]
    
    for ax, platform in zip(axes, platforms):
        positions = []
        data = []
        box_colors = []
        
        for i, N in enumerate(Ns):
            base = i * 1.5
            for j, impl in enumerate(impls):
                vals = buckets.get((platform, impl, N), [])
                if not vals:
                    continue
                pos = base + j * 0.4
                positions.append(pos)
                data.append(vals)
                box_colors.append(colors[j])
        
        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.35,
                showfliers=True,
                showmeans=True,
                meanline=True,
                patch_artist=True
            )
            
            # Color boxes
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            if SHOW_POINTS:
                _add_jitter_points(ax, data, positions, jitter=0.05)
        
        # X-axis ticks
        tick_positions = []
        tick_labels = []
        for i, N in enumerate(Ns):
            base = i * 1.5
            center = base + 0.2
            tick_positions.append(center)
            tick_labels.append(str(N))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_title(f"{platform}")
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Time (ms)")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label='Original'),
        Patch(facecolor=colors[1], alpha=0.6, label='Optimized')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle("OpenCL Scan: Original vs Optimized", y=1.08, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_OPENCL_COMPARE, dpi=200, bbox_inches='tight')
    print(f"[DONE] Wrote {OUT_OPENCL_COMPARE}")


# ---------- Plot 2: Speedup Comparison ----------

def plot_speedup_comparison(speedups, Ns, platforms):
    """
    Show speedup for both opencl and opencl_optimized
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    impls = ["opencl", "opencl_optimized"]
    impl_labels = ["Original", "Optimized"]
    colors = ["#3498db", "#e74c3c"]
    
    for ax, platform in zip(axes, platforms):
        positions = []
        data = []
        box_colors = []
        
        for i, N in enumerate(Ns):
            base = i * 1.5
            for j, impl in enumerate(impls):
                vals = speedups.get((platform, impl, N), [])
                if not vals:
                    continue
                pos = base + j * 0.4
                positions.append(pos)
                data.append(vals)
                box_colors.append(colors[j])
        
        if data:
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.35,
                showfliers=True,
                showmeans=True,
                meanline=True,
                patch_artist=True
            )
            
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            if SHOW_POINTS:
                _add_jitter_points(ax, data, positions, jitter=0.05)
        
        # X-axis ticks
        tick_positions = []
        tick_labels = []
        for i, N in enumerate(Ns):
            base = i * 1.5
            center = base + 0.2
            tick_positions.append(center)
            tick_labels.append(str(N))
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("N")
        ax.set_title(f"{platform}")
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.3)
    
    axes[0].set_ylabel("Speedup (Sequential / OpenCL)")
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label='Original'),
        Patch(facecolor=colors[1], alpha=0.6, label='Optimized')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02))
    
    fig.suptitle("Speedup: Sequential vs OpenCL (Original & Optimized)", y=1.08, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_SPEEDUP, dpi=200, bbox_inches='tight')
    print(f"[DONE] Wrote {OUT_SPEEDUP}")


# ---------- Plot 3: Improvement Factor ----------

def plot_improvement(improvements, Ns, platforms):
    """
    Show improvement: opencl_time / opencl_optimized_time
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    
    num_platforms = len(platforms)
    group_gap = 1.5
    box_width = 0.4
    colors = ["#2ecc71", "#f39c12"]
    
    positions = []
    data = []
    box_colors = []
    
    for i, N in enumerate(Ns):
        base = i * group_gap
        for j, platform in enumerate(platforms):
            vals = improvements.get((platform, N), [])
            if not vals:
                continue
            pos = base + j * box_width
            positions.append(pos)
            data.append(vals)
            box_colors.append(colors[j])
    
    if data:
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width * 0.9,
            showfliers=True,
            showmeans=True,
            meanline=True,
            patch_artist=True
        )
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        if SHOW_POINTS:
            _add_jitter_points(ax, data, positions)
    
    # X-axis ticks
    tick_positions = []
    tick_labels = []
    for i, N in enumerate(Ns):
        base = i * group_gap
        center = base + (num_platforms - 1) * box_width / 2
        tick_positions.append(center)
        tick_labels.append(str(N))
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("N")
    ax.set_ylabel("Improvement Factor (Original / Optimized)")
    ax.set_title("Optimization Improvement: How Much Faster is Optimized vs Original?")
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, linewidth=1, label='No improvement')
    ax.grid(True, alpha=0.3)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label='AMD'),
        Patch(facecolor=colors[1], alpha=0.6, label='RTX')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout()
    fig.savefig(OUT_IMPROVEMENT, dpi=200)
    print(f"[DONE] Wrote {OUT_IMPROVEMENT}")


# ---------- Plot 4: Bar Chart of All Times ----------

def plot_bar_times(buckets, Ns, platforms):
    """
    Bar chart showing sequential, opencl, and opencl_optimized times
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    impls = ["sequential", "opencl", "opencl_optimized"]
    impl_labels = ["Sequential", "OpenCL Original", "OpenCL Optimized"]
    colors = ["#95a5a6", "#3498db", "#e74c3c"]
    
    for ax, platform in zip(axes, platforms):
        # Calculate means for each implementation
        x = np.arange(len(Ns))
        width = 0.25
        
        for i, impl in enumerate(impls):
            means = []
            stds = []
            for N in Ns:
                vals = buckets.get((platform, impl, N), [])
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(0)
                    stds.append(0)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, means, width, label=impl_labels[i], 
                         color=colors[i], alpha=0.8, yerr=stds, capsize=3)
            
            # Add value labels on top of bars (only for non-zero values)
            for j, (bar, mean) in enumerate(zip(bars, means)):
                if mean > 0:
                    height = bar.get_height()
                    # Format label based on magnitude
                    if mean < 1:
                        label = f'{mean:.3f}'
                    elif mean < 10:
                        label = f'{mean:.2f}'
                    else:
                        label = f'{mean:.1f}'
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           label, ha='center', va='bottom', fontsize=7)
        
        ax.set_xlabel('N')
        ax.set_ylabel('Time (ms)')
        ax.set_title(f'{platform}')
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle("Execution Times: Sequential vs OpenCL (Original & Optimized)", 
                 y=1.02, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_BAR_TIMES, dpi=200, bbox_inches='tight')
    print(f"[DONE] Wrote {OUT_BAR_TIMES}")

def plot_bar_times_opencl_only(buckets, Ns, platforms):
    """
    Bar chart showing ONLY opencl and opencl_optimized times
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    impls = ["opencl", "opencl_optimized"]
    impl_labels = ["OpenCL Original", "OpenCL Optimized"]
    colors = ["#3498db", "#e74c3c"]
    
    for ax, platform in zip(axes, platforms):
        x = np.arange(len(Ns))
        width = 0.28
        
        for i, impl in enumerate(impls):
            means = []
            stds = []
            for N in Ns:
                vals = buckets.get((platform, impl, N), [])
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(0)
                    stds.append(0)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                label=impl_labels[i],
                color=colors[i],
                alpha=0.85,
                yerr=stds,
                capsize=3
            )

            # Bar labels
            for bar, mean in zip(bars, means):
                if mean > 0:
                    height = bar.get_height()
                    if mean < 1:
                        label = f"{mean:.3f}"
                    elif mean < 10:
                        label = f"{mean:.2f}"
                    else:
                        label = f"{mean:.1f}"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=7
                    )

        ax.set_xlabel("N")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{platform}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left")

    fig.suptitle(
        "Execution Times: OpenCL Original vs Optimized (No Sequential)",
        y=1.02,
        fontsize=14,
        fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(OUT_BAR_OPENCL_ONLY, dpi=200, bbox_inches="tight")
    print(f"[DONE] Wrote {OUT_BAR_OPENCL_ONLY}")



# ---------- Plot 5: Complete Overview ----------

def plot_complete_overview(buckets, speedups, improvements, Ns, platforms):
    """
    3-panel overview: Times, Speedups, Improvements
    """
    import matplotlib.pyplot as plt
    
    _apply_style()
    
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    impls = ["opencl", "opencl_optimized"]
    colors = ["#3498db", "#e74c3c"]
    
    # Panel 1: Absolute times for one platform (RTX)
    ax1 = fig.add_subplot(gs[0, 0])
    platform = "RTX"
    positions = []
    data = []
    box_colors = []
    
    for i, N in enumerate(Ns):
        base = i * 1.5
        for j, impl in enumerate(impls):
            vals = buckets.get((platform, impl, N), [])
            if not vals:
                continue
            pos = base + j * 0.4
            positions.append(pos)
            data.append(vals)
            box_colors.append(colors[j])
    
    if data:
        bp = ax1.boxplot(data, positions=positions, widths=0.35, patch_artist=True,
                         showfliers=False, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    tick_pos = [i * 1.5 + 0.2 for i in range(len(Ns))]
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels([str(n) for n in Ns])
    ax1.set_yscale("log")
    ax1.set_xlabel("N")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title(f"OpenCL Times ({platform})")
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Speedups comparison
    ax2 = fig.add_subplot(gs[0, 1])
    platform = "RTX"
    positions = []
    data = []
    box_colors = []
    
    for i, N in enumerate(Ns):
        base = i * 1.5
        for j, impl in enumerate(impls):
            vals = speedups.get((platform, impl, N), [])
            if not vals:
                continue
            pos = base + j * 0.4
            positions.append(pos)
            data.append(vals)
            box_colors.append(colors[j])
    
    if data:
        bp = ax2.boxplot(data, positions=positions, widths=0.35, patch_artist=True,
                         showfliers=False, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([str(n) for n in Ns])
    ax2.set_xlabel("N")
    ax2.set_ylabel("Speedup vs Sequential")
    ax2.set_title(f"Speedup ({platform})")
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Improvement factors (both platforms)
    ax3 = fig.add_subplot(gs[0, 2])
    platform_colors = ["#2ecc71", "#f39c12"]
    positions = []
    data = []
    box_colors = []
    
    for i, N in enumerate(Ns):
        base = i * 1.5
        for j, plat in enumerate(platforms):
            vals = improvements.get((plat, N), [])
            if not vals:
                continue
            pos = base + j * 0.4
            positions.append(pos)
            data.append(vals)
            box_colors.append(platform_colors[j])
    
    if data:
        bp = ax3.boxplot(data, positions=positions, widths=0.35, patch_artist=True,
                         showfliers=False, showmeans=True, meanline=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([str(n) for n in Ns])
    ax3.set_xlabel("N")
    ax3.set_ylabel("Improvement Factor")
    ax3.set_title("Optimization Improvement")
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_impl = [
        Patch(facecolor=colors[0], alpha=0.6, label='Original'),
        Patch(facecolor=colors[1], alpha=0.6, label='Optimized')
    ]
    legend_plat = [
        Patch(facecolor=platform_colors[0], alpha=0.6, label='AMD'),
        Patch(facecolor=platform_colors[1], alpha=0.6, label='RTX')
    ]
    
    fig.legend(handles=legend_impl, loc='upper left', ncol=2, bbox_to_anchor=(0.12, 0.98))
    fig.legend(handles=legend_plat, loc='upper right', ncol=2, bbox_to_anchor=(0.88, 0.98))
    
    fig.suptitle("Complete Performance Overview", y=1.02, fontsize=15, fontweight='bold')
    fig.savefig(OUT_ALL, dpi=200, bbox_inches='tight')
    print(f"[DONE] Wrote {OUT_ALL}")


# ---------- Main ----------

def main():
    missing = []
    for _, fname in FILES:
        if not os.path.exists(fname):
            missing.append(fname)

    if missing:
        print("[ERROR] Missing input files:")
        for m in missing:
            print(f"  - {m}")
        return 1

    os.makedirs(PLOTS_DIR, exist_ok=True)

    rows = []
    platforms = []
    for label, fname in FILES:
        platforms.append(label)
        rows.extend(load_rows(fname, label))

    if not rows:
        print("[ERROR] No usable rows found (type=int).")
        return 1

    Ns = all_Ns(rows)
    buckets = collect_buckets(rows)
    speedups = collect_speedups(rows)
    improvements = collect_improvements(rows)

    # Generate all plots
    plot_opencl_comparison(buckets, Ns, platforms)
    plot_speedup_comparison(speedups, Ns, platforms)
    plot_improvement(improvements, Ns, platforms)
    plot_bar_times(buckets, Ns, platforms)
    plot_bar_times_opencl_only(buckets, Ns, platforms)
    plot_complete_overview(buckets, speedups, improvements, Ns, platforms)

    print("\n[DONE] All plots generated successfully!")
    print(f"  - {OUT_OPENCL_COMPARE}")
    print(f"  - {OUT_SPEEDUP}")
    print(f"  - {OUT_IMPROVEMENT}")
    print(f"  - {OUT_BAR_TIMES}")
    print(f"  - {OUT_BAR_OPENCL_ONLY}")
    print(f"  - {OUT_ALL}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())