#!/usr/bin/env python3
"""
Compare RTX vs AMD vs LOCAL scan benchmarks with grouped boxplots.

Expected input files (same directory as this script):
  - scan_benchmark_int_amd.csv
  - scan_benchmark_int_rtx.csv
  - scan_benchmark_int_local.csv

CSV format:
  run,impl,elapsed_ms,N,type,host

We only use rows where type == "int".

Outputs (written into ./plots/):
  - compare_opencl_time.png        (grouped boxplots)
  - compare_sequential_time.png    (grouped boxplots)
  - compare_speedup.png            (grouped boxplots)
  - compare_all_boxplots.png       (combined overview)

Run:
  python3 plot.py
"""

import csv
import os
from collections import defaultdict


FILES = [
    ("AMD", "scan_benchmark_int_amd.csv"),
    ("RTX", "scan_benchmark_int_rtx.csv"),
    ("LOCAL", "scan_benchmark_int_local.csv"),
]

PLOTS_DIR = "plots"

OUT_OPENCL = os.path.join(PLOTS_DIR, "compare_opencl_time.png")
OUT_SEQ = os.path.join(PLOTS_DIR, "compare_sequential_time.png")
OUT_SPEEDUP = os.path.join(PLOTS_DIR, "compare_speedup.png")
OUT_ALL_BOX = os.path.join(PLOTS_DIR, "compare_all_boxplots.png")

# Toggle for showing raw points on top of boxplots
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
    speedups[(platform, N)] = [seq/opencl, ...] for matched runs
    """
    seq_map = {}
    cl_map = {}

    for r in rows:
        key = (r["platform"], r["N"], r["run"])
        if r["impl"] == "sequential":
            seq_map[key] = r["elapsed_ms"]
        elif r["impl"] == "opencl":
            cl_map[key] = r["elapsed_ms"]

    speedups = defaultdict(list)
    for key, seq_ms in seq_map.items():
        if key in cl_map and cl_map[key] > 0:
            platform, N, _run = key
            speedups[(platform, N)].append(seq_ms / cl_map[key])

    return speedups


# ---------- Plot helpers ----------

def _apply_style():
    import matplotlib.pyplot as plt
    # nice default look (matplotlib-only)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass


def _add_jitter_points(ax, ys_list, positions, jitter=0.06, alpha=0.35, markersize=3):
    """
    Overlay raw points with tiny x jitter to show distribution.
    ys_list: list of lists of values
    positions: x positions matching ys_list
    """
    import random

    rnd = random.Random(42)  # deterministic for reproducibility
    for vals, pos in zip(ys_list, positions):
        for v in vals:
            x = pos + (rnd.random() * 2 - 1) * jitter
            ax.plot(x, v, marker="o", linestyle="None", alpha=alpha, markersize=markersize)


def plot_grouped_box_time(buckets, Ns, platforms, impl, out_path, title, show_points=True):
    """
    For each N, draw one box per platform for the given impl.
    """
    import matplotlib.pyplot as plt

    _apply_style()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    num_platforms = len(platforms)

    # layout tuning
    group_gap = 1.35   # distance between N groups
    box_width = 0.22   # width of each platform box

    positions = []
    data = []

    # Build box data in stable order
    for i, N in enumerate(Ns):
        base = i * group_gap
        for j, platform in enumerate(platforms):
            vals = buckets.get((platform, impl, N), [])
            if not vals:
                continue
            pos = base + j * box_width
            positions.append(pos)
            data.append(vals)

    if not data:
        print(f"[WARN] No data for impl={impl} in boxplot.")
        return

    ax.boxplot(
        data,
        positions=positions,
        widths=box_width * 0.9,
        showfliers=True,
        showmeans=True,
        meanline=True,
    )

    if show_points:
        _add_jitter_points(ax, data, positions)

    # Put ticks in the center of each N group
    tick_positions = []
    tick_labels = []
    for i, N in enumerate(Ns):
        base = i * group_gap
        center = base + (num_platforms - 1) * box_width / 2
        tick_positions.append(center)
        tick_labels.append(str(N))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)

    # simple hint for readability
    order_text = " / ".join(platforms)
    ax.text(
        0.01, 0.01,
        f"Platform order per N: {order_text}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[DONE] Wrote {out_path}")


def plot_speedup_box(speedups, Ns, platforms, out_path, show_points=True):
    import matplotlib.pyplot as plt

    _apply_style()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    num_platforms = len(platforms)
    group_gap = 1.35
    box_width = 0.22

    positions = []
    data = []

    for i, N in enumerate(Ns):
        base = i * group_gap
        for j, platform in enumerate(platforms):
            vals = speedups.get((platform, N), [])
            if not vals:
                continue
            pos = base + j * box_width
            positions.append(pos)
            data.append(vals)

    if not data:
        print("[WARN] No speedup data for boxplot.")
        return

    ax.boxplot(
        data,
        positions=positions,
        widths=box_width * 0.9,
        showfliers=True,
        showmeans=True,
        meanline=True,
    )

    if show_points:
        _add_jitter_points(ax, data, positions)

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
    ax.set_ylabel("Speedup (Sequential / OpenCL)")
    ax.set_title("Inclusive Scan Speedup (int): RTX vs AMD vs LOCAL")

    order_text = " / ".join(platforms)
    ax.text(
        0.01, 0.01,
        f"Platform order per N: {order_text}",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom"
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"[DONE] Wrote {out_path}")


# ---------- Specific plots ----------

def plot_opencl_box(buckets, Ns, platforms):
    title = "OpenCL Inclusive Scan Time (int): RTX vs AMD vs LOCAL"
    plot_grouped_box_time(buckets, Ns, platforms, "opencl", OUT_OPENCL, title, show_points=SHOW_POINTS)


def plot_sequential_box(buckets, Ns, platforms):
    title = "Sequential Inclusive Scan Time (int): RTX vs AMD vs LOCAL"
    plot_grouped_box_time(buckets, Ns, platforms, "sequential", OUT_SEQ, title, show_points=SHOW_POINTS)


def plot_all_box_overview(buckets, Ns, platforms):
    """
    One figure with two panels: Sequential + OpenCL grouped boxplots.
    Great for reports.
    """
    import matplotlib.pyplot as plt

    _apply_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)

    num_platforms = len(platforms)
    group_gap = 1.35
    box_width = 0.22

    for ax, impl, label in zip(
        axes,
        ["sequential", "opencl"],
        ["Sequential", "OpenCL"]
    ):
        positions = []
        data = []

        for i, N in enumerate(Ns):
            base = i * group_gap
            for j, platform in enumerate(platforms):
                vals = buckets.get((platform, impl, N), [])
                if not vals:
                    continue
                pos = base + j * box_width
                positions.append(pos)
                data.append(vals)

        if data:
            ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.9,
                showfliers=True,
                showmeans=True,
                meanline=True,
            )
            if SHOW_POINTS:
                _add_jitter_points(ax, data, positions)

        tick_positions = []
        tick_labels = []
        for i, N in enumerate(Ns):
            base = i * group_gap
            center = base + (num_platforms - 1) * box_width / 2
            tick_positions.append(center)
            tick_labels.append(str(N))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(label)
        ax.set_xlabel("N")
        ax.set_yscale("log")

    axes[0].set_ylabel("Time (ms)")
    fig.suptitle("Inclusive Scan Times (int): Boxplot Overview", y=1.02)

    fig.tight_layout()
    fig.savefig(OUT_ALL_BOX, dpi=200)
    print(f"[DONE] Wrote {OUT_ALL_BOX}")


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
        print("Make sure the three CSVs are in the same directory as plot.py.")
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

    # Time boxplots
    plot_opencl_box(buckets, Ns, platforms)
    plot_sequential_box(buckets, Ns, platforms)

    # Speedup boxplots
    speedups = collect_speedups(rows)
    plot_speedup_box(speedups, Ns, platforms, OUT_SPEEDUP, show_points=SHOW_POINTS)

    # Combined overview
    plot_all_box_overview(buckets, Ns, platforms)

    print("[DONE] All comparison plots generated in ./plots/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

