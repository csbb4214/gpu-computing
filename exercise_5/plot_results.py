#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

FILES = [
    "results/results_paul.csv",
    "results/results_peter.csv",
    "results/results_ifi.csv",
]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_INFO = {
    "paul": "Intel Iris Xe Graphics",
    "peter": "Nvidia RTX2070 (Laptop)",
    "ifi": "ifi - Nvidia RTX2070",
}

VERSION_LABELS = {
    "sequential_reduction": "Sequential",
    "parallel_reduction": "Parallel WG",
    "multistage_reduction": "Multi-stage",
}

VERSION_ORDER = ["sequential_reduction", "parallel_reduction", "multistage_reduction"]

# Different markers per algorithm
VERSION_MARKERS = {
    "sequential_reduction": "o",  # circle
    "parallel_reduction": "s",    # square
    "multistage_reduction": "X",  # cross
}

sns.set_theme(style="whitegrid", context="talk")


# -------------------------------------------------------
# Load & prepare data
# -------------------------------------------------------

dfs = []
for f in FILES:
    if not os.path.exists(f):
        print(f"⚠️ Missing file: {f}")
        continue

    df = pd.read_csv(f)

    # Strip whitespace from column names (e.g. "elapsed_ms ")
    df.columns = df.columns.str.strip()

    dev = os.path.splitext(os.path.basename(f))[0].replace("results_", "")
    df["device"] = dev

    # Clean string columns defensively
    if "precision" in df.columns:
        df["precision"] = df["precision"].astype(str).str.strip()
    if "version" in df.columns:
        df["version"] = df["version"].astype(str).str.strip()

    dfs.append(df)

if not dfs:
    raise SystemExit("No data files found!")

df = pd.concat(dfs, ignore_index=True)

# Type conversions
df["N"] = df["N"].astype(int)
df["elapsed_ms"] = df["elapsed_ms"].astype(float)

df = df.sort_values("N")

group_cols = ["device", "version", "precision", "N"]
df_mean = (
    df.groupby(group_cols)["elapsed_ms"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "elapsed_ms_mean", "std": "elapsed_ms_std"})
)

print(f"Loaded data with {len(df_mean)} unique configurations")
print("Devices:", df_mean["device"].unique())
print("Versions:", df_mean["version"].unique())
print("Precisions:", df_mean["precision"].unique())


# -------------------------------------------------------
# Helper: nicer N labels
# -------------------------------------------------------

def format_N_label(n: int) -> str:
    if n == 1024:
        return "1024"
    elif n == 1024**2:
        return "1024²"
    elif n == 1024**2 * 512:
        return "1024²×512"
    else:
        return f"{n}"


# -------------------------------------------------------
# Plot 1: Runtime vs N per device & precision (log-x, points only)
# -------------------------------------------------------

def plot_runtime_vs_N_per_device(precision_val: str, log_y: bool = False) -> None:
    """
    For each device:
      x-axis: vector size N (numeric, log scale)
      y-axis: time in ms (mean)
      markers: different reduction versions (no connecting lines)
    """
    data_prec = df_mean[df_mean["precision"] == precision_val]

    if data_prec.empty:
        print(f"No data for precision={precision_val}")
        return

    devices = sorted(data_prec["device"].unique())

    for dev in devices:
        dev_data = data_prec[data_prec["device"] == dev]
        if dev_data.empty:
            continue

        plt.figure(figsize=(10, 6))

        Ns = sorted(dev_data["N"].unique())

        for version in VERSION_ORDER:
            ver_data = dev_data[dev_data["version"] == version]
            if ver_data.empty:
                continue

            x_nums = []
            y_vals = []

            for n in Ns:
                row = ver_data[ver_data["N"] == n]
                if not row.empty:
                    x_nums.append(float(n))
                    y_vals.append(row["elapsed_ms_mean"].values[0])

            if not x_nums:
                continue

            y_vals = np.array(y_vals, dtype=float)

            if log_y:
                positive = y_vals[y_vals > 0]
                if positive.size > 0:
                    min_pos = np.min(positive)
                else:
                    min_pos = 1e-6
                y_vals[y_vals <= 0] = min_pos / 10.0

            marker = VERSION_MARKERS.get(version, "o")

            plt.scatter(
                x_nums,
                y_vals,
                s=70,
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                label=VERSION_LABELS.get(version, version),
            )

        plt.xscale("log")
        plt.xticks(Ns, [format_N_label(n) for n in Ns])
        plt.xlabel("Vector size N")
        plt.ylabel("Time (ms)")

        dev_name = DEVICE_INFO.get(dev, dev)
        scale_label_y = " (log y-scale)" if log_y else ""
        plt.title(f"{dev_name} – Runtime vs N – {precision_val}{scale_label_y}")

        if log_y:
            plt.yscale("log")

        plt.legend(title="Algorithm")
        plt.tight_layout()

        fname = f"runtime_vs_N_{dev}_{precision_val}"
        if log_y:
            fname += "_logy"
        fname += ".png"

        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Plot 2: Comparison of all devices for largest N (bars, improved log ticks + labels)
# -------------------------------------------------------

def plot_comparison_largest_N(precision_val: str, log_y: bool = False) -> None:
    """
    For the largest N:
      x-axis: device
      bars: different reduction versions
      When log_y=True: more fine-grained log ticks and value labels on top of bars.
    """
    data_prec = df_mean[df_mean["precision"] == precision_val]

    if data_prec.empty:
        print(f"No data for precision={precision_val}")
        return

    max_N = data_prec["N"].max()
    data_N = data_prec[data_prec["N"] == max_N]

    if data_N.empty:
        print(f"No data for precision={precision_val} with N={max_N}")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    devices = sorted(data_N["device"].unique())
    x = np.arange(len(devices))
    bar_width = 0.25  # 3 versions -> 3 bars per device

    all_rects = []  # collect bars for value labels

    for i, version in enumerate(VERSION_ORDER):
        ver_data = data_N[data_N["version"] == version]
        if ver_data.empty:
            continue

        heights = []
        for dev in devices:
            row = ver_data[ver_data["device"] == dev]
            if not row.empty:
                heights.append(row["elapsed_ms_mean"].values[0])
            else:
                heights.append(0.0)

        heights = np.array(heights, dtype=float)

        if log_y:
            positive = heights[heights > 0]
            if positive.size > 0:
                min_pos = np.min(positive)
            else:
                min_pos = 1e-6
            heights[heights <= 0] = min_pos / 10.0

        rects = ax.bar(
            x + (i - 1) * bar_width,  # -1, 0, 1 for 3 versions
            heights,
            width=bar_width,
            label=VERSION_LABELS.get(version, version),
            edgecolor="black",
            alpha=0.8,
        )
        all_rects.extend(rects)

    plt.xticks(
        x,
        [DEVICE_INFO.get(dev, dev) for dev in devices],
        rotation=15,
        ha="right",
    )
    plt.xlabel("Device")
    plt.ylabel("Time (ms)")
    scale_label_y = " (log y-scale)" if log_y else ""
    plt.title(
        f"Device comparison for N={format_N_label(max_N)} – {precision_val}{scale_label_y}"
    )

    if log_y:
        ax.set_yscale("log")

        # More fine-grained log ticks
        # Major ticks: powers of 10
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        # Minor ticks: 2,3,...9 * 10^n
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # Slightly denser grid to help reading
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

    # Add value labels on top of bars (helps a lot for log plot)
    for rect in all_rects:
        height = rect.get_height()
        if height <= 0:
            continue
        # Slight offset above the bar (works for linear and log)
        offset = 1.05 if not log_y else 1.1
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            height * offset,
            f"{height:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

    plt.legend(title="Algorithm")
    plt.tight_layout()

    fname = f"comparison_largestN_{precision_val}"
    if log_y:
        fname += "_logy"
    fname += ".png"

    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Plot 3: Speedup relative to sequential_reduction (points only, markers)
# -------------------------------------------------------

def compute_speedup_df() -> pd.DataFrame | None:
    """
    Create a pivot table:
      index: device, precision, N
      columns: version
      values: elapsed_ms_mean
    Then add speedup columns: sequential_time / algo_time
    """
    pivot = df_mean.pivot_table(
        index=["device", "precision", "N"],
        columns="version",
        values="elapsed_ms_mean",
    ).reset_index()

    if "sequential_reduction" not in pivot.columns:
        print("⚠️ No sequential_reduction data found, skipping speedup plots.")
        return None

    for ver in VERSION_ORDER:
        if ver == "sequential_reduction":
            continue
        if ver in pivot.columns:
            pivot[f"speedup_{ver}"] = pivot["sequential_reduction"] / pivot[ver]

    return pivot


def plot_speedup_vs_N_per_device(precision_val: str) -> None:
    """
    For each device:
      x-axis: vector size N (numeric, log x-scale)
      y-axis: speedup vs sequential
      markers only, different marker per algorithm
    """
    pivot = compute_speedup_df()
    if pivot is None:
        return

    data_prec = pivot[pivot["precision"] == precision_val]
    if data_prec.empty:
        print(f"No speedup data for precision={precision_val}")
        return

    devices = sorted(data_prec["device"].unique())

    for dev in devices:
        dev_data = data_prec[data_prec["device"] == dev]
        if dev_data.empty:
            continue

        plt.figure(figsize=(10, 6))

        Ns = sorted(dev_data["N"].unique())

        for ver in VERSION_ORDER:
            if ver == "sequential_reduction":
                continue

            speed_col = f"speedup_{ver}"
            if speed_col not in dev_data.columns:
                continue

            x_nums = []
            y_vals = []
            for n in Ns:
                row = dev_data[dev_data["N"] == n]
                if not row.empty and not np.isnan(row[speed_col].values[0]):
                    x_nums.append(float(n))
                    y_vals.append(row[speed_col].values[0])

            if not x_nums:
                continue

            marker = VERSION_MARKERS.get(ver, "o")

            plt.scatter(
                x_nums,
                y_vals,
                s=70,
                marker=marker,
                edgecolor="black",
                linewidth=0.8,
                label=f"Speedup vs Sequential: {VERSION_LABELS.get(ver, ver)}",
            )

        plt.xscale("log")
        plt.xticks(Ns, [format_N_label(n) for n in Ns])
        plt.xlabel("Vector size N")
        plt.ylabel("Speedup (sequential_time / algo_time)")
        dev_name = DEVICE_INFO.get(dev, dev)
        plt.title(f"{dev_name} – Speedup vs Sequential – {precision_val}")
        plt.axhline(1.0, linestyle="--", linewidth=1.0, color="gray")

        plt.legend()
        plt.tight_layout()

        fname = f"speedup_vs_N_{dev}_{precision_val}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Generate all plots
# -------------------------------------------------------

if __name__ == "__main__":
    # 1) Runtime vs N – per device and precision (linear y + log y), points only
    for prec in ["int", "float"]:
        plot_runtime_vs_N_per_device(prec, log_y=False)
        plot_runtime_vs_N_per_device(prec, log_y=True)

    # 2) Device comparison for largest N (bars, with fine log ticks & labels)
    for prec in ["int", "float"]:
        plot_comparison_largest_N(prec, log_y=False)
        plot_comparison_largest_N(prec, log_y=True)

    # 3) Speedup vs sequential (points only)
    for prec in ["int", "float"]:
        plot_speedup_vs_N_per_device(prec)

    print(f"\nAll plots generated in '{OUT_DIR}' directory.")

