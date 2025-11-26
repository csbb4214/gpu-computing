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
    "results/results_peter.csv",
    "results/results_ifi_rtx.csv",
    "results/results_ifi_amd.csv",
]

OUT_DIR = "plots_matrix_mul"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_INFO = {
    "peter": "RTX 2070 Laptop",
    "ifi_rtx": "ifi – RTX 2070",
    "ifi_amd": "ifi – AMD GPU",
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

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Derive device key from filename: results_peter.csv -> "peter"
    dev = os.path.splitext(os.path.basename(f))[0].replace("results_", "")
    df["device"] = dev

    # Clean string columns defensively
    if "precision" in df.columns:
        df["precision"] = df["precision"].astype(str).str.strip()
    if "impl" in df.columns:
        df["impl"] = df["impl"].astype(str).str.strip()

    dfs.append(df)

if not dfs:
    raise SystemExit("No data files found!")

df = pd.concat(dfs, ignore_index=True)

# Type conversions
df["N"] = df["N"].astype(int)
df["elapsed_ms"] = df["elapsed_ms"].astype(float)

# sort for nicer plots
df = df.sort_values(["device", "precision", "N"])

# We ONLY care about opencl here (falls du später noch openmp etc. drin hast)
if "impl" in df.columns:
    df = df[df["impl"] == "opencl"]

group_cols = ["device", "precision", "N"]
df_mean = (
    df.groupby(group_cols)["elapsed_ms"]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "elapsed_ms_mean", "std": "elapsed_ms_std"})
)

print(f"Loaded data with {len(df_mean)} unique configurations")
print("Devices:", df_mean["device"].unique())
print("Precisions:", df_mean["precision"].unique())
print("N values:", sorted(df_mean["N"].unique()))


# -------------------------------------------------------
# Helper: nicer N labels (hier sind es Matrix-Dimensionen)
# -------------------------------------------------------

def format_N_label(n: int) -> str:
    # simple: just show e.g. "512", "1024", "2000", "2048"
    return str(n)


# -------------------------------------------------------
# Plot 1: Runtime vs N per device (float vs double)
# -------------------------------------------------------

def plot_runtime_vs_N_per_device(log_y: bool = False) -> None:
    """
    Für jedes Device:
      x: N (Matrixdimension)
      y: elapsed_ms_mean
      Linien: float vs double
    """
    devices = sorted(df_mean["device"].unique())

    for dev in devices:
        dev_data = df_mean[df_mean["device"] == dev]
        if dev_data.empty:
            continue

        plt.figure(figsize=(10, 6))

        for prec, marker in zip(["float", "double"], ["o", "s"]):
            prec_data = dev_data[dev_data["precision"] == prec]
            if prec_data.empty:
                continue

            Ns = prec_data["N"].values.astype(float)
            times = prec_data["elapsed_ms_mean"].values.astype(float)

            plt.plot(
                Ns,
                times,
                marker=marker,
                linewidth=2,
                markersize=8,
                label=prec,
            )

        plt.xscale("log")
        Ns_all = sorted(dev_data["N"].unique())
        plt.xticks(Ns_all, [format_N_label(n) for n in Ns_all])
        plt.xlabel("Matrix size N (N×N)")
        plt.ylabel("Time (ms)")

        dev_name = DEVICE_INFO.get(dev, dev)
        scale_label_y = " (log y-scale)" if log_y else ""
        plt.title(f"{dev_name} – Runtime vs N{scale_label_y}")

        if log_y:
            plt.yscale("log")

        plt.legend(title="Precision")
        plt.tight_layout()

        fname = f"runtime_vs_N_{dev}"
        if log_y:
            fname += "_logy"
        fname += ".png"

        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Plot 2: Device comparison per precision (Runtime vs N)
# -------------------------------------------------------

def plot_device_comparison_per_precision(log_y: bool = False) -> None:
    """
    Für jede Precision (float / double):
      x: N
      y: elapsed_ms_mean
      Linien: verschiedene Devices → direkter Vergleich
    """
    for prec in ["float", "double"]:
        data_prec = df_mean[df_mean["precision"] == prec]
        if data_prec.empty:
            print(f"No data for precision={prec}")
            continue

        plt.figure(figsize=(10, 6))

        devices = sorted(data_prec["device"].unique())
        markers = ["o", "s", "D", "^"]

        for dev, marker in zip(devices, markers):
            dev_data = data_prec[data_prec["device"] == dev]
            if dev_data.empty:
                continue

            Ns = dev_data["N"].values.astype(float)
            times = dev_data["elapsed_ms_mean"].values.astype(float)

            plt.plot(
                Ns,
                times,
                marker=marker,
                linewidth=2,
                markersize=8,
                label=DEVICE_INFO.get(dev, dev),
            )

        plt.xscale("log")
        Ns_all = sorted(data_prec["N"].unique())
        plt.xticks(Ns_all, [format_N_label(n) for n in Ns_all])
        plt.xlabel("Matrix size N (N×N)")
        plt.ylabel("Time (ms)")
        scale_label_y = " (log y-scale)" if log_y else ""
        plt.title(f"Device comparison – {prec}{scale_label_y}")

        if log_y:
            plt.yscale("log")

        plt.legend(title="Device")
        plt.tight_layout()

        fname = f"device_comparison_{prec}"
        if log_y:
            fname += "_logy"
        fname += ".png"

        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Plot 3: Bars for largest N (per precision)
# -------------------------------------------------------

def plot_bar_largest_N() -> None:
    """
    Für das größte N:
      Balkenplot pro Precision, je ein Balken pro Device.
      → Sehr direkter Vergleich: welcher GPU ist am schnellsten?
    """
    max_N = df_mean["N"].max()
    data_N = df_mean[df_mean["N"] == max_N]

    if data_N.empty:
        print("No data for largest N.")
        return

    for prec in ["float", "double"]:
        data_prec = data_N[data_N["precision"] == prec]
        if data_prec.empty:
            print(f"No data for precision={prec} at N={max_N}")
            continue

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        devices = sorted(data_prec["device"].unique())
        heights = []
        for dev in devices:
            row = data_prec[data_prec["device"] == dev]
            if not row.empty:
                heights.append(row["elapsed_ms_mean"].values[0])
            else:
                heights.append(np.nan)

        x = np.arange(len(devices))
        rects = ax.bar(
            x,
            heights,
            tick_label=[DEVICE_INFO.get(d, d) for d in devices],
            edgecolor="black",
            alpha=0.9,
        )

        plt.ylabel("Time (ms)")
        plt.title(f"Device comparison at N={max_N} – {prec}")

        # Labels on top of bars
        for rect, h in zip(rects, heights):
            if np.isnan(h):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h * 1.01,
                f"{h:.3g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

        fname = f"bar_largestN_{prec}_N{max_N}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {fname}")


# -------------------------------------------------------
# Generate all plots
# -------------------------------------------------------

if __name__ == "__main__":
    # 1) pro Device: float vs double
    plot_runtime_vs_N_per_device(log_y=False)
    plot_runtime_vs_N_per_device(log_y=True)

    # 2) pro Precision: alle Devices
    plot_device_comparison_per_precision(log_y=False)
    plot_device_comparison_per_precision(log_y=True)

    # 3) Balken für größtes N
    plot_bar_largest_N()

    print(f"\nAll plots generated in '{OUT_DIR}' directory.")

