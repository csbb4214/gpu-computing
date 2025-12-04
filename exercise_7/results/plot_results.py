#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional, nur für hübsches Styling
from matplotlib.ticker import LogLocator, LogFormatter

# -------------------------------------------------------
# Konfiguration
# -------------------------------------------------------

FILES = [
    "auto_levels_results_rtx.csv",
    "auto_levels_results_amd.csv",
    "auto_levels_results_local.csv",
]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_INFO = {
    "local": "Local machine (RTX 2070 Laptop GPU)",
    "rtx": "IFI – RTX 2070",
    "amd": "IFI – AMD GPU",
}

# hübsches Theme (kannst du auch weglassen)
sns.set_theme(style="whitegrid", context="talk")


# -------------------------------------------------------
# Daten laden & vorbereiten
# -------------------------------------------------------

dfs = []
for f in FILES:
    if not os.path.exists(f):
        print(f"⚠️ Missing file: {f}")
        continue

    df_part = pd.read_csv(f)
    df_part.columns = df_part.columns.str.strip()

    # Dateiname -> Device:
    # auto_levels_results_rtx.csv -> "rtx"
    base = os.path.splitext(os.path.basename(f))[0]
    if base.startswith("auto_levels_results_"):
        dev = base.replace("auto_levels_results_", "")
    elif base.startswith("auto_levels_"):
        dev = base.replace("auto_levels_", "")
    elif base.startswith("results_"):
        dev = base.replace("results_", "")
    else:
        dev = base

    df_part["device"] = dev

    # Strings aufräumen
    if "impl" in df_part.columns:
        df_part["impl"] = df_part["impl"].astype(str).str.strip()

    dfs.append(df_part)

if not dfs:
    raise SystemExit("No data files found!")

df = pd.concat(dfs, ignore_index=True)

# Typen setzen
df["elapsed_ms"] = df["elapsed_ms"].astype(float)
# time_ia / time_ib können bei 'serial' NaN sein, das ist okay
if "time_ia" in df.columns:
    df["time_ia"] = pd.to_numeric(df["time_ia"], errors="coerce")
if "time_ib" in df.columns:
    df["time_ib"] = pd.to_numeric(df["time_ib"], errors="coerce")

print(f"Loaded {len(df)} rows in total")
print("Devices:", df["device"].unique())
print("Implementations:", df["impl"].unique())


# -------------------------------------------------------
# Aggregation: Mittelwert + Std pro Device & impl
# -------------------------------------------------------

stats = (
    df.groupby(["device", "impl"])["elapsed_ms"]
      .agg(mean="mean", std="std", count="count")
      .reset_index()
)

print("\nAggregated stats (elapsed_ms):")
print(stats)

# optional: Pivot für spätere Auswertungen
pivot_mean = stats.pivot(index="device", columns="impl", values="mean")


# -------------------------------------------------------
# Plot: Balken serial vs opencl pro Device
# -------------------------------------------------------

def plot_serial_vs_opencl_bar(log_y: bool = False):
    devices = sorted(stats["device"].unique())
    x = np.arange(len(devices))
    width = 0.35

    means_serial = []
    means_opencl = []
    stds_serial = []
    stds_opencl = []

    for dev in devices:
        row_serial = stats[(stats["device"] == dev) & (stats["impl"] == "serial")]
        row_opencl = stats[(stats["device"] == dev) & (stats["impl"] == "opencl")]

        if not row_serial.empty:
            means_serial.append(row_serial["mean"].values[0])
            stds_serial.append(row_serial["std"].values[0])
        else:
            means_serial.append(np.nan)
            stds_serial.append(0.0)

        if not row_opencl.empty:
            means_opencl.append(row_opencl["mean"].values[0])
            stds_opencl.append(row_opencl["std"].values[0])
        else:
            means_opencl.append(np.nan)
            stds_opencl.append(0.0)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Bars für serial & opencl
    ax.bar(
        x - width / 2,
        means_serial,
        width,
        yerr=stds_serial,
        capsize=5,
        label="serial",
    )
    ax.bar(
        x + width / 2,
        means_opencl,
        width,
        yerr=stds_opencl,
        capsize=5,
        label="opencl",
    )

    # x-Ticks = Devices (schönere Namen aus DEVICE_INFO)
    labels = [DEVICE_INFO.get(d, d) for d in devices]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    ax.set_ylabel("Time (ms)")

    title = "auto_levels – serial vs opencl (Mean ± Std)"
    if log_y:
        title += " [log-scale]"
    ax.set_title(title)

    # -------- Log-Skala mit feingranularen Ticks + Grid --------
    if log_y:
        ax.set_yscale("log")

        # Major-Ticks: 1·10^k
        major_locator = LogLocator(base=10.0, subs=(1.0,), numticks=20)
        # Minor-Ticks: 2–9·10^k
        minor_locator = LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100)
        ax.yaxis.set_major_locator(major_locator)
        ax.yaxis.set_minor_locator(minor_locator)

        # Labels im Stil 1e1, 1e2, ...
        ax.yaxis.set_major_formatter(LogFormatter(base=10.0, labelOnlyBase=False))

        # horizontale Linien für Major + Minor
        ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, which="minor", linestyle=":", alpha=0.3)
    else:
        # Bei linearer Skala nur normale Gridlines
        ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5)

    ax.legend()

    # Speedup (serial / opencl) als Text über den Bars
    for i, dev in enumerate(devices):
        s = means_serial[i]
        o = means_opencl[i]
        if np.isnan(s) or np.isnan(o) or o == 0:
            continue
        speedup = s / o

        h_max = max(s, o)
        # Bei log-Skala lieber ein bisschen höher skalieren
        y_text = h_max * (1.2 if log_y else 1.03)

        ax.text(
            x[i],
            y_text,
            f"×{speedup:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    suffix = "_logy" if log_y else ""
    out_path = os.path.join(OUT_DIR, f"auto_levels_serial_vs_opencl{suffix}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    # normaler Plot (linear)
    plot_serial_vs_opencl_bar(log_y=False)
    # feingranularer Log-Plot mit 1–10 pro Dekade
    plot_serial_vs_opencl_bar(log_y=True)

    print(f"\nAll plots generated in '{OUT_DIR}' directory.")

