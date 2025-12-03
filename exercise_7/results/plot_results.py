#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # optional, nur für hübsches Styling

# -------------------------------------------------------
# Konfiguration
# -------------------------------------------------------

FILES = [
    "auto_levels_results_rtx.csv",
    "auto_levels_results_amd.csv",
    "auto_levels_results_local.csv"
]

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_INFO = {
    "local": "Local machine",
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

    # Device-Namen aus Dateinamen ableiten:
    # "auto_levels_rtx.csv" -> "rtx"
    dev = os.path.splitext(os.path.basename(f))[0].replace("auto_levels_", "")
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


# Hilfstabelle: für Speedup-Berechnung
# => pivot: Zeilen = device, Spalten = impl, Wert = mean elapsed_ms
pivot_mean = stats.pivot(index="device", columns="impl", values="mean")

# -------------------------------------------------------
# Plot: Balken serial vs opencl pro Device
# -------------------------------------------------------

def plot_serial_vs_opencl_bar():
    devices = sorted(stats["device"].unique())
    impls = ["serial", "opencl"]  # Reihenfolge festlegen

    x = np.arange(len(devices))
    width = 0.35

    # y-Werte (Mittelwerte) bauen
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
    rects1 = ax.bar(
        x - width / 2,
        means_serial,
        width,
        yerr=stds_serial,
        capsize=5,
        label="serial",
    )
    rects2 = ax.bar(
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
    ax.set_title("auto_levels – serial vs opencl (Mean ± Std)")
    ax.legend()

    # Speedup (serial / opencl) als Text über den Bars
    for i, dev in enumerate(devices):
        s = means_serial[i]
        o = means_opencl[i]
        if np.isnan(s) or np.isnan(o) or o == 0:
            continue
        speedup = s / o

        # Position: oberhalb der höheren der beiden Bars
        h_max = max(s, o)
        ax.text(
            x[i],
            h_max * 1.03,
            f"×{speedup:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "auto_levels_serial_vs_opencl.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
