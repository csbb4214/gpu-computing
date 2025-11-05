import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Updated file paths and device list
FILES = ["results/results_paul.csv", "results/results_jonas.csv", 
         "results/results_peter.csv", "results/results_ifi.csv"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Time columns to plot
TIME_COLS = ["total_kernel", "total_read", "total_write", 
            "write_f", "write_tmp", "write_u", "average_queue"]

# Load CSVs and add device column from filename
dfs = []
for f in FILES:
    if not os.path.exists(f):
        raise SystemExit(f"Missing file: {f}")
    df = pd.read_csv(f)
    dev = os.path.splitext(os.path.basename(f))[0].replace("results_", "")
    df["device"] = dev
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# Ensure proper types
df["N"] = df["N"].astype(int)
df["IT"] = df["IT"].astype(int)
df["precision"] = df["precision"].astype(str)
for col in TIME_COLS:
    df[col] = df[col].astype(float)

# Group by all columns except the time values and calculate mean of 10 runs
group_cols = ["precision", "N", "IT", "device"]
df = df.groupby(group_cols).mean().reset_index()

devices = ["paul", "jonas", "peter", "ifi"]

# 1) Graph: N=2048 IT=1000 precision=float
sel = df[(df["N"] == 2048) & (df["IT"] == 1000) & (df["precision"] == "float")]
plt.figure(figsize=(10,6))
x = np.arange(len(TIME_COLS))
for dev in devices:
    times = sel[sel["device"] == dev][TIME_COLS].values[0]
    plt.plot(x, times, marker="o", label=dev)
plt.xticks(x, TIME_COLS, rotation=45)
plt.xlabel("measurement")
plt.yscale('log')
plt.ylabel("time (ms) [log scale]")
plt.title("device comparison — N=2048 IT=1000 precision=float")
plt.grid(alpha=0.25)
plt.legend(title="device")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "N2048_IT1000_float_by_device.png"), dpi=150)
plt.close()

# 2) Graph: same for double (excluding paul)
sel = df[(df["N"] == 2048) & (df["IT"] == 1000) & (df["precision"] == "double")]
plt.figure(figsize=(10,6))
x = np.arange(len(TIME_COLS))
for dev in [d for d in devices if d != "paul"]:  # exclude paul for double
    times = sel[sel["device"] == dev][TIME_COLS].values[0]
    plt.plot(x, times, marker="o", label=dev)
plt.xticks(x, TIME_COLS, rotation=45)
plt.xlabel("measurement")
plt.yscale('log')
plt.ylabel("time (ms) [log scale]")
plt.title("device comparison — N=2048 IT=1000 precision=double")
plt.grid(alpha=0.25)
plt.legend(title="device")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "N2048_IT1000_double_by_device.png"), dpi=150)
plt.close()

# 3) Graph: compare different values of N and IT for device ifi
df_ifi = df[df["device"] == "ifi"]
precision_vals = ["float", "double"]

for prec in precision_vals:
    dfp = df_ifi[df_ifi["precision"] == prec]
    ITs = sorted(dfp["IT"].unique())
    Ns = sorted(dfp["N"].unique())
    x = np.arange(len(TIME_COLS))

    fig, ax = plt.subplots(figsize=(12,7))
    colors = sns.color_palette("tab10", n_colors=max(1, len(ITs)))
    linestyles = ['-', '--', '-.', ':']
    linewidth = 2.0

    # Plot a separate line for each (IT, N) pair: color = IT, linestyle = N
    for it_idx, it in enumerate(ITs):
        for n_idx, n in enumerate(Ns):
            subset = dfp[(dfp["IT"] == it) & (dfp["N"] == n)]
            if not subset.empty:
                times = subset[TIME_COLS].values[0]
                ax.plot(
                    x, times,
                    linestyle=linestyles[n_idx % len(linestyles)],
                    color=colors[it_idx % len(colors)],
                    linewidth=linewidth,
                    marker="o",
                    alpha=0.95
                )

    ax.set_xticks(x)
    ax.set_xticklabels(TIME_COLS, rotation=45)
    ax.set_xlabel("measurement")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"IFI parameter comparison — precision={prec}")
    ax.grid(alpha=0.25)

    # Legends: one for IT (colors) and one for N (linestyles)
    color_proxies = [Line2D([0], [0], color=colors[i % len(colors)], lw=3) 
                    for i in range(len(ITs))]
    color_labels = [f"IT={it}" for it in ITs]
    linestyle_proxies = [Line2D([0], [0], color='black', lw=3, 
                        linestyle=linestyles[i % len(linestyles)]) 
                        for i in range(len(Ns))]
    linestyle_labels = [f"N={n}" for n in Ns]

    legend1 = ax.legend(color_proxies, color_labels, loc="upper left")
    ax.add_artist(legend1)
    ax.legend(linestyle_proxies, linestyle_labels, loc="upper right")

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, f"IFI_times_{prec}.png")
    plt.savefig(outpath, dpi=150)

    # Log-y version
    ax.set_yscale('log')
    ax.set_ylabel("time (ms) [log scale]")
    ax.grid(which='both', alpha=0.25)
    outpath_log = os.path.join(OUT_DIR, f"IFI_times_{prec}_log.png")
    plt.savefig(outpath_log, dpi=150)
    plt.close(fig)

print("Plots written to:", OUT_DIR)