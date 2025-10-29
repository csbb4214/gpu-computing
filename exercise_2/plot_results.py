import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

FILES = ["results_paul.csv", "results_jonas.csv", "results_ifi.csv"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# load CSVs and add device column from filename
dfs = []
for f in FILES:
    if not os.path.exists(f):
        raise SystemExit(f"Missing file: {f}")
    df = pd.read_csv(f)
    dev = os.path.splitext(os.path.basename(f))[0].replace("results_", "")
    df["device"] = dev
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

# ensure proper types
df["N"] = df["N"].astype(int)
df["IT"] = df["IT"].astype(int)
df["time_ms"] = df["time_ms"].astype(float)
df["precision"] = df["precision"].astype(str)
df["mode"] = df["mode"].astype(str)

# fixed mode order for consistent plots
mode_order = ["serial", "openmp", "opencl_V1", "opencl_V2"]
devices = ["paul", "jonas", "ifi"]

# 1) Graph: N=2048 IT=1000 precision=float, one line per device across modes
sel = df[(df["N"] == 2048) & (df["IT"] == 1000) & (df["precision"] == "float")]
plt.figure(figsize=(7,4))
x = np.arange(len(mode_order))
for dev in devices:
    row = sel[sel["device"] == dev].set_index("mode").reindex(mode_order)
    times = row["time_ms"].values
    plt.plot(x, times, marker="o", label=dev)
plt.xticks(x, mode_order, rotation=20)
plt.xlabel("mode")
plt.ylabel("time (ms)")
plt.title("device comparison — N=2048 IT=1000 precision=float")
plt.grid(alpha=0.25)
plt.legend(title="device")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "N2048_IT1000_float_by_device.png"), dpi=150)

# 2) Graph: same for double
sel = df[(df["N"] == 2048) & (df["IT"] == 1000) & (df["precision"] == "double")]
plt.figure(figsize=(7,4))
x = np.arange(len(mode_order))
for dev in devices:
    row = sel[sel["device"] == dev].set_index("mode").reindex(mode_order)
    times = row["time_ms"].values
    plt.plot(x, times, marker="o", label=dev)
plt.xticks(x, mode_order, rotation=20)
plt.xlabel("mode")
plt.ylabel("time (ms)")
plt.title("device comparison — N=2048 IT=1000 precision=double")
plt.grid(alpha=0.25)
plt.legend(title="device")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "N2048_IT1000_double_by_device.png"), dpi=150)

# 3) Graph: compare different values of N and IT for device ifi.
# Produce one line-plot per precision (float/double). X-axis = mode, Y-axis = time_ms.
# Color = IT, linestyle = N. Save both linear and log-y versions.
df_ifi = df[df["device"] == "ifi"]
precision_vals = ["float", "double"]
mode_order = ["serial", "openmp", "opencl_V1", "opencl_V2"]

for prec in precision_vals:
    dfp = df_ifi[df_ifi["precision"] == prec]
    ITs = sorted(dfp["IT"].unique())
    Ns = sorted(dfp["N"].unique())
    x = np.arange(len(mode_order))

    fig, ax = plt.subplots(figsize=(9,5))
    colors = sns.color_palette("tab10", n_colors=max(1, len(ITs)))
    linestyles = ['-', '--', '-.', ':']  # different line styles for different N values
    linewidth = 2.0

    # plot a separate line for each (IT, N) pair: color = IT, linestyle = N
    for it_idx, it in enumerate(ITs):
        for n_idx, n in enumerate(Ns):
            times = []
            subset = dfp[(dfp["IT"] == it) & (dfp["N"] == n)]
            for mode in mode_order:
                vals = subset[subset["mode"] == mode]["time_ms"].values
                times.append(vals[0] if vals.size > 0 else np.nan)
            ax.plot(
                x, times,
                linestyle=linestyles[n_idx % len(linestyles)],
                color=colors[it_idx % len(colors)],
                linewidth=linewidth,
                marker=None,
                alpha=0.95
            )

    ax.set_xticks(x)
    ax.set_xticklabels(mode_order, rotation=20)
    ax.set_xlabel("mode")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"IFI parameter comparison — precision={prec}")
    ax.grid(alpha=0.25)

    # legends: one for IT (colors) and one for N (linestyles)
    color_proxies = [Line2D([0], [0], color=colors[i % len(colors)], lw=3) for i in range(len(ITs))]
    color_labels = [f"IT={it}" for it in ITs]
    linestyle_proxies = [Line2D([0], [0], color='black', lw=3, linestyle=linestyles[i % len(linestyles)]) for i in range(len(Ns))]
    linestyle_labels = [f"N={n}" for n in Ns]

    legend1 = ax.legend(color_proxies, color_labels, loc="upper left")
    ax.add_artist(legend1)
    ax.legend(linestyle_proxies, linestyle_labels, loc="upper right")

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, f"IFI_modes_{prec}.png")
    plt.savefig(outpath, dpi=150)

    # also save a log-y version (useful when times span orders of magnitude)
    ax.set_yscale('log')
    ax.set_ylabel("time (ms) [log scale]")
    ax.grid(which='both', alpha=0.25)
    outpath_log = os.path.join(OUT_DIR, f"IFI_modes_{prec}_log.png")
    plt.savefig(outpath_log, dpi=150)
    plt.close(fig)

print("Plots written to:", OUT_DIR)