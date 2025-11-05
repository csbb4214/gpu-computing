import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================
FILES = ["results/results_paul.csv", "results_jonas.csv", "results_peter.csv"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load and prepare data
# ============================================================
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
df["mode"] = df["mode"].astype(str)

# Compute total runtime (ms)
df["time_ms"] = df["total_write"] + df["total_kernel"] + df["total_read"]

# Average over repeated runs
df = (
    df.groupby(["device", "mode", "precision", "N", "IT"], as_index=False)
    .agg({"time_ms": "mean"})
)

# Device order for consistency
devices = ["paul", "jonas", "ifi"]

# ============================================================
# 1️⃣ Plot: compare devices (precision = float)
# ============================================================
sel = df[df["precision"] == "float"]
plt.figure(figsize=(6, 4))

for (n, it), group in sel.groupby(["N", "IT"]):
    devs_present = [dev for dev in devices if dev in group["device"].values]
    x = np.arange(len(devs_present))
    heights = [group[group["device"] == dev]["time_ms"].mean() for dev in devs_present]
    plt.bar(x, heights, label=f"N={n}, IT={it}", alpha=0.7)

plt.xticks(np.arange(len(devices)), devices)
plt.ylabel("time (ms)")
plt.title("Device comparison — precision=float")
plt.legend(title="Params")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_devices_float.png"), dpi=150)
plt.close()

# ============================================================
# 2️⃣ Plot: compare devices (precision = double)
# ============================================================
sel = df[df["precision"] == "double"]
plt.figure(figsize=(6, 4))

for (n, it), group in sel.groupby(["N", "IT"]):
    devs_present = [dev for dev in devices if dev in group["device"].values]
    x = np.arange(len(devs_present))
    heights = [group[group["device"] == dev]["time_ms"].mean() for dev in devs_present]
    plt.bar(x, heights, label=f"N={n}, IT={it}", alpha=0.7)

plt.xticks(np.arange(len(devices)), devices)
plt.ylabel("time (ms)")
plt.title("Device comparison — precision=double")
plt.legend(title="Params")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "compare_devices_double.png"), dpi=150)
plt.close()

# ============================================================
# 3️⃣ IFI device: N / IT comparison per precision
# ============================================================
df_ifi = df[df["device"] == "ifi"]
precision_vals = ["float", "double"]

for prec in precision_vals:
    dfp = df_ifi[df_ifi["precision"] == prec]
    if dfp.empty:
        continue

    ITs = sorted(dfp["IT"].unique())
    Ns = sorted(dfp["N"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("tab10", n_colors=len(ITs))

    x = np.arange(len(Ns))
    for it_idx, it in enumerate(ITs):
        subset = dfp[dfp["IT"] == it]
        times = [subset[subset["N"] == n]["time_ms"].mean() for n in Ns]
        ax.plot(
            x,
            times,
            label=f"IT={it}",
            color=colors[it_idx % len(colors)],
            linestyle="-",
            marker="o",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_xlabel("N")
    ax.set_ylabel("time (ms)")
    ax.set_title(f"IFI parameter comparison — precision={prec}")
    ax.grid(alpha=0.25)
    ax.legend(title="IT")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"IFI_N_IT_{prec}.png"), dpi=150)

    # Log version
    ax.set_yscale("log")
    ax.set_ylabel("time (ms) [log scale]")
    plt.savefig(os.path.join(OUT_DIR, f"IFI_N_IT_{prec}_log.png"), dpi=150)
    plt.close(fig)

print("✅ Plots written to:", OUT_DIR)
