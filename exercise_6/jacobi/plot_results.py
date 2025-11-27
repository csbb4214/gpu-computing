import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Configuration
FILES = ["results/results_2070.csv", "results/results_amd.csv"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Device info
DEVICE_INFO = {
    "2070": "ifi - Nvidia RTX2070",
    "amd": "ifi - AMD"
}

# Load and prepare data
dfs = []
for f in FILES:
    if not os.path.exists(f):
        print(f"⚠️ Missing file: {f}")
        continue
    df = pd.read_csv(f)
    dev = os.path.splitext(os.path.basename(f))[0].replace("results_", "")
    df["device"] = dev
    dfs.append(df)

if not dfs:
    raise SystemExit("No data files found!")

df = pd.concat(dfs, ignore_index=True)

# Ensure proper types
df["N"] = df["N"].astype(int)
df["IT"] = df["IT"].astype(int)
df["LOCAL_WORKGROUP_DIM_1"] = df["LOCAL_WORKGROUP_DIM_1"].astype(int)
df["LOCAL_WORKGROUP_DIM_2"] = df["LOCAL_WORKGROUP_DIM_2"].astype(int)
df["precision"] = df["precision"].astype(str)
df["elapsed_ms"] = df["elapsed_ms"].astype(float)

# Extract version number from "opencl_V3" format
df["version_num"] = df["version"].str.extract(r'V(\d+)').astype(int)
df["version"] = "V" + df["version_num"].astype(str)

# Create workgroup dimension label for plotting
df["workgroup"] = df["LOCAL_WORKGROUP_DIM_1"].astype(str) + "x" + df["LOCAL_WORKGROUP_DIM_2"].astype(str)

# Calculate mean of 5 runs for each configuration
group_cols = ["version", "precision", "N", "IT", "LOCAL_WORKGROUP_DIM_1", "LOCAL_WORKGROUP_DIM_2", "device", "workgroup"]
df_mean = df.groupby(group_cols)["elapsed_ms"].mean().reset_index()

print(f"Loaded data with {len(df_mean)} unique configurations")
print(f"Versions found: {df_mean['version'].unique()}")
print(f"Devices found: {df_mean['device'].unique()}")
print(f"Sample data:\n{df_mean.head()}")

# Filter for V3 only
df_v3 = df_mean[df_mean["version"] == "V3"]

# 1) DEVICE COMPARISON FOR SPECIFIC WORKGROUP (1x256)
def plot_n_it_comparison_for_workgroup(workgroup_dim1, workgroup_dim2, log_scale=False):
    """Plot N and IT comparison for a specific workgroup configuration across devices"""
    
    # Filter for the specific workgroup
    data = df_v3[
        (df_v3["LOCAL_WORKGROUP_DIM_1"] == workgroup_dim1) &
        (df_v3["LOCAL_WORKGROUP_DIM_2"] == workgroup_dim2)
    ]
    
    if data.empty:
        print(f"No data for workgroup {workgroup_dim1}x{workgroup_dim2}")
        return
    
    # Create separate plots for float and double
    for precision in ["float", "double"]:
        precision_data = data[data["precision"] == precision]
        
        if precision_data.empty:
            print(f"No {precision} data for workgroup {workgroup_dim1}x{workgroup_dim2}")
            continue
        
        plt.figure(figsize=(14, 8))
        
        # Get unique values
        devices = sorted(precision_data["device"].unique())
        N_values = sorted(precision_data["N"].unique())
        IT_values = sorted(precision_data["IT"].unique())
        
        # Color palette for devices
        colors = sns.color_palette("tab10", n_colors=len(devices))
        
        # Create x-axis labels combining N and IT
        configs = []
        for N in N_values:
            for IT in IT_values:
                configs.append(f"N={N}\nIT={IT}")
        
        x = np.arange(len(configs))
        bar_width = 0.8 / len(devices)
        
        # Plot bars for each device
        for device_idx, device in enumerate(devices):
            device_data = precision_data[precision_data["device"] == device]
            
            times = []
            for N in N_values:
                for IT in IT_values:
                    config_data = device_data[
                        (device_data["N"] == N) & 
                        (device_data["IT"] == IT)
                    ]
                    if not config_data.empty:
                        times.append(config_data["elapsed_ms"].values[0])
                    else:
                        times.append(0)
            
            bar_positions = x + device_idx * bar_width - (len(devices) - 1) * bar_width / 2
            
            plt.bar(
                bar_positions,
                times,
                width=bar_width,
                color=colors[device_idx],
                label=DEVICE_INFO.get(device, device),
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
        
        plt.xlabel("Problem Size (N) and Iterations (IT)", fontsize=12)
        plt.ylabel("Time (ms)", fontsize=12)
        
        if log_scale:
            plt.yscale('log')
            scale_label = "Log Scale"
        else:
            scale_label = "Linear Scale"
        
        plt.title(f"Device Comparison - Workgroup {workgroup_dim1}x{workgroup_dim2}, {precision.capitalize()} ({scale_label})\n",
                 fontsize=14, fontweight='bold')
        plt.legend(title="Device", fontsize=10, title_fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(x, configs, fontsize=9)
        
        plt.tight_layout()
        
        if log_scale:
            filename = f"device_comparison_wg{workgroup_dim1}x{workgroup_dim2}_{precision}_log.png"
        else:
            filename = f"device_comparison_wg{workgroup_dim1}x{workgroup_dim2}_{precision}_linear.png"
        
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {filename}")

# 2) WORKGROUP COMPARISON FOR FIXED N AND IT
def plot_workgroup_comparison_bar(N_val, IT_val, precision_val, log_scale=False):
    """Plot workgroup performance comparison across devices for fixed N and IT"""
    
    data = df_v3[
        (df_v3["N"] == N_val) & 
        (df_v3["IT"] == IT_val) &
        (df_v3["precision"] == precision_val)
    ]
    
    if data.empty:
        print(f"No data for N={N_val}, IT={IT_val}, precision={precision_val}")
        return
    
    plt.figure(figsize=(14, 8))
    
    devices = sorted(data["device"].unique())
    workgroups = sorted(data["workgroup"].unique())
    
    colors = sns.color_palette("tab10", n_colors=len(devices))
    
    x = np.arange(len(workgroups))
    bar_width = 0.8 / len(devices)
    
    for device_idx, device in enumerate(devices):
        device_data = data[data["device"] == device].sort_values("workgroup")
        
        if not device_data.empty:
            device_times = []
            for wg in workgroups:
                wg_data = device_data[device_data["workgroup"] == wg]
                if not wg_data.empty:
                    device_times.append(wg_data["elapsed_ms"].values[0])
                else:
                    device_times.append(0)
            
            bar_positions = x + device_idx * bar_width - (len(devices) - 1) * bar_width / 2
            
            plt.bar(
                bar_positions,
                device_times,
                width=bar_width,
                color=colors[device_idx],
                label=DEVICE_INFO.get(device, device),
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
    
    plt.xlabel("Local Work Group Sizes", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    
    if log_scale:
        plt.yscale('log')
        scale_label = "Log Scale"
    else:
        scale_label = "Linear Scale"
    
    plt.title(f"Workgroup Comparison - N={N_val}, IT={IT_val}, {precision_val.capitalize()} ({scale_label})\n",
             fontsize=14, fontweight='bold')
    plt.legend(title="Device", fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(x, workgroups, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if log_scale:
        filename = f"workgroup_comparison_N{N_val}_IT{IT_val}_{precision_val}_log.png"
    else:
        filename = f"workgroup_comparison_N{N_val}_IT{IT_val}_{precision_val}_linear.png"
    
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {filename}")

# 3) INDIVIDUAL DEVICE DETAILS
def plot_device_details_bar(device_name):
    """Plot individual device performance with different N and IT values"""
    
    device_data = df_v3[df_v3["device"] == device_name]
    
    if device_data.empty:
        print(f"No V3 data found for device: {device_name}")
        return
    
    # Create separate plots for float and double
    for precision in ["float", "double"]:
        precision_data = device_data[device_data["precision"] == precision]
        
        if precision_data.empty:
            print(f"No {precision} data found for device: {device_name}")
            continue
        
        for log_scale in [False, True]:
            plt.figure(figsize=(16, 8))
            
            N_values = sorted(precision_data["N"].unique())
            IT_values = sorted(precision_data["IT"].unique())
            workgroups = sorted(precision_data["workgroup"].unique())
            
            colors = sns.color_palette("tab10", n_colors=len(IT_values))
            hatch_patterns = ['', '//', '\\\\', 'xx', '..', '**']
            
            x = np.arange(len(workgroups))
            bar_width = 0.8 / (len(N_values) * len(IT_values))
            
            for n_idx, N in enumerate(N_values):
                for it_idx, IT in enumerate(IT_values):
                    config_data = precision_data[
                        (precision_data["N"] == N) & 
                        (precision_data["IT"] == IT)
                    ].sort_values("workgroup")
                    
                    if not config_data.empty:
                        config_times = []
                        for wg in workgroups:
                            wg_data = config_data[config_data["workgroup"] == wg]
                            if not wg_data.empty:
                                config_times.append(wg_data["elapsed_ms"].values[0])
                            else:
                                config_times.append(0)
                        
                        overall_index = n_idx * len(IT_values) + it_idx
                        bar_positions = x + overall_index * bar_width - (len(N_values) * len(IT_values) - 1) * bar_width / 2
                        
                        hatch = hatch_patterns[n_idx % len(hatch_patterns)]
                        
                        plt.bar(
                            bar_positions,
                            config_times,
                            width=bar_width,
                            color=colors[it_idx],
                            alpha=0.8,
                            edgecolor='black',
                            linewidth=0.5,
                            hatch=hatch
                        )
            
            plt.xlabel("Local Work Group Sizes", fontsize=12)
            plt.ylabel("Time (ms)", fontsize=12)
            
            if log_scale:
                plt.yscale('log')
                scale_label = "Log Scale"
            else:
                scale_label = "Linear Scale"
            
            device_display_name = DEVICE_INFO.get(device_name, device_name)
            plt.title(f"{device_display_name} - V3 Performance - {precision.capitalize()} ({scale_label})\n",
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.xticks(x, workgroups, rotation=45, ha='right')
            
            color_proxies = [Line2D([0], [0], color=colors[i], lw=4) 
                            for i in range(len(IT_values))]
            color_labels = [f"IT={it}" for it in IT_values]
            
            hatch_proxies = [Patch(facecolor='white', edgecolor='black', 
                                  hatch=hatch_patterns[i % len(hatch_patterns)], 
                                  linewidth=1) 
                            for i in range(len(N_values))]
            hatch_labels = [f"N={n}" for n in N_values]
            
            legend1 = plt.legend(color_proxies, color_labels, loc="upper left", title="Iterations")
            plt.gca().add_artist(legend1)
            plt.legend(hatch_proxies, hatch_labels, loc="upper right", title="Problem Size")
            
            plt.tight_layout()
            
            if log_scale:
                filename = f"{device_name}_v3_comparison_{precision}_log.png"
            else:
                filename = f"{device_name}_v3_comparison_{precision}_linear.png"
            
            plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Plot saved: {filename}")

# GENERATE ALL PLOTS

print("\n=== Generating Plots ===\n")

# 1. Device comparison for workgroup 1x256 (N and IT variations)
print("1. Device comparison for workgroup 1x256...")
for log_scale in [False, True]:
    plot_n_it_comparison_for_workgroup(1, 256, log_scale=log_scale)

# 2. Workgroup comparison for fixed N and IT
print("\n2. Workgroup comparison for N=4096, IT=1000...")
for precision in ["float", "double"]:
    for log_scale in [False, True]:
        plot_workgroup_comparison_bar(4096, 1000, precision, log_scale=log_scale)

# 3. Individual device details for all devices
print("\n3. Individual device details...")
devices = df_v3["device"].unique()
for device in devices:
    plot_device_details_bar(device)

print(f"\n=== All plots generated in '{OUT_DIR}' directory! ===")
print("\nPlots created:")
print("\n1. Device Comparison (Workgroup 1x256):")
print("   - Float (Linear and Log Scale)")
print("   - Double (Linear and Log Scale)")
print("\n2. Workgroup Comparison (N=4096, IT=1000):")
print("   - Float (Linear and Log Scale)")
print("   - Double (Linear and Log Scale)")
print("\n3. Individual Device Details:")
for device in devices:
    device_name = DEVICE_INFO.get(device, device)
    print(f"   - {device_name} Float (Linear and Log Scale)")
    print(f"   - {device_name} Double (Linear and Log Scale)")