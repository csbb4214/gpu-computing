import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Configuration
FILES = ["results/results_paul.csv", "results/results_jonas.csv", 
         "results/results_peter.csv", "results/results_ifi.csv"]
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Device info
DEVICE_INFO = {
    "paul":  "Intel Iris Xe Graphics",
    "jonas": "AMD Radeon RX 6700 XT", 
    "peter": "Nvidia RTX2070 (Laptop)",
    "ifi":   "ifi - Nvidia RTX2070"
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

# Extract version number from "opencl_V2" format
df["version_num"] = df["version"].str.extract(r'V(\d+)').astype(int)
df["version"] = "V" + df["version_num"].astype(str)

# Create workgroup dimension label for plotting
df["workgroup"] = df["LOCAL_WORKGROUP_DIM_1"].astype(str) + "x" + df["LOCAL_WORKGROUP_DIM_2"].astype(str)

# Calculate mean of 5 runs for each configuration
group_cols = ["version", "precision", "N", "IT", "LOCAL_WORKGROUP_DIM_1", "LOCAL_WORKGROUP_DIM_2", "device", "workgroup"]
df_mean = df.groupby(group_cols)["elapsed_ms"].mean().reset_index()

print(f"Loaded data with {len(df_mean)} unique configurations")
print(f"Versions found: {df_mean['version'].unique()}")
print(f"Sample data:\n{df_mean.head()}")

# Filter for N=4096, IT=1000 for the main comparisons
df_filtered = df_mean[
    (df_mean["N"] == 4096) & 
    (df_mean["IT"] == 1000)
]

# 1) V3 ONLY PLOTS
def plot_workgroup_performance_bar_v3_only(precision_val, log_scale=False):
    """Plot workgroup performance as bar graph for V3 only"""
    data = df_filtered[
        (df_filtered["precision"] == precision_val) & 
        (df_filtered["version"] == "V3")
    ]
    
    if data.empty:
        print(f"No V3 data for precision={precision_val} with N=4096, IT=1000")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Get unique devices and workgroups
    devices = sorted(data["device"].unique())
    workgroups = sorted(data["workgroup"].unique())
    
    # Color palette for devices
    colors = sns.color_palette("tab10", n_colors=len(devices))
    
    # Set up bar positions
    x = np.arange(len(workgroups))
    bar_width = 0.8 / len(devices)  # Adjust width based on number of devices
    
    # Plot bars for each device
    for i, device in enumerate(devices):
        device_data = data[data["device"] == device].sort_values("workgroup")
        
        if not device_data.empty:
            # Ensure the data is in the same order as workgroups
            device_times = []
            for wg in workgroups:
                wg_data = device_data[device_data["workgroup"] == wg]
                if not wg_data.empty:
                    device_times.append(wg_data["elapsed_ms"].values[0])
                else:
                    device_times.append(0)
            
            # Calculate bar positions
            bar_positions = x + i * bar_width - (len(devices) - 1) * bar_width / 2
            
            plt.bar(
                bar_positions,
                device_times,
                width=bar_width,
                color=colors[i],
                label=DEVICE_INFO.get(device, device),
                alpha=0.8,
                edgecolor='black',
                linewidth=0.5
            )
    
    plt.xlabel("Local Work Group Sizes", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    
    # Set scale and title based on log_scale parameter
    if log_scale:
        plt.yscale('log')
        scale_label = "Log Scale"
    else:
        scale_label = "Linear Scale"
    
    plt.title(f"V3 Performance - N=4096, IT=1000, {precision_val.capitalize()} ({scale_label})\n", fontsize=14, fontweight='bold')
    plt.legend(title="Device", fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks and labels
    plt.xticks(x, workgroups, rotation=45, ha='right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with appropriate filename
    if log_scale:
        filename = f"v3_performance_N4096_IT1000_{precision_val}_log.png"
    else:
        filename = f"v3_performance_N4096_IT1000_{precision_val}_linear.png"
    
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {filename}")

# 2) V2 vs V3 COMPARISON PLOTS
def plot_workgroup_performance_bar_v2_v3(precision_val, log_scale=False):
    """Plot workgroup performance as bar graph for specific precision with V2 and V3"""
    data = df_filtered[df_filtered["precision"] == precision_val]
    
    if data.empty:
        print(f"No data for precision={precision_val} with N=4096, IT=1000")
        return
    
    plt.figure(figsize=(16, 8))
    
    # Get unique devices, workgroups, and versions
    devices = sorted(data["device"].unique())
    workgroups = sorted(data["workgroup"].unique())
    versions = sorted(data["version"].unique())
    
    # Color palette for devices
    colors = sns.color_palette("tab10", n_colors=len(devices))
    
    # Set up bar positions
    x = np.arange(len(workgroups))
    # Adjust width based on number of devices and versions
    bar_width = 0.8 / (len(devices) * len(versions))
    
    # Plot bars for each device and version
    for device_idx, device in enumerate(devices):
        for version_idx, version in enumerate(versions):
            device_version_data = data[(data["device"] == device) & (data["version"] == version)].sort_values("workgroup")
            
            if not device_version_data.empty:
                # Ensure the data is in the same order as workgroups
                device_times = []
                for wg in workgroups:
                    wg_data = device_version_data[device_version_data["workgroup"] == wg]
                    if not wg_data.empty:
                        device_times.append(wg_data["elapsed_ms"].values[0])
                    else:
                        device_times.append(0)
                
                # Calculate bar positions
                # Group by workgroup, then by device, then by version within device
                overall_index = device_idx * len(versions) + version_idx
                bar_positions = x + overall_index * bar_width - (len(devices) * len(versions) - 1) * bar_width / 2
                
                # Different bar styles for V2 and V3
                if version == "V2":
                    # V2: Hatched pattern with black edges
                    hatch_pattern = "//"
                    edge_color = 'black'
                    alpha = 0.9
                else:  # V3
                    # V3: Solid fill with black edges
                    hatch_pattern = ""
                    edge_color = 'black'
                    alpha = 0.8
                
                plt.bar(
                    bar_positions,
                    device_times,
                    width=bar_width,
                    color=colors[device_idx],
                    alpha=alpha,
                    edgecolor=edge_color,
                    linewidth=1.0,
                    hatch=hatch_pattern
                )
    
    plt.xlabel("Local Work Group Sizes", fontsize=12)
    plt.ylabel("Time (ms)", fontsize=12)
    
    # Set scale and title based on log_scale parameter
    if log_scale:
        plt.yscale('log')
        scale_label = "Log Scale"
    else:
        scale_label = "Linear Scale"
    
    plt.title(f"V2 vs V3 Performance - N=4096, IT=1000, {precision_val.capitalize()} ({scale_label})\n", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis ticks and labels
    plt.xticks(x, workgroups, rotation=45, ha='right')
    
    # Create dual legends: one for devices (colors) and one for versions (hatch patterns)
    color_proxies = [Line2D([0], [0], color=colors[i], lw=4) 
                    for i in range(len(devices))]
    color_labels = [DEVICE_INFO.get(dev, dev) for dev in devices]
    
    # For version patterns
    version_proxies = [
        Patch(facecolor='white', edgecolor='black', hatch='', linewidth=1),  # V3: solid
        Patch(facecolor='white', edgecolor='black', hatch='//', linewidth=1)  # V2: hatched
    ]
    version_labels = ['V3', 'V2']
    
    legend1 = plt.legend(color_proxies, color_labels, loc="upper left", title="Device")
    plt.gca().add_artist(legend1)
    plt.legend(version_proxies, version_labels, loc="upper right", title="Version")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with appropriate filename
    if log_scale:
        filename = f"v2_v3_performance_N4096_IT1000_{precision_val}_log.png"
    else:
        filename = f"v2_v3_performance_N4096_IT1000_{precision_val}_linear.png"
    
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {filename}")

# 3) INDIVIDUAL DEVICE DETAILS PLOTS
def plot_device_details_bar(device_name):
    """Plot individual device performance as bar graph for V3 only with different N and IT values"""
    # Filter for specific device and V3 only
    device_data = df_mean[
        (df_mean["device"] == device_name) & 
        (df_mean["version"] == "V3")
    ]
    
    if device_data.empty:
        print(f"No V3 data found for device: {device_name}")
        return
    
    # Create both linear and log scale plots
    for log_scale in [False, True]:
        plt.figure(figsize=(16, 8))
        
        # Get unique N and IT values
        N_values = sorted(device_data["N"].unique())
        IT_values = sorted(device_data["IT"].unique())
        workgroups = sorted(device_data["workgroup"].unique())
        
        # Color palette for IT values
        colors = sns.color_palette("tab10", n_colors=len(IT_values))
        
        # Hatch patterns for N values
        hatch_patterns = ['', '//', '\\\\', 'xx', '..', '**']
        
        # Set up bar positions
        x = np.arange(len(workgroups))
        # Adjust width based on number of N and IT combinations
        bar_width = 0.8 / (len(N_values) * len(IT_values))
        
        # Plot bars for each combination of N and IT
        for n_idx, N in enumerate(N_values):
            for it_idx, IT in enumerate(IT_values):
                config_data = device_data[
                    (device_data["N"] == N) & 
                    (device_data["IT"] == IT)
                ].sort_values("workgroup")
                
                if not config_data.empty:
                    # Ensure the data is in the same order as workgroups
                    config_times = []
                    for wg in workgroups:
                        wg_data = config_data[config_data["workgroup"] == wg]
                        if not wg_data.empty:
                            config_times.append(wg_data["elapsed_ms"].values[0])
                        else:
                            config_times.append(0)
                    
                    # Calculate bar positions
                    # Group by workgroup, then by N, then by IT within N
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
        
        # Set scale and title
        if log_scale:
            plt.yscale('log')
            scale_label = "Log Scale"
        else:
            scale_label = "Linear Scale"
        
        device_display_name = DEVICE_INFO.get(device_name, device_name)
        plt.title(f"{device_display_name} - V3 Performance ({scale_label})\n", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Set x-axis ticks and labels
        plt.xticks(x, workgroups, rotation=45, ha='right')
        
        # Create dual legends: one for IT (colors) and one for N (hatch patterns)
        color_proxies = [Line2D([0], [0], color=colors[i], lw=4) 
                        for i in range(len(IT_values))]
        color_labels = [f"IT={it}" for it in IT_values]
        
        # For hatch patterns, we create rectangle patches with the hatch patterns
        hatch_proxies = [Patch(facecolor='white', edgecolor='black', 
                              hatch=hatch_patterns[i % len(hatch_patterns)], 
                              linewidth=1) 
                        for i in range(len(N_values))]
        hatch_labels = [f"N={n}" for n in N_values]
        
        legend1 = plt.legend(color_proxies, color_labels, loc="upper left", title="Iterations")
        plt.gca().add_artist(legend1)
        plt.legend(hatch_proxies, hatch_labels, loc="upper right", title="Problem Size")
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        if log_scale:
            filename = f"{device_name}_v3_comparison_log.png"
        else:
            filename = f"{device_name}_v3_comparison_linear.png"
        
        plt.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {filename}")

# GENERATE ALL PLOTS

# 1. V3 Only plots
for precision in ["float", "double"]:
    # Linear scale
    plot_workgroup_performance_bar_v3_only(precision, log_scale=False)
    # Log scale
    plot_workgroup_performance_bar_v3_only(precision, log_scale=True)

# 2. V2 vs V3 Comparison plots
for precision in ["float", "double"]:
    # Linear scale
    plot_workgroup_performance_bar_v2_v3(precision, log_scale=False)
    # Log scale
    plot_workgroup_performance_bar_v2_v3(precision, log_scale=True)

# 3. Individual device details plots for all devices
devices = ["paul", "jonas", "peter", "ifi"]
for device in devices:
    plot_device_details_bar(device)

print(f"\nAll plots generated in '{OUT_DIR}' directory!")
print("Graphs created:")
print("V3 Only:")
print("  1. V3 Performance Bar - N=4096, IT=1000, Float (Linear Scale)")
print("  2. V3 Performance Bar - N=4096, IT=1000, Float (Log Scale)")
print("  3. V3 Performance Bar - N=4096, IT=1000, Double (Linear Scale)")
print("  4. V3 Performance Bar - N=4096, IT=1000, Double (Log Scale)")
print("V2 vs V3 Comparison:")
print("  5. V2 vs V3 Performance Bar - N=4096, IT=1000, Float (Linear Scale)")
print("  6. V2 vs V3 Performance Bar - N=4096, IT=1000, Float (Log Scale)")
print("  7. V2 vs V3 Performance Bar - N=4096, IT=1000, Double (Linear Scale)")
print("  8. V2 vs V3 Performance Bar - N=4096, IT=1000, Double (Log Scale)")
print("Individual Device Details:")
for i, device in enumerate(devices, start=9):
    device_name = DEVICE_INFO.get(device, device)
    print(f"  {i}. {device_name} V3 Comparison Bar (Linear Scale)")
    print(f"  {i+1}. {device_name} V3 Comparison Bar (Log Scale)")