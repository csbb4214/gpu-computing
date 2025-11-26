#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(nvidia_file='results_nvidia.csv', amd_file='results_amd.csv'):
    """Load benchmark results from CSV files"""
    nvidia = pd.read_csv(nvidia_file)
    nvidia['platform'] = 'IFIgpu2070 (NVIDIA)'
    
    amd = pd.read_csv(amd_file)
    amd['platform'] = 'IFIAMD (AMD)'
    
    return pd.concat([nvidia, amd], ignore_index=True)

def calculate_statistics(df):
    """Calculate mean, std, min, max for each configuration"""
    stats = df.groupby(['platform', 'version', 'precision', 'N']).agg({
        'elapsed_ms': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    stats.columns = ['platform', 'version', 'precision', 'N', 
                     'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'runs']
    
    return stats

def calculate_speedup(stats):
    """Calculate speedup factors"""
    # Baseline: sequential on each platform
    baseline_nvidia = stats[(stats['platform'] == 'IFIgpu2070 (NVIDIA)') & 
                           (stats['version'] == 'sequential_reduction')]
    baseline_amd = stats[(stats['platform'] == 'IFIAMD (AMD)') & 
                        (stats['version'] == 'sequential_reduction')]
    
    results = []
    
    for platform in stats['platform'].unique():
        platform_data = stats[stats['platform'] == platform]
        baseline = baseline_nvidia if 'NVIDIA' in platform else baseline_amd
        
        for _, row in platform_data.iterrows():
            if row['version'] != 'sequential_reduction':
                base = baseline[(baseline['precision'] == row['precision']) & 
                               (baseline['N'] == row['N'])]
                
                if not base.empty:
                    speedup = base['mean_ms'].values[0] / row['mean_ms']
                    results.append({
                        'platform': platform,
                        'version': row['version'],
                        'precision': row['precision'],
                        'N': row['N'],
                        'speedup': speedup,
                        'mean_ms': row['mean_ms']
                    })
    
    return pd.DataFrame(results)

def plot_performance_comparison(stats, output_dir='plots'):
    """Create performance comparison plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 1. Absolute performance comparison
    for precision in stats['precision'].unique():
        data = stats[stats['precision'] == precision]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Performance Comparison - {precision} precision', fontsize=16)
        
        for idx, n_val in enumerate(sorted(data['N'].unique())):
            ax = axes[idx]
            subset = data[data['N'] == n_val]
            
            x = np.arange(len(subset['version'].unique()))
            width = 0.35
            
            nvidia_data = subset[subset['platform'] == 'IFIgpu2070 (NVIDIA)']
            amd_data = subset[subset['platform'] == 'IFIAMD (AMD)']
            
            versions = sorted(nvidia_data['version'].unique())
            nvidia_means = [nvidia_data[nvidia_data['version'] == v]['mean_ms'].values[0] 
                           if v in nvidia_data['version'].values else 0 for v in versions]
            amd_means = [amd_data[amd_data['version'] == v]['mean_ms'].values[0] 
                        if v in amd_data['version'].values else 0 for v in versions]
            
            ax.bar(x - width/2, nvidia_means, width, label='NVIDIA', alpha=0.8)
            ax.bar(x + width/2, amd_means, width, label='AMD', alpha=0.8)
            
            ax.set_ylabel('Time (ms)')
            ax.set_title(f'N = {n_val:,}')
            ax.set_xticks(x)
            ax.set_xticklabels([v.replace('_reduction', '') for v in versions], 
                              rotation=45, ha='right')
            ax.legend()
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_{precision}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}/")

def plot_speedup_analysis(speedup_df, output_dir='plots'):
    """Create speedup comparison plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    for precision in speedup_df['precision'].unique():
        data = speedup_df[speedup_df['precision'] == precision]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for platform in data['platform'].unique():
            platform_data = data[data['platform'] == platform]
            
            for version in platform_data['version'].unique():
                version_data = platform_data[platform_data['version'] == version]
                version_data = version_data.sort_values('N')
                
                marker = 'o' if 'NVIDIA' in platform else 's'
                linestyle = '-' if version == 'parallel_reduction' else '--'
                
                ax.plot(version_data['N'], version_data['speedup'], 
                       marker=marker, linestyle=linestyle,
                       label=f"{platform} - {version.replace('_reduction', '')}")
        
        ax.set_xlabel('Problem Size (N)')
        ax.set_ylabel('Speedup vs Sequential')
        ax.set_title(f'Speedup Comparison - {precision} precision')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/speedup_{precision}.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_report(stats, speedup_df, output_file='performance_report.txt'):
    """Generate text report with key findings"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("REDUCTION PERFORMANCE ANALYSIS: NVIDIA vs AMD\n")
        f.write("="*80 + "\n\n")
        
        # Platform comparison
        f.write("ABSOLUTE PERFORMANCE (mean execution time in ms)\n")
        f.write("-"*80 + "\n")
        
        for precision in sorted(stats['precision'].unique()):
            f.write(f"\n{precision.upper()} PRECISION:\n")
            data = stats[stats['precision'] == precision]
            
            for n_val in sorted(data['N'].unique()):
                f.write(f"\n  N = {n_val:,}:\n")
                subset = data[data['N'] == n_val]
                
                for version in sorted(subset['version'].unique()):
                    nvidia = subset[(subset['version'] == version) & 
                                   (subset['platform'] == 'IFIgpu2070 (NVIDIA)')]
                    amd = subset[(subset['version'] == version) & 
                                (subset['platform'] == 'IFIAMD (AMD)')]
                    
                    if not nvidia.empty and not amd.empty:
                        nvidia_time = nvidia['mean_ms'].values[0]
                        amd_time = amd['mean_ms'].values[0]
                        ratio = nvidia_time / amd_time
                        
                        f.write(f"    {version:30s}: ")
                        f.write(f"NVIDIA={nvidia_time:8.3f}ms  AMD={amd_time:8.3f}ms  ")
                        f.write(f"Ratio={ratio:.2f}x\n")
        
        # Speedup comparison
        f.write("\n\n" + "="*80 + "\n")
        f.write("SPEEDUP ANALYSIS (vs sequential on same platform)\n")
        f.write("-"*80 + "\n")
        
        for precision in sorted(speedup_df['precision'].unique()):
            f.write(f"\n{precision.upper()} PRECISION:\n")
            data = speedup_df[speedup_df['precision'] == precision]
            
            for n_val in sorted(data['N'].unique()):
                f.write(f"\n  N = {n_val:,}:\n")
                subset = data[data['N'] == n_val]
                
                for version in sorted(subset['version'].unique()):
                    nvidia = subset[(subset['version'] == version) & 
                                   (subset['platform'] == 'IFIgpu2070 (NVIDIA)')]
                    amd = subset[(subset['version'] == version) & 
                                (subset['platform'] == 'IFIAMD (AMD)')]
                    
                    if not nvidia.empty and not amd.empty:
                        nvidia_speedup = nvidia['speedup'].values[0]
                        amd_speedup = amd['speedup'].values[0]
                        
                        f.write(f"    {version:30s}: ")
                        f.write(f"NVIDIA={nvidia_speedup:6.2f}x  AMD={amd_speedup:6.2f}x\n")
        
        # Key findings
        f.write("\n\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-"*80 + "\n")
        
        # Find best performing configurations
        for precision in sorted(speedup_df['precision'].unique()):
            data = speedup_df[speedup_df['precision'] == precision]
            max_n = data['N'].max()
            
            nvidia_best = data[(data['platform'] == 'IFIgpu2070 (NVIDIA)') & 
                              (data['N'] == max_n)].nlargest(1, 'speedup')
            amd_best = data[(data['platform'] == 'IFIAMD (AMD)') & 
                           (data['N'] == max_n)].nlargest(1, 'speedup')
            
            f.write(f"\n{precision.upper()} PRECISION (N={max_n:,}):\n")
            if not nvidia_best.empty:
                f.write(f"  NVIDIA best: {nvidia_best['version'].values[0]} ")
                f.write(f"with {nvidia_best['speedup'].values[0]:.2f}x speedup\n")
            if not amd_best.empty:
                f.write(f"  AMD best: {amd_best['version'].values[0]} ")
                f.write(f"with {amd_best['speedup'].values[0]:.2f}x speedup\n")
    
    print(f"\nReport saved to {output_file}")

def main():
    print("Loading benchmark results...")
    df = load_results()
    
    print("Calculating statistics...")
    stats = calculate_statistics(df)
    
    print("Calculating speedups...")
    speedup_df = calculate_speedup(stats)
    
    print("Generating plots...")
    plot_performance_comparison(stats)
    plot_speedup_analysis(speedup_df)
    
    print("Generating report...")
    generate_report(stats, speedup_df)
    
    print("\n✓ Analysis complete!")
    print("  - Plots saved in plots/")
    print("  - Report saved as performance_report.txt")

if __name__ == '__main__':
    main()