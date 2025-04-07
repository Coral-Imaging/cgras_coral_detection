#!/usr/bin/env python3

"""
results_analyzer.py
Analyzes and visualizes results from multiple coral detection scaling experiment runs.
"""

import os
import json
import argparse
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from scipy import stats

def find_experiment_runs(base_dir, experiment_prefix):
    """Find all experiment runs with a specific prefix."""
    base_path = Path(base_dir)
    run_paths = list(base_path.glob(f"{experiment_prefix}*"))
    run_paths = [p for p in run_paths if p.is_dir()]
    
    print(f"Found {len(run_paths)} experiment runs for prefix '{experiment_prefix}'")
    return run_paths

def load_metrics_from_runs(run_paths):
    """Load metrics from all experiment runs."""
    all_metrics = {}
    
    for run_path in run_paths:
        run_name = run_path.name
        metrics_file = run_path / "metrics.json"
        
        if not metrics_file.exists():
            print(f"Warning: No metrics.json found for run {run_name}")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Convert string keys to integers
            metrics = {int(k): v for k, v in metrics.items()}
            all_metrics[run_name] = metrics
            print(f"Loaded metrics for run {run_name} with {len(metrics)} data points")
        
        except Exception as e:
            print(f"Error loading metrics for run {run_name}: {e}")
    
    return all_metrics

def combine_metrics(all_metrics):
    """Combine metrics from multiple runs into a single dataframe."""
    # Collect all image counts across all experiments
    all_counts = set()
    for run_metrics in all_metrics.values():
        all_counts.update(run_metrics.keys())
    
    all_counts = sorted(all_counts)
    
    # Create dataframe for combined metrics
    combined_data = []
    
    for run_name, run_metrics in all_metrics.items():
        for count, metrics in run_metrics.items():
            row = {
                'run': run_name,
                'image_count': count,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'mAP50': metrics['mAP50'],
                'mAP50-95': metrics['mAP50-95']
            }
            combined_data.append(row)
    
    df = pd.DataFrame(combined_data)
    return df

def calculate_summary_statistics(df):
    """Calculate summary statistics for each image count."""
    # Group by image count and calculate statistics
    summary = df.groupby('image_count').agg({
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max'],
        'mAP50': ['mean', 'std', 'min', 'max'],
        'mAP50-95': ['mean', 'std', 'min', 'max']
    })
    
    # Flatten the multi-index columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Add confidence intervals (95%)
    for metric in ['precision', 'recall', 'mAP50', 'mAP50-95']:
        group_counts = df.groupby('image_count').size()
        
        # Calculate confidence intervals
        t_crit = stats.t.ppf(0.975, group_counts - 1)
        summary[f'{metric}_ci95'] = t_crit * summary[f'{metric}_std'] / np.sqrt(group_counts)
    
    return summary

def fit_power_law(x, y):
    """Fit a power law curve to the data (y = a * x^b)."""
    # Convert to log space for linear regression
    log_x = np.log(x)
    log_y = np.log(y)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
    
    # Convert back from log space
    a = np.exp(intercept)
    b = slope
    
    # Generate fitted curve
    y_fit = a * np.power(x, b)
    
    return a, b, r_value**2, y_fit

def plot_combined_metrics(df, summary, output_dir):
    """Create plots of combined metrics with error bars."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
    metric_titles = {
        'precision': 'Precision',
        'recall': 'Recall',
        'mAP50': 'mAP@50',
        'mAP50-95': 'mAP@50-95'
    }
    
    # Combined plot
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        x = summary.index.values
        y = summary[f'{metric}_mean'].values
        yerr = summary[f'{metric}_ci95'].values
        
        plt.errorbar(x, y, yerr=yerr, marker='o', label=metric_titles[metric], capsize=4)
    
    plt.xscale('log')
    plt.xlabel('Number of Training Images (log scale)')
    plt.ylabel('Metric Value')
    plt.title('Coral Detection Performance vs. Training Set Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add x-ticks with actual values
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()
    
    plt.savefig(output_path / "combined_performance.png", dpi=300)
    
    # Individual plots with fitted curves
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        x = summary.index.values
        y_mean = summary[f'{metric}_mean'].values
        y_err = summary[f'{metric}_ci95'].values
        
        # Plot actual data with error bars
        plt.errorbar(x, y_mean, yerr=y_err, marker='o', label='Measured', color='blue', capsize=4)
        
        # Fit power law curve
        a, b, r2, y_fit = fit_power_law(x, y_mean)
        
        # Plot fitted curve
        plt.plot(x, y_fit, 'r--', label=f'Power Law Fit (y = {a:.3f} × x^{b:.3f}, R² = {r2:.3f})')
        
        plt.xscale('log')
        plt.xlabel('Number of Training Images (log scale)')
        plt.ylabel(metric_titles[metric])
        plt.title(f'{metric_titles[metric]} vs. Training Set Size')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add x-ticks with actual values
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.tight_layout()
        
        plt.savefig(output_path / f"{metric}_performance.png", dpi=300)
    
    # Create plot showing all runs (with jitter for visibility)
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Group data by image count
        grouped = df.groupby('image_count')
        
        # Plot individual runs with jitter
        for count, group in grouped:
            # Add jitter to x-position for visibility
            jitter = np.random.normal(0, 0.02, size=len(group))
            jittered_x = np.array([count] * len(group)) * (1 + jitter)
            
            plt.scatter(jittered_x, group[metric], alpha=0.5, label='_nolegend_')
        
        # Plot mean and confidence interval
        x = summary.index.values
        y_mean = summary[f'{metric}_mean'].values
        y_err = summary[f'{metric}_ci95'].values
        
        plt.errorbar(x, y_mean, yerr=y_err, marker='o', label='Mean ± 95% CI', 
                     color='red', capsize=4, markersize=8, linewidth=2)
        
        plt.xscale('log')
        plt.xlabel('Number of Training Images (log scale)')
        plt.ylabel(metric_titles[metric])
        plt.title(f'{metric_titles[metric]} vs. Training Set Size (All Runs)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(ScalarFormatter())
        plt.tight_layout()
        
        plt.savefig(output_path / f"{metric}_all_runs.png", dpi=300)
    
    return output_path

def calculate_diminishing_returns(summary, metrics):
    """Calculate points of diminishing returns for each metric."""
    diminishing_returns = {}
    
    for metric in metrics:
        x = summary.index.values
        y = summary[f'{metric}_mean'].values
        
        # Calculate percentage improvement between adjacent points
        improvements = []
        for i in range(1, len(x)):
            absolute_improvement = y[i] - y[i-1]
            percentage_improvement = absolute_improvement / y[i-1] * 100
            improvements.append({
                'from_count': x[i-1],
                'to_count': x[i],
                'absolute_improvement': absolute_improvement,
                'percentage_improvement': percentage_improvement,
                'images_added': x[i] - x[i-1]
            })
        
        # Calculate improvement per image
        for imp in improvements:
            imp['improvement_per_image'] = imp['absolute_improvement'] / imp['images_added']
        
        diminishing_returns[metric] = improvements
    
    return diminishing_returns

def export_results_to_csv(df, summary, diminishing_returns, output_dir):
    """Export all results to CSV files."""
    output_path = Path(output_dir)
    
    # Export combined metrics
    df.to_csv(output_path / "all_runs_metrics.csv", index=False)
    
    # Export summary statistics
    summary.to_csv(output_path / "summary_statistics.csv")
    
    # Export diminishing returns data
    for metric, data in diminishing_returns.items():
        pd.DataFrame(data).to_csv(output_path / f"{metric}_diminishing_returns.csv", index=False)
    
    print(f"Exported results to {output_path}")

def recommend_optimal_image_count(summary, diminishing_returns, metrics):
    """Recommend optimal image count based on diminishing returns."""
    recommendations = {}
    
    # Criteria thresholds
    thresholds = {
        'percentage_improvement': 5.0,  # Less than 5% improvement
        'improvement_per_image': 0.001  # Very small improvement per additional image
    }
    
    for metric in metrics:
        improvements = diminishing_returns[metric]
        
        # Find point where improvement falls below thresholds
        for i, imp in enumerate(improvements):
            if imp['percentage_improvement'] < thresholds['percentage_improvement']:
                recommendations[metric] = {
                    'recommended_count': imp['from_count'],
                    'reason': f"Less than {thresholds['percentage_improvement']}% improvement when adding more images",
                    'percentage_improvement': imp['percentage_improvement'],
                    'of_maximum': summary[f'{metric}_mean'].iloc[-1] / summary[f'{metric}_mean'][imp['from_count']] * 100
                }
                break
            
            if imp['improvement_per_image'] < thresholds['improvement_per_image']:
                recommendations[metric] = {
                    'recommended_count': imp['from_count'],
                    'reason': f"Very small improvement per additional image ({imp['improvement_per_image']:.6f})",
                    'percentage_improvement': imp['percentage_improvement'],
                    'of_maximum': summary[f'{metric}_mean'].iloc[-1] / summary[f'{metric}_mean'][imp['from_count']] * 100
                }
                break
        
        # If no recommendation made, use the maximum
        if metric not in recommendations:
            recommendations[metric] = {
                'recommended_count': summary.index.max(),
                'reason': "No clear diminishing returns detected, using maximum count",
                'percentage_improvement': None,
                'of_maximum': 100.0
            }
    
    return recommendations

def create_recommendation_plot(summary, recommendations, output_dir):
    """Create a plot showing recommended image counts."""
    output_path = Path(output_dir)
    
    metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
    metric_titles = {
        'precision': 'Precision',
        'recall': 'Recall',
        'mAP50': 'mAP@50',
        'mAP50-95': 'mAP@50-95'
    }
    
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        x = summary.index.values
        y = summary[f'{metric}_mean'].values
        
        # Plot the full curve
        plt.plot(x, y, marker='o', label=metric_titles[metric])
        
        # Mark the recommendation point
        recommended_count = recommendations[metric]['recommended_count']
        recommended_value = summary[f'{metric}_mean'][recommended_count]
        
        plt.scatter([recommended_count], [recommended_value], marker='*', s=200, 
                   color='red', zorder=10, label='_nolegend_')
        
        # Add annotation
        plt.annotate(f"Optimal: {int(recommended_count)}",
                    xy=(recommended_count, recommended_value),
                    xytext=(0, 20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->'),
                    ha='center')
    
    plt.xscale('log')
    plt.xlabel('Number of Training Images (log scale)')
    plt.ylabel('Metric Value')
    plt.title('Recommended Image Counts for Optimal Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    plt.tight_layout()
    
    plt.savefig(output_path / "recommended_image_counts.png", dpi=300)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze results from coral detection scaling experiments")
    parser.add_argument('--base_dir', required=True, help='Base directory containing experiment runs')
    parser.add_argument('--experiment_prefix', default='CoralScaling', help='Prefix for experiment run directories')
    parser.add_argument('--output_dir', default='analysis_results', help='Directory for output files')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Find all experiment runs
    run_paths = find_experiment_runs(args.base_dir, args.experiment_prefix)
    
    if not run_paths:
        print(f"No experiment runs found in {args.base_dir} with prefix '{args.experiment_prefix}'")
        return
    
    # Load metrics from all runs
    all_metrics = load_metrics_from_runs(run_paths)
    
    if not all_metrics:
        print("No valid metrics found in any experiment run")
        return
    
    # Combine metrics into a single dataframe
    df = combine_metrics(all_metrics)
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(df)
    
    # Define metric list
    metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']
    
    # Calculate points of diminishing returns
    diminishing_returns = calculate_diminishing_returns(summary, metrics)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create plots
    plot_combined_metrics(df, summary, output_dir)
    
    # Export results to CSV
    export_results_to_csv(df, summary, diminishing_returns, output_dir)
    
    # Recommend optimal image count
    recommendations = recommend_optimal_image_count(summary, diminishing_returns, metrics)
    
    # Print recommendations
    print("\nRecommended Image Counts:")
    for metric, rec in recommendations.items():
        print(f"{metric}: {int(rec['recommended_count'])} images ({rec['of_maximum']:.1f}% of maximum performance)")
        print(f"  Reason: {rec['reason']}")
    
    # Create recommendation plot
    create_recommendation_plot(summary, recommendations, output_dir)
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()