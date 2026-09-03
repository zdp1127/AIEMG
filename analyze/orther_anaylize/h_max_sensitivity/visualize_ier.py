#!/usr/bin/env python3
"""
IER (Intrinsic-Extrinsic Reward) Visualization Script
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path


def set_style():
    """Set plotting style"""
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def load_ier_history(data_dir):
    """Load IER history data"""
    json_path = os.path.join(data_dir, 'present', 'ier_history.json')

    if not os.path.exists(json_path):
        print(f"Warning: IER history file not found: {json_path}")
        print("Please run MCTS to generate IER history data first")
        return None

    with open(json_path, 'r') as f:
        data = json.load(f)

    return {
        'iterations': np.array(data.get('iterations', [])),
        'Nov_P_mean': np.array(data.get('Nov_P_mean', [])),
        'Nov_H_mean': np.array(data.get('Nov_H_mean', [])),
        'Rep_mean': np.array(data.get('Rep_mean', [])),
        'IER_mean': np.array(data.get('IER_mean', [])),
        'pareto_pool_size': np.array(data.get('pareto_pool_size', [])),
        'active_history_size': np.array(data.get('active_history_size', []))
    }


def load_csv_history(data_dir):
    """Load IER history data from CSV"""
    csv_path = os.path.join(data_dir, 'present', 'ier_history.csv')

    if not os.path.exists(csv_path):
        return None

    iterations = []
    nov_p = []
    nov_h = []
    rep = []
    ier = []
    pareto_size = []
    history_size = []

    with open(csv_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 8:
                try:
                    iterations.append(int(parts[0]))
                    nov_p.append(float(parts[4]))
                    nov_h.append(float(parts[5]))
                    rep.append(float(parts[6]))
                    ier.append(float(parts[7]))
                    pareto_size.append(int(parts[8]))
                    history_size.append(int(parts[9]))
                except (ValueError, IndexError):
                    continue

    if not iterations:
        return None

    return {
        'iterations': np.array(iterations),
        'Nov_P_mean': np.array(nov_p),
        'Nov_H_mean': np.array(nov_h),
        'Rep_mean': np.array(rep),
        'IER_mean': np.array(ier),
        'pareto_pool_size': np.array(pareto_size),
        'active_history_size': np.array(history_size)
    }


def smooth_curve(data, window=5):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_nov_h_comparison(data, output_path=None):
    """Plot Nov_H comparison over iterations"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    iterations = data['iterations']

    if len(iterations) > 0:
        ax1.plot(iterations, data['Nov_H_mean'], 'b-', linewidth=2, label='MCTS+IER', alpha=0.7)

        if len(iterations) > 10:
            smoothed = smooth_curve(data['Nov_H_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax1.plot(iterations_smooth, smoothed, 'b--', linewidth=2, label='Smoothed')

    ax1.set_xlabel('MCTS Iterations')
    ax1.set_ylabel('Nov_H')
    ax1.set_title('Nov_H vs MCTS Iterations')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2 = axes[1]
    if len(iterations) > 0:
        ax2.semilogy(iterations, data['Nov_H_mean'] + 0.01, 'b-', linewidth=2, label='MCTS+IER', alpha=0.7)

        if len(iterations) > 10:
            smoothed = smooth_curve(data['Nov_H_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax2.semilogy(iterations_smooth, smoothed + 0.01, 'b--', linewidth=2, label='Smoothed')

    ax2.set_xlabel('MCTS Iterations')
    ax2.set_ylabel('Nov_H (log scale)')
    ax2.set_title('Nov_H vs MCTS Iterations (Log Scale)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {output_path}")

    plt.close()


def plot_all_components(data, output_path=None):
    """Plot all IER components"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations = data['iterations']

    ax1 = axes[0, 0]
    if len(iterations) > 0:
        ax1.plot(iterations, data['Nov_P_mean'], 'b-', linewidth=2, label='Nov_P', alpha=0.7)
        if len(iterations) > 10:
            smoothed = smooth_curve(data['Nov_P_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax1.plot(iterations_smooth, smoothed, 'b--', linewidth=2, label='Smoothed')
    ax1.set_xlabel('MCTS Iterations')
    ax1.set_ylabel('Nov_P')
    ax1.set_title('Pareto-front Novelty (Nov_P)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    ax2 = axes[0, 1]
    if len(iterations) > 0:
        ax2.plot(iterations, data['Nov_H_mean'], 'g-', linewidth=2, label='Nov_H', alpha=0.7)
        if len(iterations) > 10:
            smoothed = smooth_curve(data['Nov_H_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax2.plot(iterations_smooth, smoothed, 'g--', linewidth=2, label='Smoothed')
    ax2.set_xlabel('MCTS Iterations')
    ax2.set_ylabel('Nov_H')
    ax2.set_title('History-based Novelty (Nov_H)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])

    ax3 = axes[1, 0]
    if len(iterations) > 0:
        ax3.plot(iterations, data['Rep_mean'], 'r-', linewidth=2, label='Rep', alpha=0.7)
        if len(iterations) > 10:
            smoothed = smooth_curve(data['Rep_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax3.plot(iterations_smooth, smoothed, 'r--', linewidth=2, label='Smoothed')
    ax3.set_xlabel('MCTS Iterations')
    ax3.set_ylabel('Rep')
    ax3.set_title('Repetition Penalty (Rep)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])

    ax4 = axes[1, 1]
    if len(iterations) > 0:
        ax4.plot(iterations, data['IER_mean'], 'm-', linewidth=2, label='IER', alpha=0.7)
        if len(iterations) > 10:
            smoothed = smooth_curve(data['IER_mean'], window=min(10, len(iterations)//3))
            iterations_smooth = iterations[:len(smoothed)]
            ax4.plot(iterations_smooth, smoothed, 'm--', linewidth=2, label='Smoothed')
    ax4.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='Zero line')
    ax4.set_xlabel('MCTS Iterations')
    ax4.set_ylabel('IER')
    ax4.set_title('Intrinsic-Extrinsic Reward (IER)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {output_path}")

    plt.close()


def plot_pool_sizes(data, output_path=None):
    """Plot molecular pool size changes"""
    fig, ax = plt.subplots(figsize=(10, 5))

    iterations = data['iterations']

    if len(iterations) > 0:
        ax.plot(iterations, data['pareto_pool_size'], 'b-', linewidth=2,
                label='Pareto Pool Size', alpha=0.7)
        ax.plot(iterations, data['active_history_size'], 'r-', linewidth=2,
                label='Active History Size', alpha=0.7)

    ax.set_xlabel('MCTS Iterations')
    ax.set_ylabel('Number of Molecules')
    ax.set_title('Molecular Pool Sizes over Iterations')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {output_path}")

    plt.close()


def plot_combined_ier(data, output_path=None):
    """Plot IER component stacked chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    iterations = data['iterations']

    if len(iterations) > 0:
        # Plot each component
        ax.fill_between(iterations, 0, data['Nov_P_mean'], alpha=0.3, label='alpha*Nov_P', color='blue')
        ax.fill_between(iterations, data['Nov_P_mean'],
                        data['Nov_P_mean'] + (1-data['alpha'] if 'alpha' in data else 0.5) * data['Nov_H_mean'],
                        alpha=0.3, label='(1-alpha)*Nov_H', color='green')
        ax.plot(iterations, data['IER_mean'], 'r-', linewidth=2, label='IER (final)')
        ax.plot(iterations, data['Nov_P_mean'], 'b--', linewidth=1, alpha=0.5)
        ax.plot(iterations, data['Nov_H_mean'], 'g--', linewidth=1, alpha=0.5)

    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('MCTS Iterations')
    ax.set_ylabel('Reward Value')
    ax.set_title('IER Components Decomposition')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='IER Visualization Tool')
    parser.add_argument('--data_dir', '-d', type=str,
                       default=os.path.dirname(os.path.abspath(__file__)),
                       help='Data directory path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory path (defaults to data directory)')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output or data_dir

    set_style()

    print(f"Loading IER history data: {data_dir}")

    # Try loading JSON format
    data = load_ier_history(data_dir)

    # If JSON doesn't exist, try CSV format
    if data is None:
        data = load_csv_history(data_dir)

    if data is None or len(data['iterations']) == 0:
        print("Error: Unable to load IER history data")
        print("Please ensure MCTS has run and generated ier_history.json or ier_history.csv")
        return

    print(f"Loaded {len(data['iterations'])} data points")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Nov_H comparison chart
    plot_nov_h_comparison(
        data,
        output_path=os.path.join(output_dir, 'ier_nov_h_comparison.png')
    )

    # 2. All IER components chart
    plot_all_components(
        data,
        output_path=os.path.join(output_dir, 'ier_all_components.png')
    )

    # 3. Molecular pool size changes chart
    plot_pool_sizes(
        data,
        output_path=os.path.join(output_dir, 'ier_pool_sizes.png')
    )

    # 4. IER decomposition stacked chart
    plot_combined_ier(
        data,
        output_path=os.path.join(output_dir, 'ier_combined.png')
    )

    print("\nVisualization complete!")
    print(f"Output directory: {output_dir}")
    print("Generated charts:")
    print("  - ier_nov_h_comparison.png")
    print("  - ier_all_components.png")
    print("  - ier_pool_sizes.png")
    print("  - ier_combined.png")


if __name__ == '__main__':
    main()
