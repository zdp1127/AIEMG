#!/usr/bin/env python3
"""
Five-Dimension Correlation Analysis Script
Analyze the meaning and correlation of five dimensions in scores.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Data path (dynamically resolved from script location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_DIR = os.path.dirname(_SCRIPT_DIR)  # template_for_data directory
DATA_DIR = os.path.join(_TEMPLATE_DIR, "present")
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_results")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Five-dimension names (based on code comments)
DIMENSION_NAMES = {
    0: 'Docking Score\n(Docking)',
    1: 'QED\n(Drug-likeness)',
    2: 'IER/eToxPred\n(Overall/Toxicity)',
    3: 'SA Score\n(Synthetic Accessibility)',
    4: 'ACS\n(Activity Cliff Score)'
}

# Simplified dimension names (for plot labels)
DIM_NAMES_SHORT = ['Docking', 'QED', 'IER', 'SA_Score', 'ACS']

def load_scores(filepath):
    """Load score data"""
    scores = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Remove brackets
                line = line.strip('[]')
                parts = [float(x.strip()) for x in line.split(',')]
                scores.append(parts)
    return np.array(scores)

def analyze_dimensions(scores):
    """Analyze statistical features of each dimension"""
    print("=" * 80)
    print("Five-Dimension Statistical Analysis")
    print("=" * 80)
    
    dim_descriptions = {
        0: "Docking Score: Molecular docking score, reflecting binding affinity to target, higher is better",
        1: "QED (Drug-likeness): Quantitative estimate of drug-likeness, evaluating if molecule has drug-like properties, higher is better",
        2: "IER/eToxPred (Overall/Toxicity): Integrated evaluation score or toxicity prediction score, higher is better",
        3: "SA Score (Synthetic Accessibility): Synthesis complexity score, higher is easier to synthesize",
        4: "ACS (Activity Cliff Score): Activity Cliff Score, evaluating activity differences between similar molecules"
    }
    
    for i, name in DIMENSION_NAMES.items():
        data = scores[:, i]
        print(f"\n[Dimension {i+1}] {dim_descriptions[i]}")
        print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
        print(f"  Median: {np.median(data):.4f}")

def compute_correlation_matrix(scores):
    """Compute correlation matrix"""
    return np.corrcoef(scores.T)

def perform_statistical_tests(scores):
    """Perform statistical significance tests"""
    n_dims = scores.shape[1]
    print("\n" + "=" * 80)
    print("Pearson Correlation Coefficient Statistical Significance Test")
    print("=" * 80)
    
    results = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            r, p = stats.pearsonr(scores[:, i], scores[:, j])
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            results.append({
                'Dim1': DIM_NAMES_SHORT[i],
                'Dim2': DIM_NAMES_SHORT[j],
                'r': r,
                'p_value': p,
                'significance': significance
            })
            print(f"{DIM_NAMES_SHORT[i]} vs {DIM_NAMES_SHORT[j]}: r={r:.4f}, p={p:.2e} {significance}")
    
    return results

def create_heatmap(corr_matrix, output_path):
    """Create correlation heatmap"""
    # Set font support for Chinese
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask (only show lower triangle)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Draw heatmap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    heatmap = sns.heatmap(
        corr_matrix,
        mask=None,  # Show full matrix
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt='.3f',
        annot_kws={'size': 11, 'weight': 'bold'},
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        xticklabels=DIM_NAMES_SHORT,
        yticklabels=DIM_NAMES_SHORT,
        ax=ax
    )
    
    ax.set_title('Five-Dimension Correlation Heatmap\n(Dimension Correlation Analysis)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add description text
    description = """
    Dimensions:
    1. Docking: Molecular docking score (binding affinity)
    2. QED: Quantitative estimate of drug-likeness
    3. IER: Integrated Evaluation Rating / eToxPred
    4. SA_Score: Synthetic Accessibility score
    5. ACS: Activity Cliff Score
    """
    ax.text(0.5, -0.15, description, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nHeatmap saved to: {output_path}")

def create_pairplot(scores, output_path):
    """Create pairplot (scatter matrix)"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    df = pd.DataFrame(scores, columns=DIM_NAMES_SHORT)
    
    g = sns.pairplot(df, diag_kind='kde', 
                     plot_kws={'alpha': 0.5, 's': 20, 'edgecolor': 'none'},
                     diag_kws={'fill': True, 'alpha': 0.6})
    
    g.fig.suptitle('Five-Dimension Pairwise Relationships', y=1.02, fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(g.axes.flatten()):
        ax.set_xlabel(DIM_NAMES_SHORT[i % 5], fontsize=9)
        ax.set_ylabel(DIM_NAMES_SHORT[i // 5], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Pairplot saved to: {output_path}")

def create_distribution_plot(scores, output_path):
    """Create distribution plots for each dimension"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i in range(5):
        ax = axes[i]
        data = scores[:, i]
        
        # Draw histogram and KDE
        ax.hist(data, bins=30, density=True, alpha=0.7, color=colors[i], edgecolor='white')
        
        # Add KDE curve
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), color='black', linewidth=2)
        
        # Add statistical info
        stats_text = f'mean={data.mean():.3f}\nstd={data.std():.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(DIM_NAMES_SHORT[i], fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.suptitle('Five-Dimension Value Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Distribution plot saved to: {output_path}")

def generate_report(scores, corr_matrix, stat_results):
    """Generate analysis report"""
    report_path = os.path.join(OUTPUT_DIR, "correlation_analysis_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Five-Dimension Correlation Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("[Dimension Description]\n")
        f.write("-" * 40 + "\n")
        dim_descriptions = {
            0: "Docking Score: Molecular docking score, reflecting binding affinity to target",
            1: "QED (Drug-likeness): Quantitative estimate of drug-likeness, evaluating if molecule has drug-like properties",
            2: "IER/eToxPred (Overall/Toxicity): Integrated evaluation score or toxicity prediction score",
            3: "SA Score (Synthetic Accessibility): Synthesis complexity score, higher is easier to synthesize",
            4: "ACS (Activity Cliff Score): Activity Cliff Score, evaluating activity differences between similar molecules"
        }
        for i, desc in dim_descriptions.items():
            f.write(f"  Dimension{i+1}: {desc}\n")
        
        f.write("\n[Correlation Matrix]\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'':>12}")
        for name in DIM_NAMES_SHORT:
            f.write(f"{name:>12}")
        f.write("\n")
        for i, row in enumerate(corr_matrix):
            f.write(f"{DIM_NAMES_SHORT[i]:>12}")
            for val in row:
                f.write(f"{val:>12.4f}")
            f.write("\n")
        
        f.write("\n[Correlation Interpretation]\n")
        f.write("-" * 40 + "\n")
        for result in stat_results:
            r = result['r']
            if abs(r) >= 0.7:
                strength = "strong correlation"
            elif abs(r) >= 0.4:
                strength = "moderate correlation"
            elif abs(r) >= 0.2:
                strength = "weak correlation"
            else:
                strength = "almost no correlation"
            
            direction = "positive" if r > 0 else "negative"
            f.write(f"  {result['Dim1']} vs {result['Dim2']}: {direction} {strength} (r={r:.4f}) {result['significance']}\n")
        
        f.write("\n[Key Findings]\n")
        f.write("-" * 40 + "\n")
        
        # Find strongest positive and negative correlations
        sorted_results = sorted(stat_results, key=lambda x: abs(x['r']), reverse=True)
        
        f.write("  1. Strongest correlation:\n")
        f.write(f"     - {sorted_results[0]['Dim1']} vs {sorted_results[0]['Dim2']}: r={sorted_results[0]['r']:.4f}\n")
        
        # Find significantly correlated dimension pairs
        sig_results = [r for r in stat_results if r['p_value'] < 0.05]
        if sig_results:
            f.write("\n  2. Statistically significant correlations (p<0.05):\n")
            for r in sig_results[:5]:
                f.write(f"     - {r['Dim1']} vs {r['Dim2']}: r={r['r']:.4f} {r['significance']}\n")
    
    print(f"\nAnalysis report saved to: {report_path}")
    return report_path

def main():
    print("Starting five-dimension correlation analysis...\n")
    
    # Load data
    scores_path = os.path.join(DATA_DIR, "scores.txt")
    scores = load_scores(scores_path)
    print(f"Loaded {scores.shape[0]} molecules with {scores.shape[1]} dimensions\n")
    
    # Analyze dimensions
    analyze_dimensions(scores)
    
    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(scores)
    
    # Statistical tests
    stat_results = perform_statistical_tests(scores)
    
    # Generate charts
    print("\n" + "=" * 80)
    print("Generating visualization charts")
    print("=" * 80)
    
    create_heatmap(corr_matrix, os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    create_pairplot(scores, os.path.join(OUTPUT_DIR, "pairplot.png"))
    create_distribution_plot(scores, os.path.join(OUTPUT_DIR, "distributions.png"))
    
    # Generate report
    generate_report(scores, corr_matrix, stat_results)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
