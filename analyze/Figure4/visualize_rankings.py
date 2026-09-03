import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set academic style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# Read the two ranking files
egfr_path = r'E:\MCTS\2026.7.10修改意见版本\chembl网页\chembl_with_dock\aiemg_vs_chembl_ranked.csv'
her2_path = r'E:\MCTS\2026.7.10修改意见版本\chembl网页\chembl_with_dock\aiemg_vs_chembl_her2_ranked.csv'

df_egfr = pd.read_csv(egfr_path)
df_her2 = pd.read_csv(her2_path)

aiemg_egfr = df_egfr[df_egfr['source'] == 'AIEMG']['rank'].values
aiemg_her2 = df_her2[df_her2['source'] == 'AIEMG']['rank'].values

total_egfr = len(df_egfr)
total_her2 = len(df_her2)

# Calculate percentile for each molecule
percentile_egfr = (1 - aiemg_egfr / total_egfr) * 100
percentile_her2 = (1 - aiemg_her2 / total_her2) * 100

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ============ Plot 1: Cumulative Distribution (CDF) ============
ax1 = axes[0, 0]
for name, percentile, color in [('EGFR', percentile_egfr, '#2E86AB'), ('HER2', percentile_her2, '#E94F37')]:
    sorted_pct = np.sort(percentile)
    cdf = np.arange(1, len(sorted_pct) + 1) / len(sorted_pct) * 100
    ax1.plot(sorted_pct, cdf, label=name, linewidth=2, color=color)
ax1.set_xlabel('Percentile of AIEMG Molecules', fontsize=12)
ax1.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax1.set_title('A', fontsize=14, fontweight='bold', loc='left')
ax1.legend(loc='lower right', framealpha=0.9)
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(50, color='gray', linestyle='--', alpha=0.5)

# ============ Plot 2: Rank Distribution Histogram ============
ax2 = axes[0, 1]
bins = np.linspace(0, 100, 21)  # Top 0-5%, 5-10%, etc.
ax2.hist(percentile_egfr, bins=bins, alpha=0.6, label='EGFR', color='#2E86AB', edgecolor='white')
ax2.hist(percentile_her2, bins=bins, alpha=0.6, label='HER2', color='#E94F37', edgecolor='white')
ax2.set_xlabel('Percentile Rank', fontsize=12)
ax2.set_ylabel('Number of Molecules', fontsize=12)
ax2.set_title('B', fontsize=14, fontweight='bold', loc='left')
ax2.legend(loc='upper right')
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.set_xticklabels(['Top 0-20%', '20-40%', '40-60%', '60-80%', '80-100%', '>100%'])
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3, axis='y')

# ============ Plot 3: Box Plot Comparison ============
ax3 = axes[1, 0]
box_data = [percentile_egfr, percentile_her2]
bp = ax3.boxplot(box_data, labels=['EGFR', 'HER2'], patch_artist=True)
colors = ['#2E86AB', '#E94F37']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax3.set_ylabel('Percentile Rank', fontsize=12)
ax3.set_title('C', fontsize=14, fontweight='bold', loc='left')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Median')
ax3.legend(loc='lower right')

# Add statistics text
for i, (pct, name) in enumerate([(percentile_egfr, 'EGFR'), (percentile_her2, 'HER2')]):
    ax3.text(i + 1, 5, f'Mean: {pct.mean():.1f}\nMedian: {np.median(pct):.1f}', 
             ha='center', va='bottom', fontsize=9)

# ============ Plot 4: Bar Chart - Top-N Summary ============
ax4 = axes[1, 1]
top_n = [1, 5, 10, 25, 50]
thresholds = [99, 95, 90, 75, 50]

egfr_counts = [(percentile_egfr >= t).sum() / len(aiemg_egfr) * 100 for t in thresholds]
her2_counts = [(percentile_her2 >= t).sum() / len(aiemg_her2) * 100 for t in thresholds]

x = np.arange(len(top_n))
width = 0.35
bars1 = ax4.bar(x - width/2, egfr_counts, width, label='EGFR', color='#2E86AB', alpha=0.7)
bars2 = ax4.bar(x + width/2, her2_counts, width, label='HER2', color='#E94F37', alpha=0.7)

ax4.set_ylabel('Percentage of AIEMG Molecules (%)', fontsize=12)
ax4.set_xlabel('Top-N Threshold', fontsize=12)
ax4.set_title('D', fontsize=14, fontweight='bold', loc='left')
ax4.set_xticks(x)
ax4.set_xticklabels([f'Top {n}' for n in top_n])
ax4.legend(loc='upper right')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
output_path = r'E:\MCTS\2026.7.10修改意见版本\chembl网页\chembl_with_dock\aiemg_ranking_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Figure saved: {output_path}')

# Print summary statistics
print('\n' + '='*60)
print('Summary Statistics')
print('='*60)
print(f'\nEGFR:')
print(f'  Mean percentile: {percentile_egfr.mean():.2f}')
print(f'  Median percentile: {np.median(percentile_egfr):.2f}')
print(f'  Top 10% molecules: {(percentile_egfr >= 90).sum()} ({(percentile_egfr >= 90).sum()/len(aiemg_egfr)*100:.1f}%)')
print(f'  Top 50% molecules: {(percentile_egfr >= 50).sum()} ({(percentile_egfr >= 50).sum()/len(aiemg_egfr)*100:.1f}%)')

print(f'\nHER2:')
print(f'  Mean percentile: {percentile_her2.mean():.2f}')
print(f'  Median percentile: {np.median(percentile_her2):.2f}')
print(f'  Top 10% molecules: {(percentile_her2 >= 90).sum()} ({(percentile_her2 >= 90).sum()/len(aiemg_her2)*100:.1f}%)')
print(f'  Top 50% molecules: {(percentile_her2 >= 50).sum()} ({(percentile_her2 >= 50).sum()/len(aiemg_her2)*100:.1f}%)')
