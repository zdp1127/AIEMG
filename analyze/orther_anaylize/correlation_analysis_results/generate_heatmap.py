#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


corr_matrix = np.array([
    [1.0000, 0.1583, -0.2573, 0.4332, 0.3355],
    [0.1583, 1.0000, -0.2485, 0.3159, 0.1915],
    [-0.2573, -0.2485, 1.0000, -0.4925, 0.0196],
    [0.4332, 0.3159, -0.4925, 1.0000, 0.2164],
    [0.3355, 0.1915, 0.0196, 0.2164, 1.0000]
])


dim_names = ['Docking', 'QED', 'IER', 'SA_Score', 'ACS']


plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


cmap = sns.diverging_palette(250, 10, as_cmap=True)

ax = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.3f',
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=1,
    linecolor='white',
    annot_kws={'size': 14, 'weight': 'bold'},
    cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation Coefficient'}
)


ax.set_xticklabels(dim_names, fontsize=13, fontweight='bold', rotation=45, ha='right')
ax.set_yticklabels(dim_names, fontsize=13, fontweight='bold', rotation=0)

ax.set_title('Correlation Heatmap of Five Dimensions\n(Docking, QED, IER, SA_Score, ACS)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()


output_path = '/home/zhoudp409100230054/zhoudp409100230054/MCTS/AIE_Chem_8.15版本_IER问题/AIEMG/template_for_data/present_8.25/correlation_analysis_results/correlation_heatmap_300dpi.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')


output_path_600 = '/home/zhoudp409100230054/zhoudp409100230054/MCTS/AIE_Chem_8.15版本_IER问题/AIEMG/template_for_data/present_8.25/correlation_analysis_results/correlation_heatmap_600dpi.png'
plt.savefig(output_path_600, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')


print(f"  - 300dpi: {output_path}")
print(f"  - 600dpi: {output_path_600}")

plt.close()
