"""
H_max Sensitivity Analysis Experiment Script
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


def calc_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except:
        pass
    return None


def tanimoto_distance(fp1, fp2):
    if fp1 is None or fp2 is None:
        return 1.0
    try:
        sim = DataStructs.FingerprintSimilarity(fp1, fp2)
        return 1.0 - sim
    except:
        return 1.0


def analyze_h_max_sensitivity(molecules, output_dir):
    h_max_values = list(range(50, 1501, 50))
    k_neighbors = 5

    # Store history novelty evolution for each H_max
    history_novelty_evolution = {}
    pareto_novelty_evolution = {}
    repetition_evolution = {}

    # Revisit rate analysis: use full history library (not limited by H_max)
    full_history_fps = []  # Full history library
    revisit_rate_list = []  # Maximum similarity for each molecule

    print(f"Analyzing {len(molecules)} molecules...")
    print(f"Testing H_max values: {h_max_values}")

    # Calculate fingerprints for all molecules
    all_fps = [calc_fingerprint(smi) for smi in molecules]
    valid_fps = [fp for fp in all_fps if fp is not None]

    for h_max in h_max_values:
        print(f"\nProcessing H_max = {h_max}...")

        history_fps = deque(maxlen=h_max)
        pareto_fps = []

        nov_h_list = []
        nov_p_list = []
        rep_list = []

        pareto_fps = []

        for i, fp in enumerate(all_fps):
            if fp is None:
                nov_h_list.append(np.nan)
                nov_p_list.append(np.nan)
                rep_list.append(np.nan)
                continue

            if len(history_fps) == 0:
                nov_h = 1.0
            else:
                distances = [tanimoto_distance(fp, hfp) for hfp in history_fps]
                k = min(k_neighbors, len(distances))
                distances.sort()
                nov_h = np.mean(distances[:k])

            if len(pareto_fps) == 0:
                nov_p = 1.0
            else:
                distances = [tanimoto_distance(fp, pfp) for pfp in pareto_fps]
                nov_p = min(distances) if distances else 1.0

            if len(history_fps) == 0:
                rep = 0.0
            else:
                similarities = []
                for hfp in history_fps:
                    try:
                        sim = DataStructs.FingerprintSimilarity(fp, hfp)
                        similarities.append(sim)
                    except:
                        similarities.append(0.0)

                distances = [tanimoto_distance(fp, hfp) for hfp in history_fps]
                k = min(k_neighbors, len(distances))
                sorted_indices = np.argsort(distances)[:k]

                tau_rep = 0.5
                count_similar = sum(1 for idx in sorted_indices if similarities[idx] >= tau_rep)
                rep = count_similar / k if k > 0 else 0.0

            nov_h_list.append(nov_h)
            nov_p_list.append(nov_p)
            rep_list.append(rep)

            history_fps.append(fp)
            if i % 10 == 0 and fp is not None:
                pareto_fps.append(fp)

        history_novelty_evolution[h_max] = nov_h_list
        pareto_novelty_evolution[h_max] = nov_p_list
        repetition_evolution[h_max] = rep_list

    print("\nCalculating revisit rate (full history library)...")
    tau_revisit = 0.85

    for i, fp in enumerate(all_fps):
        if fp is None:
            revisit_rate_list.append(np.nan)
            continue

        if len(full_history_fps) == 0:
            revisit_rate_list.append(0.0)
        else:
            max_similarity = 0.0
            for hfp in full_history_fps:
                try:
                    sim = DataStructs.FingerprintSimilarity(fp, hfp)
                    if sim > max_similarity:
                        max_similarity = sim
                except:
                    pass
            revisit_rate_list.append(max_similarity)

        full_history_fps.append(fp)

    # Figure 1: History novelty vs molecule count
    ax1 = axes[0, 0]
    x = np.arange(len(molecules))
    for h_max in h_max_values:
        ax1.plot(x, history_novelty_evolution[h_max],
                 label=f'H_max={h_max}', alpha=0.7)
    ax1.set_xlabel('Molecule Index')
    ax1.set_ylabel('Nov_H (History Novelty)')
    ax1.set_title('History Novelty vs Search Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(molecules)])

    # Figure 2: Late-stage stability analysis (enlarged last 20%)
    ax2 = axes[0, 1]
    start_idx = int(len(molecules) * 0.8)
    for h_max in h_max_values:
        ax2.plot(x[start_idx:], history_novelty_evolution[h_max][start_idx:],
                 label=f'H_max={h_max}', alpha=0.7)
    ax2.set_xlabel('Molecule Index (last 20%)')
    ax2.set_ylabel('Nov_H')
    ax2.set_title('Late-Stage Stability Analysis (Last 20%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Figure 3: Pareto novelty
    ax3 = axes[1, 0]
    for h_max in h_max_values:
        ax3.plot(x, pareto_novelty_evolution[h_max],
                 label=f'H_max={h_max}', alpha=0.7)
    ax3.set_xlabel('Molecule Index')
    ax3.set_ylabel('Nov_P (Pareto Novelty)')
    ax3.set_title('Pareto Novelty vs Search Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Figure 4: Repetition penalty
    ax4 = axes[1, 1]
    for h_max in h_max_values:
        ax4.plot(x, repetition_evolution[h_max],
                 label=f'H_max={h_max}', alpha=0.7)
    ax4.set_xlabel('Molecule Index')
    ax4.set_ylabel('Rep (Repetition Penalty)')
    ax4.set_title('Repetition Penalty vs Search Progress')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Figure 5: Revisit rate (full history library)
    ax5 = axes[2, 0]
    revisit_rate = np.array(revisit_rate_list)
    ax5.plot(x, revisit_rate, color='#e74c3c', alpha=0.7, linewidth=1)
    ax5.axhline(y=tau_revisit, color='black', linestyle='--', label=f'Threshold (tau={tau_revisit})')
    ax5.set_xlabel('Molecule Index')
    ax5.set_ylabel('Max Similarity to History')
    ax5.set_title('Revisit Rate (Full History, No H_max Limit)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, len(molecules)])

    # Figure 6: Revisit rate last 20% analysis
    ax6 = axes[2, 1]
    ax6.plot(x[start_idx:], revisit_rate[start_idx:], color='#e74c3c', alpha=0.7, linewidth=1)
    ax6.axhline(y=tau_revisit, color='black', linestyle='--', label=f'Threshold (tau={tau_revisit})')

    # Calculate statistics for last 20%
    late_revisit = revisit_rate[start_idx:]
    late_revisit_valid = late_revisit[~np.isnan(late_revisit)]
    mean_revisit = np.nanmean(late_revisit)
    above_threshold = np.sum(late_revisit >= tau_revisit) / len(late_revisit_valid) * 100 if len(late_revisit_valid) > 0 else 0

    ax6.set_xlabel('Molecule Index (last 20%)')
    ax6.set_ylabel('Max Similarity to History')
    ax6.set_title(f'Late-Stage Revisit Rate (Mean={mean_revisit:.3f}, Above tau: {above_threshold:.1f}%)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'h_max_sensitivity_analysis.png'), dpi=150)
    plt.close()

    print(f"\nChart saved to: {output_dir}/h_max_sensitivity_analysis.png")

    report = []
    report.append("=" * 60)
    report.append("H_max Sensitivity Analysis Report")
    report.append("=" * 60)
    report.append(f"\nNumber of molecules analyzed: {len(molecules)}")
    report.append(f"Tested H_max values: {h_max_values}")
    report.append(f"k_neighbors: {k_neighbors}")
    report.append("\n" + "-" * 60)
    report.append("Statistical properties for each H_max value:")
    report.append("-" * 60)

    stability_metrics = {}

    for h_max in h_max_values:
        nov_h = np.array(history_novelty_evolution[h_max])
        valid_idx = ~np.isnan(nov_h)

        late_start = int(len(nov_h) * 0.8)
        late_nov_h = nov_h[valid_idx][late_start:] if valid_idx.sum() > late_start else nov_h[valid_idx]

        late_std = np.std(late_nov_h) if len(late_nov_h) > 0 else np.nan
        late_mean = np.mean(late_nov_h) if len(late_nov_h) > 0 else np.nan

        cv = late_std / late_mean if late_mean > 0 else np.nan

        stability_metrics[h_max] = {
            'late_mean': late_mean,
            'late_std': late_std,
            'cv': cv,
            'full_mean': np.nanmean(nov_h),
            'full_std': np.nanstd(nov_h)
        }

        report.append(f"\nH_max = {h_max}:")
        report.append(f"  Overall mean: {np.nanmean(nov_h):.4f}")
        report.append(f"  Overall std: {np.nanstd(nov_h):.4f}")
        report.append(f"  Last 20% mean: {late_mean:.4f}")
        report.append(f"  Last 20% std: {late_std:.4f}")
        report.append(f"  Last 20% CV: {cv:.4f}")

    report.append("\n" + "=" * 60)
    report.append("Revisit Rate Analysis (Full History Library, Not Limited by H_max)")
    report.append("=" * 60)
    report.append(f"\nSimilarity threshold tau = {tau_revisit}")

    revisit_rate = np.array(revisit_rate_list)
    full_revisit = revisit_rate[~np.isnan(revisit_rate)]
    late_start = int(len(revisit_rate) * 0.8)
    late_revisit = revisit_rate[late_start:][~np.isnan(revisit_rate[late_start:])]

    report.append(f"\nOverall revisit statistics:")
    report.append(f"  Mean max similarity: {np.mean(full_revisit):.4f}")
    report.append(f"  Std: {np.std(full_revisit):.4f}")
    report.append(f"  Percentage above threshold (tau={tau_revisit}): {np.sum(full_revisit >= tau_revisit) / len(full_revisit) * 100:.2f}%")

    report.append(f"\nLast 20% revisit statistics:")
    report.append(f"  Mean max similarity: {np.mean(late_revisit):.4f}")
    report.append(f"  Std: {np.std(late_revisit):.4f}")
    report.append(f"  Percentage above threshold (tau={tau_revisit}): {np.sum(late_revisit >= tau_revisit) / len(late_revisit) * 100:.2f}%")

    valid_cvs = {h: m['cv'] for h, m in stability_metrics.items() if not np.isnan(m['cv'])}
    if valid_cvs:
        best_h_max = min(valid_cvs, key=valid_cvs.get)
        report.append("\n" + "=" * 60)
        report.append(f"Recommended H_max = {best_h_max}")
        report.append("=" * 60)

    report_path = os.path.join(output_dir, 'h_max_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"\nReport saved to: {report_path}")

    return stability_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='H_max sensitivity analysis')
    parser.add_argument('--input', '-i', required=True,
                        help='Input file path')
    parser.add_argument('--output', '-o', default='./analysis_results',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input, 'r') as f:
        molecules = [line.strip() for line in f if line.strip()]

    print(f"Read {len(molecules)} molecules")

    metrics = analyze_h_max_sensitivity(molecules, args.output)

    return metrics


if __name__ == '__main__':
    main()
