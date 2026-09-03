#!/usr/bin/env python3
"""
Visualization for MCTS Molecular Design Experiments:
1. Hypervolume curve over time
2. Search Trajectory 2D visualization in chemical space
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import os
import random

# Set global random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Set font to Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'normal'  # Default font weight not bold

# Data path (dynamically resolved from script location)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.dirname(_SCRIPT_DIR)  # template_for_data directory
DATA_DIR = os.path.join(_DATA_DIR, "present")

# Output path for allLigands.txt (dynamically resolved)
OUTPUT_DIR = os.path.join(_DATA_DIR, "output")

# Data directory for reference molecules (dynamically resolved)
DATA_ROOT = _DATA_DIR  # template_for_data directory

def load_metrics():
    """Load metrics data"""
    timestamps = []
    num_molecules = []
    hypervolumes = []
    elapsed_seconds = []
    molecules_per_hour = []

    with open(os.path.join(DATA_DIR, "metrics.csv"), "r") as f:
        for line in f:
            parts = line.strip().rstrip(",").split(",")
            if len(parts) >= 7:
                try:
                    # Format: timestamp, num_mol, hypervolume, elapsed_sec, mol_per_hour, ...
                    timestamps.append(parts[0])
                    num_molecules.append(int(parts[1]))
                    hypervolumes.append(float(parts[2]))
                    elapsed_seconds.append(float(parts[3]))
                    molecules_per_hour.append(float(parts[4]))
                except (ValueError, IndexError):
                    continue

    return np.array(num_molecules), np.array(hypervolumes), np.array(elapsed_seconds), np.array(molecules_per_hour)

def load_scores():
    """Load score data"""
    scores = []
    with open(os.path.join(DATA_DIR, "scores.txt"), "r") as f:
        for line in f:
            parts = line.strip().strip("[]").split(",")
            if len(parts) >= 5:
                try:
                    scores.append([float(x.strip()) for x in parts[:5]])
                except:
                    continue
    return np.array(scores)

def load_pareto_front():
    """Load Pareto front"""
    with open(os.path.join(DATA_DIR, "pareto.json"), "r") as f:
        data = json.load(f)
    return np.array(data.get("front", []))

def load_depth():
    """Load MCTS tree depth information"""
    depths = []
    with open(os.path.join(DATA_DIR, "depth.txt"), "r") as f:
        for line in f:
            try:
                depths.append(int(line.strip()))
            except:
                continue
    return np.array(depths)

def load_mcts_molecules(sample_size=5000):
    """Load MCTS-generated molecules (sampled)"""
    mcts_file = os.path.join(OUTPUT_DIR, "allLigands.txt")
    mols = []
    with open(mcts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: SMILES1$|SMILES2$, take the first SMILES
            parts = line.split('|')
            if parts:
                smiles = parts[0].strip().rstrip('$')
                if smiles and len(smiles) > 3:
                    mols.append(smiles)
    # Random sampling
    if len(mols) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(mols), sample_size, replace=False)
        mols = [mols[i] for i in sorted(indices)]
    return mols

def load_zinc_molecules(sample_size=5000):
    """Load ZINC drug library molecules (sampled)"""
    zinc_file = os.path.join(DATA_ROOT, "..", "data", "250k_rndm_zinc_drugs_clean.smi")
    mols = []
    with open(zinc_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                smiles = parts[0]
                if smiles and len(smiles) > 3:
                    mols.append(smiles)
    # Random sampling
    if len(mols) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(mols), sample_size, replace=False)
        mols = [mols[i] for i in sorted(indices)]
    return mols

def compute_fingerprints(smiles_list):
    """Compute Morgan fingerprints"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    
    fps = []
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
                valid_smiles.append(smiles)
        except:
            pass
    return fps, valid_smiles

def create_visualization():
    """Create complete visualization"""
    # Load data
    print("Loading data...")
    num_molecules, hypervolumes, elapsed, mph = load_metrics()
    scores = load_scores()
    pareto_front = load_pareto_front()
    depths = load_depth()

    print(f"Loaded {len(num_molecules)} checkpoints, {len(scores)} molecules")

    # Create figure - two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================== Left: Chemical space distribution comparison ====================
    ax1 = axes[0]
    
    print("Loading molecules for chemical space visualization...")
    mcts_mols = load_mcts_molecules()
    zinc_mols = load_zinc_molecules(sample_size=5000)
    print(f"Loaded {len(mcts_mols)} MCTS molecules, {len(zinc_mols)} ZINC molecules")
    
    # Compute fingerprints
    mcts_fps, mcts_valid = compute_fingerprints(mcts_mols)
    zinc_fps, zinc_valid = compute_fingerprints(zinc_mols)
    print(f"Valid fingerprints: {len(mcts_fps)} MCTS, {len(zinc_fps)} ZINC")
    
    # Combine fingerprints for dimensionality reduction
    all_fps = mcts_fps + zinc_fps
    from rdkit import DataStructs
    from scipy.sparse import csr_matrix
    
    # Convert to numpy array
    n_bits = 2048
    fp_matrix = np.zeros((len(all_fps), n_bits), dtype=np.int8)
    for i, fp in enumerate(all_fps):
        arr = np.zeros(n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_matrix[i, :] = arr
    
    # t-SNE dimensionality reduction
    from sklearn.manifold import TSNE
    perplexity = min(30, len(all_fps) // 4)
    coords_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                     learning_rate='auto', init='pca').fit_transform(fp_matrix)
    
    # Split coordinates
    mcts_coords = coords_2d[:len(mcts_fps)]
    zinc_coords = coords_2d[len(mcts_fps):]
    
    # Plot ZINC background
    ax1.scatter(zinc_coords[:, 0], zinc_coords[:, 1], 
                c='gray', s=8, alpha=0.35, label='ZINC250k')
    
    # Plot MCTS molecules
    ax1.scatter(mcts_coords[:, 0], mcts_coords[:, 1],
                c='steelblue', s=22, alpha=0.56, label='AIEMG Generated')
    
    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)
    ax1.set_title('Chemical Space Distribution', fontsize=14)
    
    # Custom legend (using fixed color squares)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.6, label='ZINC250k'),
        Patch(facecolor='steelblue', label='AIEMG Generated')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.2)

    # ==================== Right: Search Trajectory 2D visualization ====================
    ax2 = axes[1]

    # Use UMAP or PCA for dimensionality reduction (use PCA if UMAP not available)
    try:
        from sklearn.manifold import TSNE
        # Reduce scores to 2D for visualization
        if len(scores) > 50:
            perplexity = min(30, len(scores) // 4)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                          learning_rate='auto', init='pca')
            coords_2d = reducer.fit_transform(scores)
        else:
            coords_2d = scores[:, :2]  # Use first two dimensions directly
        use_tsne = True
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(scores)
        use_tsne = False

    print(f"Dimensionality reduction: {'t-SNE' if use_tsne else 'PCA'}")

    # Calculate color mapping (based on generation order/time)
    n_molecules = len(coords_2d)
    color_values = np.linspace(0, 1, n_molecules)

    # Create custom colormap (dark blue to bright blue/cyan)
    colors_blues = ['#1a1a2e', '#16213e', '#0f3460', '#3498db', '#00d4ff', '#7fffe0']
    cmap = LinearSegmentedColormap.from_list('search_trajectory', colors_blues)

    # Plot scatter
    scatter = ax2.scatter(coords_2d[:, 0], coords_2d[:, 1],
                         c=color_values, cmap=cmap, s=30, alpha=0.8)

    # Plot trajectory lines (connected by generation order)
    # Sample to avoid too many lines
    step = max(1, n_molecules // 200)
    sampled_indices = list(range(0, n_molecules, step))
    if sampled_indices[-1] != n_molecules - 1:
        sampled_indices.append(n_molecules - 1)

    ax2.plot(coords_2d[sampled_indices, 0], coords_2d[sampled_indices, 1],
            'w-', alpha=0.2, linewidth=0.5)

    # Mark Pareto front molecules
    if len(pareto_front) > 0:
        # Find molecules corresponding to Pareto front
        pareto_size = min(len(pareto_front), 50)
        for i in range(pareto_size):
            ax2.scatter(coords_2d[i, 0], coords_2d[i, 1],
                       s=100, c='gold', edgecolors='red', linewidths=1.5,
                       marker='*', zorder=10)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Search Progress (Early -> Late)', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Early', 'Middle', 'Late'])

    ax2.set_xlabel('Dimension 1', fontsize=12)
    ax2.set_ylabel('Dimension 2', fontsize=12)
    ax2.set_title('Search Trajectory in Chemical Space', fontsize=14)
    ax2.grid(True, alpha=0.2)

    # Add legend description
    ax2.scatter([], [], s=100, c='gold', edgecolors='red', marker='*',
               label=f'Pareto Front ({len(pareto_front)} molecules)')
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(DATA_DIR, "..", "search_trajectory_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")

    # Also save PDF version
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"Saved PDF to: {pdf_path}")

    plt.show()
    return fig

def create_detailed_trajectory_plot():
    """Create more detailed trajectory visualization with exploration vs exploitation distinction"""
    print("\nCreating detailed trajectory plot...")

    # Load data
    num_molecules, hypervolumes, elapsed, mph = load_metrics()
    scores = load_scores()
    depths = load_depth()

    # Create only the bottom half: 1 row 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ==================== Left: MCTS depth distribution ====================
    ax = axes[0]

    if len(depths) > 0:
        ax.hist(depths, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        ax.axvline(np.mean(depths), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(depths):.1f}')
        ax.axvline(np.median(depths), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(depths):.1f}')

    ax.set_xlabel('MCTS Tree Depth', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('MCTS Exploration Depth Distribution', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ==================== Right: Generation rate ====================
    ax = axes[1]

    elapsed_hours = elapsed / 3600.0
    ax.plot(elapsed_hours, mph, 'g-', linewidth=2)
    ax.fill_between(elapsed_hours, mph, alpha=0.2, color='green')

    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Molecules per Hour', fontsize=11)
    ax.set_title('Generation Rate Over Time', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = os.path.join(DATA_DIR, "..", "detailed_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved detailed analysis to: {output_path}")

    plt.show()
    return fig

if __name__ == "__main__":
    print("=" * 60)
    print("MCTS Molecular Design - Search Trajectory Visualization")
    print("=" * 60)

    # Create main visualization
    create_visualization()

    # Create detailed analysis plot
    create_detailed_trajectory_plot()

    print("\nVisualization complete!")
