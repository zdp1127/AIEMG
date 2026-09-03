
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import os
import re

SOTA_DIR = "sota_tu"  
OUTPUT_DIR = "sota_tu" 
SAMPLE_SIZE = 384  
DPI = 600 


INTEGER_PROPERTIES = [
    'Number of HBD', 'Number of HBA', 'N. Carbon atoms', 'N. heavy atoms',
    'N. Hydrogen atoms', 'N. Nitrogen atoms', 'N. Oxygen atoms',
    'N. Fluorine atoms', 'N. Sulphur atoms', 'N. Chlorine atoms',
    'Pentamers', 'Hexamers', 'Heptamers', 'Aromatic cycles'
]

CONTINUOUS_PROPERTIES = ['MW (Da)', 'clogP']

ALL_PROPERTIES = INTEGER_PROPERTIES + CONTINUOUS_PROPERTIES

def get_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        props = {}
        
        # A. Number of HBD 
        props['Number of HBD'] = Lipinski.NumHDonors(mol)
        
        # B. Number of HBA 
        props['Number of HBA'] = Lipinski.NumHAcceptors(mol)
        
        # C. MW (Da)
        props['MW (Da)'] = Descriptors.MolWt(mol)
        
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # D. N. Carbon atoms 
        props['N. Carbon atoms'] = atoms.count('C')
        
        # E. N. heavy atoms 
        props['N. heavy atoms'] = mol.GetNumHeavyAtoms()
        
        # F. N. Hydrogen atoms 
        props['N. Hydrogen atoms'] = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        
        # G. N. Nitrogen atoms 
        props['N. Nitrogen atoms'] = atoms.count('N')
        
        # H. N. Oxygen atoms
        props['N. Oxygen atoms'] = atoms.count('O')
        
        # I. N. Fluorine atoms 
        props['N. Fluorine atoms'] = atoms.count('F')
        
        # K. N. Sulphur atoms 
        props['N. Sulphur atoms'] = atoms.count('S')
        
        # L. N. Chlorine atoms
        props['N. Chlorine atoms'] = atoms.count('Cl')
        
        # 环信息
        ri = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ri.AtomRings()]
        
        # M. Pentamers
        props['Pentamers'] = ring_sizes.count(5)
        
        # N. Hexamers 
        props['Hexamers'] = ring_sizes.count(6)
        
        # O. Heptamers
        props['Heptamers'] = ring_sizes.count(7)
        
        # P. Aromatic cycles 
        props['Aromatic cycles'] = Lipinski.NumAromaticRings(mol)
        
        # Q. clogP 
        props['clogP'] = Crippen.MolLogP(mol)
        
        return props
    except Exception as e:
        return None


def process_dataset(filepath, label, sample_size=384):
  
    print(f"Processing {label} from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return []
    
    # 读取CSV文件
    try:
        df_csv = pd.read_csv(filepath)
        if 'SMILES' not in df_csv.columns:
            print(f"Error: 'SMILES' column not found in {filepath}")
            return []
        
        smiles_list = df_csv['SMILES'].dropna().tolist()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []
    
    total_count = len(smiles_list)
    print(f"  Found {total_count} molecules.")
    
    # 取前 sample_size 个分子
    if total_count > sample_size:
        sampled_smiles = smiles_list[:sample_size]
        print(f"  Taking first {sample_size} molecules.")
    else:
        sampled_smiles = smiles_list
        print(f"  Taking all {total_count} molecules.")
    
    data = []
    for smi in sampled_smiles:
        props = get_properties(smi)
        if props:
            props['Dataset'] = label
            data.append(props)
    
    print(f"  Successfully calculated properties for {len(data)} molecules.")
    return data


def safe_filename(text):
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "plot"


def strip_panel_prefix(text):
  
    return re.sub(r"^\([A-Za-z]\)\s*", "", text.strip())



def main():
    sns.set(style="ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    
  
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
 
    if not os.path.isdir(SOTA_DIR):
        print(f"Error: Directory '{SOTA_DIR}' not found.")
        return
    
    csv_files = [f for f in os.listdir(SOTA_DIR) if f.lower().endswith('.csv')]
    csv_files.sort()
    
    if not csv_files:
        print(f"Error: No CSV files found in '{SOTA_DIR}'.")
        return
    
    print(f"Found {len(csv_files)} CSV files.")

    all_data = []
    files = []
    
    for filename in csv_files:
        filepath = os.path.join(SOTA_DIR, filename)
        label = os.path.splitext(filename)[0]
        files.append((filepath, label))
        
        data = process_dataset(filepath, label, SAMPLE_SIZE)
        all_data.extend(data)
    
    if not all_data:
        print("No data collected. Exiting.")
        return
    
    df = pd.DataFrame(all_data)
    print(f"\nTotal molecules processed: {len(df)}")
    
   
    data_output_path = os.path.join(OUTPUT_DIR, "comparison_data.csv")
    df.to_csv(data_output_path, index=False)
    print(f"Data saved to {data_output_path}")
    

    dataset_order = [label for _, label in files]
    n_colors = max(len(dataset_order), 3)
    palette_colors = sns.color_palette("Set3", n_colors=n_colors)
    custom_palette = {label: palette_colors[i % len(palette_colors)] for i, label in enumerate(dataset_order)}
    
   
    titles = [
        '(A) Number of HBD', '(B) Number of HBA', '(C) N. Carbon atoms', '(D) N. heavy atoms',
        '(E) N. Hydrogen atoms', '(F) N. Nitrogen atoms', '(G) N. Oxygen atoms', '(H) N. Fluorine atoms',
        '(I) N. Sulphur atoms', '(K) N. Chlorine atoms', '(L) Pentamers', '(M) Hexamers',
        '(N) Heptamers', '(O) Aromatic cycles', '(P) MW (Da)', '(Q) clogP'
    ]
    
    # ============== 绘制组合图 (4x4) ==============
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    
    for i, prop in enumerate(ALL_PROPERTIES):
        if i >= len(axes):
            break
        
        ax = axes[i]
        
        if prop in df.columns:
            is_integer = prop in INTEGER_PROPERTIES
            
            if is_integer:
               
                sns.boxplot(x='Dataset', y=prop, data=df, ax=ax, order=dataset_order, 
                           palette=custom_palette, width=0.6, 
                           boxprops={'zorder': 1}, 
                           showfliers=False)
                sns.stripplot(x='Dataset', y=prop, data=df, ax=ax, order=dataset_order, 
                             palette=custom_palette, alpha=0.5, size=2.5, 
                             jitter=0.35, zorder=2)
            else:
               
                sns.violinplot(x='Dataset', y=prop, data=df, ax=ax, order=dataset_order, 
                               palette=custom_palette, inner='box')
            
            ax.set_title(strip_panel_prefix(titles[i]), fontsize=14)
            ax.set_xlabel('')
            ax.set_ylabel(prop)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "comparison_violin_sota_tu_top384.png")
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\nMain plot saved to {output_path}")
    plt.close()
    

    solo_dir = os.path.join(OUTPUT_DIR, "violin_solo_top384")
    os.makedirs(solo_dir, exist_ok=True)
    
    for i, prop in enumerate(ALL_PROPERTIES):
        solo_fig, solo_ax = plt.subplots(1, 1, figsize=(14, 8))
        
        if prop in df.columns:
            is_integer = prop in INTEGER_PROPERTIES
            
            if is_integer:
                sns.boxplot(x='Dataset', y=prop, data=df, ax=solo_ax, order=dataset_order, 
                           palette=custom_palette, width=0.6,
                           boxprops={'zorder': 1},
                           showfliers=False)
                sns.stripplot(x='Dataset', y=prop, data=df, ax=solo_ax, order=dataset_order, 
                             palette=custom_palette, alpha=0.5, size=4, 
                             jitter=0.35, zorder=2)
            else:
                sns.violinplot(x='Dataset', y=prop, data=df, ax=solo_ax, order=dataset_order, 
                               palette=custom_palette, inner='box')
            
            solo_ax.set_title(strip_panel_prefix(titles[i]), fontsize=16)
            solo_ax.set_xlabel('')
            solo_ax.set_ylabel(prop)
            solo_ax.tick_params(axis='x', rotation=45)
            solo_fig.tight_layout()
            
          
            title_prefix = f"{i+1:02d}"
            output_solo_path = os.path.join(solo_dir, f"{title_prefix}_{safe_filename(prop)}.png")
            solo_fig.savefig(output_solo_path, dpi=DPI, bbox_inches='tight')
            plt.close(solo_fig)
    
    print(f"Solo plots saved to {solo_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
