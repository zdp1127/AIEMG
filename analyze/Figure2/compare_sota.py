"""
Molecular Property Distribution Comparison - SOTA Models on ZINC250
生成图片: comparison_violin_sota_tu_top384.png

使用方法:
1. 在当前目录下创建 sota_tu 文件夹
2. 放入9个CSV文件: AIEMG.csv, ChemTS.csv, Graph_MCTS.csv, MARS.csv, 
   Mothra.csv, PMMG.csv, REINVENT.csv, SMILES_GA.csv, SMILES_VAE.csv
3. 每个CSV需要包含 'SMILES' 列
4. 运行: python compare_sota_tu_top384.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import os
import re

# ============== 设置 ==============
SOTA_DIR = "sota_tu"  # CSV文件所在目录
OUTPUT_DIR = "sota_tu"  # 图片输出目录
SAMPLE_SIZE = 384  # 每个数据集采样的分子数量
DPI = 600  # 图片分辨率

# ============== 属性定义 ==============
INTEGER_PROPERTIES = [
    'Number of HBD', 'Number of HBA', 'N. Carbon atoms', 'N. heavy atoms',
    'N. Hydrogen atoms', 'N. Nitrogen atoms', 'N. Oxygen atoms',
    'N. Fluorine atoms', 'N. Sulphur atoms', 'N. Chlorine atoms',
    'Pentamers', 'Hexamers', 'Heptamers', 'Aromatic cycles'
]

CONTINUOUS_PROPERTIES = ['MW (Da)', 'clogP']

ALL_PROPERTIES = INTEGER_PROPERTIES + CONTINUOUS_PROPERTIES

# ============== 属性计算函数 ==============
def get_properties(smiles):
    """计算单个分子的理化性质"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        props = {}
        
        # A. Number of HBD (氢键供体数)
        props['Number of HBD'] = Lipinski.NumHDonors(mol)
        
        # B. Number of HBA (氢键受体数)
        props['Number of HBA'] = Lipinski.NumHAcceptors(mol)
        
        # C. MW (Da) (分子量)
        props['MW (Da)'] = Descriptors.MolWt(mol)
        
        # 统计原子
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        # D. N. Carbon atoms (碳原子数)
        props['N. Carbon atoms'] = atoms.count('C')
        
        # E. N. heavy atoms (重原子数)
        props['N. heavy atoms'] = mol.GetNumHeavyAtoms()
        
        # F. N. Hydrogen atoms (氢原子数)
        props['N. Hydrogen atoms'] = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        
        # G. N. Nitrogen atoms (氮原子数)
        props['N. Nitrogen atoms'] = atoms.count('N')
        
        # H. N. Oxygen atoms (氧原子数)
        props['N. Oxygen atoms'] = atoms.count('O')
        
        # I. N. Fluorine atoms (氟原子数)
        props['N. Fluorine atoms'] = atoms.count('F')
        
        # K. N. Sulphur atoms (硫原子数)
        props['N. Sulphur atoms'] = atoms.count('S')
        
        # L. N. Chlorine atoms (氯原子数)
        props['N. Chlorine atoms'] = atoms.count('Cl')
        
        # 环信息
        ri = mol.GetRingInfo()
        ring_sizes = [len(r) for r in ri.AtomRings()]
        
        # M. Pentamers (五元环数)
        props['Pentamers'] = ring_sizes.count(5)
        
        # N. Hexamers (六元环数)
        props['Hexamers'] = ring_sizes.count(6)
        
        # O. Heptamers (七元环数)
        props['Heptamers'] = ring_sizes.count(7)
        
        # P. Aromatic cycles (芳香环数)
        props['Aromatic cycles'] = Lipinski.NumAromaticRings(mol)
        
        # Q. clogP (脂水分配系数)
        props['clogP'] = Crippen.MolLogP(mol)
        
        return props
    except Exception as e:
        return None


def process_dataset(filepath, label, sample_size=384):
    """处理单个数据集"""
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
    
    # 计算性质
    data = []
    for smi in sampled_smiles:
        props = get_properties(smi)
        if props:
            props['Dataset'] = label
            data.append(props)
    
    print(f"  Successfully calculated properties for {len(data)} molecules.")
    return data


def safe_filename(text):
    """生成安全的文件名"""
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "plot"


def strip_panel_prefix(text):
    """去除面板标题前缀"""
    return re.sub(r"^\([A-Za-z]\)\s*", "", text.strip())


# ============== 主函数 ==============
def main():
    # 设置绘图风格
    sns.set(style="ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取CSV文件列表
    if not os.path.isdir(SOTA_DIR):
        print(f"Error: Directory '{SOTA_DIR}' not found.")
        return
    
    csv_files = [f for f in os.listdir(SOTA_DIR) if f.lower().endswith('.csv')]
    csv_files.sort()
    
    if not csv_files:
        print(f"Error: No CSV files found in '{SOTA_DIR}'.")
        return
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # 处理所有数据集
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
    
    # 保存数据
    data_output_path = os.path.join(OUTPUT_DIR, "comparison_data.csv")
    df.to_csv(data_output_path, index=False)
    print(f"Data saved to {data_output_path}")
    
    # 定义数据集顺序和颜色
    dataset_order = [label for _, label in files]
    n_colors = max(len(dataset_order), 3)
    palette_colors = sns.color_palette("Set3", n_colors=n_colors)
    custom_palette = {label: palette_colors[i % len(palette_colors)] for i, label in enumerate(dataset_order)}
    
    # 面板标题
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
                # 整数属性: 箱线图 + 散点图
                sns.boxplot(x='Dataset', y=prop, data=df, ax=ax, order=dataset_order, 
                           palette=custom_palette, width=0.6, 
                           boxprops={'zorder': 1}, 
                           showfliers=False)
                sns.stripplot(x='Dataset', y=prop, data=df, ax=ax, order=dataset_order, 
                             palette=custom_palette, alpha=0.5, size=2.5, 
                             jitter=0.35, zorder=2)
            else:
                # 连续属性: 小提琴图
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
    
    # ============== 绘制单独大图 ==============
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
            
            # 保存单独图
            title_prefix = f"{i+1:02d}"
            output_solo_path = os.path.join(solo_dir, f"{title_prefix}_{safe_filename(prop)}.png")
            solo_fig.savefig(output_solo_path, dpi=DPI, bbox_inches='tight')
            plt.close(solo_fig)
    
    print(f"Solo plots saved to {solo_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
