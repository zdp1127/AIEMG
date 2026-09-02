#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calculate molecular properties: QED, SA score, MW, internal diversity, and scaffold diversity.

Input file: template_for_data/present/ligands.txt

Usage:
  python data/calc_props.py

Output:
  - template_for_data/present/ligands_props.csv: Properties for each molecule
  - Console output: Internal diversity and scaffold diversity statistics
"""

import os
import csv
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger
from rdkit.RDLogger import logger
from rdkit import DataStructs
from rdkit.Chem import QED, Descriptors, AllChem, rdMolDescriptors
import rdBase

# Try to import sascorer (use approximate calculation if unavailable)
try:
    from rdkit.Chem import rdMolDescriptors
    import sascorer
    HAS_SASCORER = True
except ImportError:
    HAS_SASCORER = False
    print("[WARN] sascorer not available, using approximate SA score")


# =============== Configuration =====================

# Get the project root directory (parent of 'data' folder)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

# Input file path (relative to project root)
INPUT_FILE = os.path.join(_PROJECT_ROOT, "template_for_data", "present", "ligands.txt")

# Output file path (relative to project root)
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, "template_for_data", "present", "ligands_props.csv")

# Morgan fingerprint parameters (unified as radius=2, nBits=2048)
FINGERPRINT_RADIUS = 2
FINGERPRINT_BITS = 2048

# =========================================================


def calc_sa_score(mol: Chem.Mol) -> float:
    """
    Calculate Synthetic Accessibility Score (SA Score).
    Uses rdkit sascorer module, range ~1 (easy to synthesize) to 10 (hard to synthesize).
    """
    if mol is None:
        return float('nan')
    
    if HAS_SASCORER:
        try:
            return sascorer.calculateScore(mol)
        except Exception:
            pass
    
    # Fallback approximate calculation
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    sp3_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    mw = Descriptors.MolWt(mol)
    
    complexity = (
        0.3 * ring_count +
        0.4 * sp3_centers +
        0.2 * rot_bonds +
        0.1 * (hbd + hba) +
        0.005 * tpsa +
        0.003 * max(mw - 200.0, 0.0)
    )
    sa = 1.0 + complexity
    return max(1.0, min(10.0, sa))


def get_scaffolds(mol: Chem.Mol) -> List[str]:
    """
    Extract molecular scaffolds (Murcko Scaffold).
    Returns scaffolds in SMARTS format.
    """
    if mol is None:
        return []
    
    try:
        core = Chem.MolToSmarts(Chem.MurckoDecompose(mol))
        return [core] if core else []
    except Exception:
        return []


def calc_internal_diversity(mols: List[Chem.Mol]) -> float:
    """
    Calculate Internal Diversity.
    Uses Tanimoto distance of Morgan fingerprints.
    Formula: 1 - (average similarity of all fingerprint pairs)
    """
    if len(mols) < 2:
        return 0.0
    
    fps = []
    for mol in mols:
        try:
            mol_with_h = Chem.AddHs(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol_with_h, 
                radius=FINGERPRINT_RADIUS, 
                nBits=FINGERPRINT_BITS
            )
            fps.append(fp)
        except Exception:
            continue
    
    if len(fps) < 2:
        return 0.0
    
    n = len(fps)
    total_similarity = 0.0
    count = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_similarity += sim
            count += 1
    
    if count == 0:
        return 0.0
    
    avg_similarity = total_similarity / count
    diversity = 1 - avg_similarity
    
    return round(diversity, 4)


def calc_scaffold_diversity(mols: List[Chem.Mol]) -> float:
    """
    Calculate Scaffold Diversity.
    Uses Murcko Scaffold unique ratio.
    Formula: unique_scaffolds / total_molecules
    """
    if len(mols) == 0:
        return 0.0
    
    all_scaffolds = []
    for mol in mols:
        scaffolds = get_scaffolds(mol)
        all_scaffolds.extend(scaffolds)
    
    if len(all_scaffolds) == 0:
        return 0.0
    
    # Count unique scaffolds
    unique_scaffolds = len(set(all_scaffolds))
    total = len(mols)
    
    diversity = unique_scaffolds / total if total > 0 else 0.0
    
    return round(diversity, 4)


def calc_properties_for_smiles(smiles: str, idx: int) -> Dict:
    """
    Calculate properties for a single molecule.
    """
    result = {
        'index': idx,
        'smiles': smiles,
        'valid': False,
        'canonical_smiles': None,
        'QED': None,
        'SA_score': None,
        'MW': None,
        'LogP': None,
        'NumHBA': None,
        'NumHBD': None,
        'NumRotatableBonds': None,
        'NumRings': None,
        'TPSA': None,
    }
    
    # Parse SMILES
    try:
        rdBase.DisableLog('rdApp.*')
        mol = Chem.MolFromSmiles(smiles)
    finally:
        rdBase.EnableLog('rdApp.*')
    
    if mol is None:
        return result
    
    result['valid'] = True
    
    # Canonicalize SMILES
    try:
        result['canonical_smiles'] = Chem.MolToSmiles(mol)
    except Exception:
        pass
    
    # QED (drug-likeness)
    try:
        result['QED'] = round(QED.default(mol), 4)
    except Exception:
        pass
    
    # SA Score (synthetic accessibility)
    try:
        result['SA_score'] = round(calc_sa_score(mol), 4)
    except Exception:
        pass
    
    # Molecular Weight (MW)
    try:
        result['MW'] = round(Descriptors.MolWt(mol), 2)
    except Exception:
        pass
    
    # LogP (lipophilicity)
    try:
        from rdkit.Chem import Crippen
        result['LogP'] = round(float(Crippen.MolLogP(mol)), 4)
    except Exception:
        pass
    
    # HBA (hydrogen bond acceptors)
    try:
        result['NumHBA'] = rdMolDescriptors.CalcNumHBA(mol)
    except Exception:
        pass
    
    # HBD (hydrogen bond donors)
    try:
        result['NumHBD'] = rdMolDescriptors.CalcNumHBD(mol)
    except Exception:
        pass
    
    # Rotatable bonds
    try:
        result['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    except Exception:
        pass
    
    # Number of rings
    try:
        result['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
    except Exception:
        pass
    
    # TPSA (topological polar surface area)
    try:
        result['TPSA'] = round(rdMolDescriptors.CalcTPSA(mol), 2)
    except Exception:
        pass
    
    return result


def read_smiles_from_file(filepath: str) -> List[str]:
    """
    Read SMILES from file.
    """
    smiles_list = []
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return smiles_list
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Take first whitespace-separated field
            parts = line.split()
            if parts:
                smiles_list.append(parts[0])
    
    return smiles_list


def write_results_to_csv(filepath: str, rows: List[Dict]) -> None:
    """
    Write results to CSV file.
    """
    if not rows:
        print("[WARN] No results to write")
        return
    
    fieldnames = [
        'index', 'smiles', 'valid', 'canonical_smiles',
        'QED', 'SA_score', 'MW', 'LogP',
        'NumHBA', 'NumHBD', 'NumRotatableBonds', 'NumRings', 'TPSA'
    ]
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[INFO] Results written to: {filepath}")


def main():
    print("=" * 60)
    print("Molecular Properties Calculator")
    print("=" * 60)
    
    # Read SMILES
    print(f"\n[INFO] Reading from: {INPUT_FILE}")
    smiles_list = read_smiles_from_file(INPUT_FILE)
    print(f"[INFO] Read {len(smiles_list)} SMILES")
    
    if len(smiles_list) == 0:
        print("[ERROR] No valid SMILES found")
        return
    
    # Calculate properties for each molecule
    print("\n[INFO] Calculating properties...")
    results = []
    valid_mols = []
    
    for i, smiles in enumerate(smiles_list):
        result = calc_properties_for_smiles(smiles, i)
        results.append(result)
        
        if result['valid']:
            try:
                rdBase.DisableLog('rdApp.*')
                mol = Chem.MolFromSmiles(result['canonical_smiles'])
                if mol:
                    valid_mols.append(mol)
            finally:
                rdBase.EnableLog('rdApp.*')
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} / {len(smiles_list)}")
    
    print(f"  Processed {len(smiles_list)} / {len(smiles_list)}")
    print(f"[INFO] Valid molecules: {len(valid_mols)} / {len(smiles_list)}")
    
    # Calculate internal diversity and scaffold diversity
    print("\n[INFO] Calculating diversity metrics...")
    
    internal_div = calc_internal_diversity(valid_mols)
    scaffold_div = calc_scaffold_diversity(valid_mols)
    
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"Total molecules: {len(smiles_list)}")
    print(f"Valid molecules: {len(valid_mols)}")
    print(f"Internal Diversity: {internal_div:.4f}")
    print(f"Scaffold Diversity: {scaffold_div:.4f}")
    
    # Calculate QED and SA score statistics for valid molecules
    valid_qeds = [r['QED'] for r in results if r['valid'] and r['QED'] is not None]
    valid_sa = [r['SA_score'] for r in results if r['valid'] and r['SA_score'] is not None]
    valid_mw = [r['MW'] for r in results if r['valid'] and r['MW'] is not None]
    
    if valid_qeds:
        print(f"\nQED Statistics (valid molecules):")
        print(f"  Mean: {np.mean(valid_qeds):.4f}")
        print(f"  Std:  {np.std(valid_qeds):.4f}")
        print(f"  Min:  {np.min(valid_qeds):.4f}")
        print(f"  Max:  {np.max(valid_qeds):.4f}")
    
    if valid_sa:
        print(f"\nSA Score Statistics (valid molecules):")
        print(f"  Mean: {np.mean(valid_sa):.4f}")
        print(f"  Std:  {np.std(valid_sa):.4f}")
        print(f"  Min:  {np.min(valid_sa):.4f}")
        print(f"  Max:  {np.max(valid_sa):.4f}")
    
    if valid_mw:
        print(f"\nMolecular Weight (MW) Statistics (valid molecules):")
        print(f"  Mean: {np.mean(valid_mw):.2f}")
        print(f"  Std:  {np.std(valid_mw):.2f}")
        print(f"  Min:  {np.min(valid_mw):.2f}")
        print(f"  Max:  {np.max(valid_mw):.2f}")
    
    # Write CSV
    print("\n[INFO] Writing results to CSV...")
    write_results_to_csv(OUTPUT_FILE, results)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
