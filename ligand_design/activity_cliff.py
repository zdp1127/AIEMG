"""
Molecular Activity Cliff Detection and Storage Module
Based on ACARL implementation, adapted for MCTS + Pareto framework
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
import os

class ActivityCliffDetector:
    """Molecular Activity Cliff Detector"""
    
    def __init__(self, alpha1=0.5, alpha2=2.0, max_memory_size=1000):
        """
        Initialize Activity Cliff Detector
        
        Args:
            alpha1 (float): Activity difference threshold
            alpha2 (float): Activity cliff index threshold  
            max_memory_size (int): Maximum memory storage size
        """
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_memory_size = max_memory_size
        
        # Store high activity molecules
        self.high_activity_memory = pd.DataFrame(columns=["smiles", "scores", "fps"])
        
        # Store activity cliff molecule pairs
        self.cliff_memory = pd.DataFrame(columns=["smiles", "scores", "fps", "cliff_type"])
        
    def calc_fingerprints(self, smiles_list):
        """Calculate molecular fingerprints"""
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = []
        valid_smiles = []
        
        for i, mol in enumerate(mols):
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(fp)
                valid_smiles.append(smiles_list[i])
            else:
                fps.append(None)
                valid_smiles.append(None)
                
        return fps, valid_smiles
    
    def detect_activity_cliffs(self, new_smiles, new_scores):
        """
        Detect activity cliffs
        
        Args:
            new_smiles (list): New molecule SMILES list
            new_scores (list): New molecule scores list
            
        Returns:
            list: Detected cliff molecule pairs
        """
        cliff_pairs = []
        
        if len(self.high_activity_memory) == 0:
            return cliff_pairs
            
        # Calculate fingerprints for new molecules
        new_fps, valid_new_smiles = self.calc_fingerprints(new_smiles)
        
        for i, (smiles, score, fp) in enumerate(zip(valid_new_smiles, new_scores, new_fps)):
            # Check if molecule and fingerprint are valid
            if smiles is None or fp is None:
                continue
                
            # Check if score is valid (using docking score, i.e., first dimension)
            if isinstance(score, list) and len(score) > 0:
                docking_score = score[0]  # Use docking score
            else:
                docking_score = score
                
            # Only proceed if docking score is valid
            if docking_score is None or docking_score < -20:  # Reasonable docking score range
                continue
                
            # Compare with molecules in memory
            for j, row in self.high_activity_memory.iterrows():
                if row['fps'] is None:
                    continue
                    
                # Calculate activity difference (using docking score, i.e., first dimension)
                if isinstance(row['scores'], list) and len(row['scores']) > 0:
                    score1 = row['scores'][0]  # Use docking score
                else:
                    score1 = row['scores']
                    
                if isinstance(score, list) and len(score) > 0:
                    score2 = score[0]  # Use docking score
                else:
                    score2 = score
                    
                diff_abs = abs(score1 - score2)
                if diff_abs < self.alpha1:
                    continue
                    
                # Calculate structural similarity
                try:
                    similarity = DataStructs.FingerprintSimilarity(row['fps'], fp)
                    dist = 1 - similarity
                    
                    if dist <= 0:  # Identical or abnormal structure
                        continue
                        
                    # Calculate Activity Cliff Index
                    ACI = diff_abs / dist
                    
                    if ACI >= self.alpha2:
                        cliff_type = "high_to_low" if score1 > score2 else "low_to_high"
                        cliff_pairs.append({
                            'smiles1': row['smiles'],
                            'scores1': score1,  # Use extracted scalar score
                            'scores2': score2,  # Use extracted scalar score
                            'smiles2': smiles,
                            'ACI': ACI,
                            'cliff_type': cliff_type
                        })
                        
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
                    
        return cliff_pairs
    
    def get_high_activity_cliff_neighbors(self, new_smiles, new_scores, beta1=None, beta2=None):
        if beta1 is None:
            beta1 = self.alpha1
        if beta2 is None:
            beta2 = self.alpha2

        new_fps, valid_new_smiles = self.calc_fingerprints(new_smiles)
        cliff_neighbors = []

        if len(self.high_activity_memory) == 0:
            return cliff_neighbors

        for i, (smiles, score, fp) in enumerate(zip(valid_new_smiles, new_scores, new_fps)):
            if smiles is None or fp is None:
                continue

            docking_score = score[0] if isinstance(score, list) and len(score) > 0 else score
            if docking_score is None or docking_score < -20:
                continue

            neighbors = []
            for _, row in self.high_activity_memory.iterrows():
                if row['fps'] is None:
                    continue

                score1 = row['scores'][0] if isinstance(row['scores'], list) and len(row['scores']) > 0 else row['scores']
                score2 = docking_score

                diff_abs = abs(score1 - score2)
                if diff_abs < beta1:
                    continue

                try:
                    similarity = DataStructs.FingerprintSimilarity(row['fps'], fp)
                    dist = 1 - similarity
                    if dist <= 0:
                        continue

                    ACI = diff_abs / dist
                    if ACI >= beta2:
                        neighbors.append({
                            'neighbor_smiles': row['smiles'],
                            'neighbor_score': score1,
                            'delta': score1 - score2,
                            'diff_abs': diff_abs,
                            'dist': dist,
                            'ACI': ACI
                        })
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue

            if neighbors:
                cliff_neighbors.append({
                    'smiles': smiles,
                    'score': docking_score,
                    'neighbors': neighbors
                })

        return cliff_neighbors
    
    def update_memory(self, new_smiles, new_scores):
        """
        Update memory storage
        
        Args:
            new_smiles (list): New molecule SMILES list
            new_scores (list): New molecule scores list
        """
        # Calculate fingerprints
        fps, valid_smiles = self.calc_fingerprints(new_smiles)
        
        # Add new molecules to high activity memory
        for smiles, score, fp in zip(valid_smiles, new_scores, fps):
            # Check if molecule and fingerprint are valid
            if smiles is not None and fp is not None:
                # Check if score is valid (using docking score, i.e., first dimension)
                if isinstance(score, list) and len(score) > 0:
                    docking_score = score[0]  # Use docking score
                else:
                    docking_score = score
                
                # Only add if docking score is valid
                if docking_score is not None and docking_score >= -20:  # Reasonable docking score range
                    new_data = pd.DataFrame({
                        "smiles": [smiles],
                        "scores": [score],  # Store complete score vector
                        "fps": [fp]
                    })
                    self.high_activity_memory = pd.concat([self.high_activity_memory, new_data], 
                                                        ignore_index=True, sort=False)
        
        # Remove duplicates and sort
        self.high_activity_memory = self.high_activity_memory.drop_duplicates(subset=["smiles"])
        
        # Create temporary column for sorting (using docking score)
        def get_docking_score(scores):
            if isinstance(scores, list) and len(scores) > 0:
                return scores[0]
            return scores
        
        self.high_activity_memory['docking_score'] = self.high_activity_memory['scores'].apply(get_docking_score)
        self.high_activity_memory = self.high_activity_memory.sort_values('docking_score', ascending=False)
        self.high_activity_memory = self.high_activity_memory.drop('docking_score', axis=1)  # Remove temporary column
        self.high_activity_memory = self.high_activity_memory.reset_index(drop=True)
        
        # Limit memory storage size
        if len(self.high_activity_memory) > self.max_memory_size:
            self.high_activity_memory = self.high_activity_memory.head(self.max_memory_size)
    
    def add_cliff_pairs(self, cliff_pairs):
        """Add cliff molecule pairs to cliff memory storage"""
        for pair in cliff_pairs:
            # Add both molecules to cliff memory storage
            for smiles, score in [(pair['smiles1'], pair['scores1']), 
                                 (pair['smiles2'], pair['scores2'])]:
                fps, _ = self.calc_fingerprints([smiles])
                if fps[0] is not None:
                    new_data = pd.DataFrame({
                        "smiles": [smiles],
                        "scores": [score],
                        "fps": [fps[0]],
                        "cliff_type": [pair['cliff_type']]
                    })
                    self.cliff_memory = pd.concat([self.cliff_memory, new_data], 
                                                ignore_index=True, sort=False)
        
        # Remove duplicates
        self.cliff_memory = self.cliff_memory.drop_duplicates(subset=["smiles"])
        self.cliff_memory = self.cliff_memory.reset_index(drop=True)
    
    def get_cliff_molecules(self, n_samples=20):
        """Get cliff molecules for training"""
        if len(self.cliff_memory) == 0:
            return [], []
            
        n_samples = min(n_samples, len(self.cliff_memory))
        sampled = self.cliff_memory.sample(n_samples)
        
        return list(sampled['smiles']), list(sampled['scores'])
    
    def get_high_activity_molecules(self, n_samples=20):
        """Get high activity molecules for training"""
        if len(self.high_activity_memory) == 0:
            return [], []
            
        n_samples = min(n_samples, len(self.high_activity_memory))
        sampled = self.high_activity_memory.head(n_samples)
        
        return list(sampled['smiles']), list(sampled['scores'])
    
    def save_memory(self, filepath):
        """Save memory storage to file"""
        memory_data = {
            'high_activity': {
                'smiles': self.high_activity_memory['smiles'].tolist(),
                'scores': self.high_activity_memory['scores'].tolist()
            },
            'cliff_memory': {
                'smiles': self.cliff_memory['smiles'].tolist(),
                'scores': self.cliff_memory['scores'].tolist(),
                'cliff_type': self.cliff_memory['cliff_type'].tolist()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(memory_data, f, indent=2)
    
    def load_memory(self, filepath):
        """Load memory storage from file"""
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r') as f:
            memory_data = json.load(f)
        
        # Rebuild DataFrame
        if 'high_activity' in memory_data:
            self.high_activity_memory = pd.DataFrame(memory_data['high_activity'])
            # Recalculate fingerprints
            if len(self.high_activity_memory) > 0:
                fps, _ = self.calc_fingerprints(self.high_activity_memory['smiles'].tolist())
                self.high_activity_memory['fps'] = fps
        
        if 'cliff_memory' in memory_data:
            self.cliff_memory = pd.DataFrame(memory_data['cliff_memory'])
            # Recalculate fingerprints
            if len(self.cliff_memory) > 0:
                fps, _ = self.calc_fingerprints(self.cliff_memory['smiles'].tolist())
                self.cliff_memory['fps'] = fps
    
    def get_statistics(self):
        """Get statistics information"""
        def get_avg_score(scores_series):
            """Calculate mean of scores, handling list types"""
            if len(scores_series) == 0:
                return 0
            
            # Extract first score (docking score) and calculate mean
            first_scores = []
            for score in scores_series:
                if isinstance(score, list) and len(score) > 0:
                    first_scores.append(score[0])  # Use docking score
                elif isinstance(score, (int, float)):
                    first_scores.append(score)
            
            return np.mean(first_scores) if first_scores else 0
        
        return {
            'high_activity_count': len(self.high_activity_memory),
            'cliff_count': len(self.cliff_memory),
            'avg_high_activity_score': get_avg_score(self.high_activity_memory['scores']),
            'avg_cliff_score': get_avg_score(self.cliff_memory['scores'])
        }
