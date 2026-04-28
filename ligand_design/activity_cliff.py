
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
import os

class ActivityCliffDetector:
    
    def __init__(self):
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_memory_size = max_memory_size
        
        self.high_activity_memory = pd.DataFrame(columns=["smiles", "scores", "fps"])
        
        self.cliff_memory = pd.DataFrame(columns=["smiles", "scores", "fps", "cliff_type"])
        
    def calc_fingerprints(self, smiles_list):
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

    @staticmethod
    def _get_docking_score(score):
        if isinstance(score, list) and len(score) > 0:
            return float(score[0])
        if isinstance(score, (int, float, np.floating)):
            return float(score)
        return None

    def _calc_aci(self, docking_m, docking_mp, fp_m, fp_mp):
        diff_abs = abs(float(docking_m) - float(docking_mp))
        similarity = DataStructs.FingerprintSimilarity(fp_mp, fp_m)
        dist = 1.0 - float(similarity)
        if dist <= 0.0:
            return None, diff_abs, dist
        return diff_abs / dist, diff_abs, dist

    def calculate_acr(self, smiles, docking_score):
        if len(self.high_activity_memory) == 0:
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        fp_m = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if fp_m is None:
            return 0.0
        beta1 = float(self.alpha1)
        beta2 = float(self.alpha2)
        if beta1 <= 0.0:
            return 0.0
        terms = []
        f_m = float(docking_score)
        for _, row in self.high_activity_memory.iterrows():
            fp_mp = row.get('fps', None)
            if fp_mp is None:
                continue
            f_mp = self._get_docking_score(row.get('scores', None))
            if f_mp is None:
                continue
            aci, diff_abs, _ = self._calc_aci(f_m, f_mp, fp_m, fp_mp)
            if aci is None:
                continue
            if diff_abs >= beta1 and aci >= beta2:
                terms.append((f_m - f_mp) / beta1)
        if len(terms) == 0:
            return 0.0
        acr = float(np.mean(terms))
        if acr > 1.0:
            acr = 1.0
        if acr < -1.0:
            acr = -1.0
        return acr
    
    def detect_activity_cliffs(self, new_smiles, new_scores):
        
        cliff_pairs = []
        
        if len(self.high_activity_memory) == 0:
            return cliff_pairs
            
        new_fps, valid_new_smiles = self.calc_fingerprints(new_smiles)
        
        for i, (smiles, score, fp) in enumerate(zip(valid_new_smiles, new_scores, new_fps)):
            if smiles is None or fp is None:
                continue
                
            if isinstance(score, list) and len(score) > 0:
                docking_score = score[0]  # 使用docking分数
            else:
                docking_score = score
                
            if docking_score is None or docking_score < -20:  # 合理的docking分数范围
                continue
                
            for j, row in self.high_activity_memory.iterrows():
                if row['fps'] is None:
                    continue
                    
                if isinstance(row['scores'], list) and len(row['scores']) > 0:
                    score1 = row['scores'][0] 
                else:
                    score1 = row['scores']
                    
                if isinstance(score, list) and len(score) > 0:
                    score2 = score[0]  
                else:
                    score2 = score
                    
                diff_abs = abs(score1 - score2)
                if diff_abs < self.alpha1:
                    continue
                    
                try:
                    similarity = DataStructs.FingerprintSimilarity(row['fps'], fp)
                    dist = 1 - similarity
                    
                    if dist == 0:  
                        continue
                        
                   
                    ACI = diff_abs / dist
                    
                    if ACI >= self.alpha2:
                        cliff_type = "high_to_low" if score1 > score2 else "low_to_high"
                        cliff_pairs.append({
                            'smiles1': row['smiles'],
                            'scores1': score1,  
                            'scores2': score2,  
                            'smiles2': smiles,
                            'ACI': ACI,
                            'cliff_type': cliff_type
                        })
                        
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    continue
                    
        return cliff_pairs
    
    def update_memory(self, new_smiles, new_scores):
        
        fps, valid_smiles = self.calc_fingerprints(new_smiles)
        
        
        for smiles, score, fp in zip(valid_smiles, new_scores, fps):
            
            if smiles is not None and fp is not None:
               
                if isinstance(score, list) and len(score) > 0:
                    docking_score = score[0]  # 使用docking分数
                else:
                    docking_score = score
                
                
                if docking_score is not None and docking_score >= -20:  # 合理的docking分数范围
                    new_data = pd.DataFrame({
                        "smiles": [smiles],
                        "scores": [score],  
                        "fps": [fp]
                    })
                    self.high_activity_memory = pd.concat([self.high_activity_memory, new_data], 
                                                        ignore_index=True, sort=False)
        
        
        self.high_activity_memory = self.high_activity_memory.drop_duplicates(subset=["smiles"])
        
        def get_docking_score(scores):
            if isinstance(scores, list) and len(scores) > 0:
                return scores[0]
            return scores
        
        self.high_activity_memory['docking_score'] = self.high_activity_memory['scores'].apply(get_docking_score)
        self.high_activity_memory = self.high_activity_memory.sort_values('docking_score', ascending=False)
        self.high_activity_memory = self.high_activity_memory.drop('docking_score', axis=1)  # 删除临时列
        self.high_activity_memory = self.high_activity_memory.reset_index(drop=True)
        
    
        if len(self.high_activity_memory) > self.max_memory_size:
            self.high_activity_memory = self.high_activity_memory.head(self.max_memory_size)
    
    def add_cliff_pairs(self, cliff_pairs):
        for pair in cliff_pairs:
            
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
        
        
        self.cliff_memory = self.cliff_memory.drop_duplicates(subset=["smiles"])
        self.cliff_memory = self.cliff_memory.reset_index(drop=True)
    
    def get_cliff_molecules(self, n_samples=20):
        
        if len(self.cliff_memory) == 0:
            return [], []
            
        n_samples = min(n_samples, len(self.cliff_memory))
        sampled = self.cliff_memory.sample(n_samples)
        
        return list(sampled['smiles']), list(sampled['scores'])
    
    def get_high_activity_molecules(self, n_samples=20):
        
        if len(self.high_activity_memory) == 0:
            return [], []
            
        n_samples = min(n_samples, len(self.high_activity_memory))
        sampled = self.high_activity_memory.head(n_samples)
        
        return list(sampled['smiles']), list(sampled['scores'])
    
    def save_memory(self, filepath):
       
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
     
        if not os.path.exists(filepath):
            return
            
        with open(filepath, 'r') as f:
            memory_data = json.load(f)
        
        
        if 'high_activity' in memory_data:
            self.high_activity_memory = pd.DataFrame(memory_data['high_activity'])
         
            if len(self.high_activity_memory) > 0:
                fps, _ = self.calc_fingerprints(self.high_activity_memory['smiles'].tolist())
                self.high_activity_memory['fps'] = fps
        
        if 'cliff_memory' in memory_data:
            self.cliff_memory = pd.DataFrame(memory_data['cliff_memory'])
         
            if len(self.cliff_memory) > 0:
                fps, _ = self.calc_fingerprints(self.cliff_memory['smiles'].tolist())
                self.cliff_memory['fps'] = fps
    
    def get_statistics(self):
       
        def get_avg_score(scores_series):
           
            if len(scores_series) == 0:
                return 0
            
          
            first_scores = []
            for score in scores_series:
                if isinstance(score, list) and len(score) > 0:
                    first_scores.append(score[0])  # 使用docking分数
                elif isinstance(score, (int, float)):
                    first_scores.append(score)
            
            return np.mean(first_scores) if first_scores else 0
        return {
            'high_activity_count': len(self.high_activity_memory),
            'cliff_count': len(self.cliff_memory),
            'avg_high_activity_score': get_avg_score(self.high_activity_memory['scores']),
            'avg_cliff_score': get_avg_score(self.cliff_memory['scores'])
        }
