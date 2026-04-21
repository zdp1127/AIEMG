import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import List, Dict, Tuple
import json
import os

class IEREvaluator:
    
    def __init__(self, pareto_pool: List[str] = None, history_pool: List[str] = None):
       
        self.pareto_pool = pareto_pool or []
        self.history_pool = history_pool or []
        
        self.pareto_fps = self._calc_fingerprints(self.pareto_pool)
        self.history_fps = self._calc_fingerprints(self.history_pool)
        
    def _calc_fingerprints(self, smiles_list: List[str]) -> List:
        
        fps = []
        for smiles in smiles_list:
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        fps.append(fp)
                    else:
                        fps.append(None)
                except:
                    fps.append(None)
            else:
                fps.append(None)
        return fps
    
    def _tanimoto_distance(self, fp1, fp2) -> float:
       
        if fp1 is None or fp2 is None:
            return 1.0  
        
        try:
            similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
            return 1.0 - similarity  # 距离 = 1 - 相似性
        except:
            return 1.0
    
    def _calculate_novelty(self, target_smiles: str, pool_fps: List) -> float:
        if not pool_fps or all(fp is None for fp in pool_fps):
            return 1.0  
        
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol is None:
                return 0.0
            
            target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2048)
            
            min_distance = 1.0
            valid_distances = []
            
            for fp in pool_fps:
                if fp is not None:
                    distance = self._tanimoto_distance(target_fp, fp)
                    valid_distances.append(distance)
                    min_distance = min(min_distance, distance)
            
            if not valid_distances:
                return 1.0
            
            avg_distance = np.mean(valid_distances)
            return avg_distance
            
        except Exception as e:
            print(f"error: {e}")
            return 0.0
    
    def _calculate_duplicate_penalty(self, target_smiles: str) -> float:
        if target_smiles in self.pareto_pool:
            return 1.0  
        
        if target_smiles in self.history_pool:
            return 0.5  
        
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol is None:
                return 0.0
            
            target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, radius=2, nBits=2048)
            
            for fp in self.pareto_fps:
                if fp is not None:
                    similarity = DataStructs.FingerprintSimilarity(target_fp, fp)
                    if similarity > 0.95:
                        return 0.8  
            
            
            for fp in self.history_fps:
                if fp is not None:
                    similarity = DataStructs.FingerprintSimilarity(target_fp, fp)
                    if similarity > 0.95:
                        return 0.3  
            
            return 0.0  
            
        except Exception as e:
            print(f"error: {e}")
            return 0.0
    
    def calculate_ier(self, target_smiles: str) -> float:
        
        pareto_novelty = self._calculate_novelty(target_smiles, self.pareto_fps)
        
        history_novelty = self._calculate_novelty(target_smiles, self.history_fps)
        
        duplicate_penalty = self._calculate_duplicate_penalty(target_smiles)
        
        ier = pareto_novelty + history_novelty - duplicate_penalty
        
        ier = max(0.0, min(2.0, ier))
        
        return ier
    
    def update_pools(self, pareto_pool: List[str] = None, history_pool: List[str] = None):
        if pareto_pool is not None:
            self.pareto_pool = pareto_pool
            self.pareto_fps = self._calc_fingerprints(pareto_pool)
        
        if history_pool is not None:
            self.history_pool = history_pool
            self.history_fps = self._calc_fingerprints(history_pool)
    
    def add_to_history(self, smiles: str):
        if smiles and smiles not in self.history_pool:
            self.history_pool.append(smiles)
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    self.history_fps.append(fp)
                else:
                    self.history_fps.append(None)
            except:
                self.history_fps.append(None)
    
    def get_pool_stats(self) -> Dict:
        
        return {
            'pareto_pool_size': len(self.pareto_pool),
            'history_pool_size': len(self.history_pool),
            'pareto_valid_fps': sum(1 for fp in self.pareto_fps if fp is not None),
            'history_valid_fps': sum(1 for fp in self.history_fps if fp is not None)
        }
