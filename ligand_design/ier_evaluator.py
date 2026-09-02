"""
IER (Intrinsic-Extrinsic Reward) Evaluation Module

Complete IER implementation based on mathematical formulas:
IER(m) = α * Nov_P(m) + (1-α) * Nov_H(m) - γ * Rep(m)

Where:
- Nov_P(m) = min_{m'∈L_P} T_dist(m, m')  (Pareto frontier novelty)
- Nov_H(m) = (1/k) * Σ_{i=1}^{k} T_dist(m, m_(i))  (Historical novelty, using k-nearest neighbors)
- Rep(m) = (1/k) * Σ_{i=1}^{k} I[T_sim(m, m_(i)) ≥ τ_rep]  (Repetition penalty)
- T_dist = 1 - T_sim (Tanimoto distance)

Reference pools:
- L_P: Pareto pool (current non-dominated solutions)
- L_H^(t): Full history pool (all molecules evaluated up to search step t)
- L̃_H^(t): Bounded active history memory (used for IER evaluation)

Formula guarantee: -γ ≤ IER(m) ≤ 1
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from typing import List, Dict, Tuple, Optional
import json
import os
import time
from collections import deque


class IEREvaluator:
    """IER (Intrinsic-Extrinsic Reward) Evaluator

    Complete IER implementation including:
    - Pareto frontier novelty (Nov_P)
    - Historical novelty (Nov_H) - using bounded active history memory
    - Repetition penalty (Rep)
    """

    def __init__(
        self,
        pareto_pool: List[str] = None,
        history_pool: List[str] = None,
        alpha: float = 0.5,
        gamma: float = 0.3,
        k_neighbors: int = 5,
        h_max: int = 1000,
        tau_rep: float = 0.85
    ):
        """
        Initialize IER Evaluator

        Args:
            pareto_pool: List of molecule SMILES in Pareto pool (L_P)
            history_pool: List of historically generated molecule SMILES (L_H^(t))
            alpha: Weight coefficient for Pareto novelty vs historical novelty [0,1]
                   α=1 uses only Pareto novelty
                   α=0 uses only historical novelty
            gamma: Repetition penalty intensity coefficient (γ ≥ 0)
            k_neighbors: Number of neighbors used for computing Nov_H and Rep
            h_max: Maximum capacity of active history memory
            tau_rep: Near-duplicate similarity threshold
        """
        self.pareto_pool = pareto_pool or []
        self.history_pool = history_pool or []

        # IER parameters
        self.alpha = alpha
        self.gamma = gamma
        self.k_neighbors = k_neighbors
        self.h_max = h_max
        self.tau_rep = tau_rep

        # Pareto fingerprint pool
        self.pareto_fps = self._calc_fingerprints(self.pareto_pool)

        # History fingerprint pool - using deque for bounded memory
        self._full_history_fps = self._calc_fingerprints(self.history_pool)
        self._history_smiles_deque = deque(self.history_pool, maxlen=h_max)

        # Current active history memory fingerprints (used for IER evaluation)
        self._update_active_history_fps()

        # Debug statistics
        self._stats = {
            'total_evaluations': 0,
            'avg_pareto_novelty': 0.0,
            'avg_history_novelty': 0.0,
            'avg_repetition': 0.0,
            'avg_ier': 0.0
        }

    def _calc_fingerprints(self, smiles_list: List[str]) -> List:
        """Calculate molecular fingerprint list"""
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
        """Calculate Tanimoto distance between two molecular fingerprints"""
        if fp1 is None or fp2 is None:
            return 1.0  # Invalid fingerprint returns maximum distance

        try:
            similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
            return 1.0 - similarity  # T_dist = 1 - T_sim
        except:
            return 1.0

    def _tanimoto_similarity(self, fp1, fp2) -> float:
        """Calculate Tanimoto similarity between two molecular fingerprints"""
        if fp1 is None or fp2 is None:
            return 0.0

        try:
            return DataStructs.FingerprintSimilarity(fp1, fp2)
        except:
            return 0.0

    def _get_fingerprint(self, smiles: str) -> Optional:
        """Get fingerprint for a molecule, returns None if invalid"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except:
            pass
        return None

    def _update_active_history_fps(self):
        """Update fingerprint list for active history memory"""
        # Get SMILES of current active history memory from deque
        active_smiles = list(self._history_smiles_deque)
        self.active_history_fps = self._calc_fingerprints(active_smiles)

    def _get_active_history_memory(self) -> Tuple[List[str], List]:
        """
        Get bounded active history memory

        Returns:
            (smiles list, fingerprint list)
        """
        active_smiles = list(self._history_smiles_deque)
        return active_smiles, self.active_history_fps

    def _calculate_pareto_novelty(self, target_fp) -> float:
        """
        Calculate Pareto frontier novelty

        Nov_P(m) = min_{m'∈L_P} T_dist(m, m')

        The minimum Tanimoto distance between the target molecule and all molecules in the Pareto pool
        """
        if not self.pareto_fps or all(fp is None for fp in self.pareto_fps):
            return 1.0  # Pareto pool is empty, return maximum novelty

        min_distance = 1.0
        for fp in self.pareto_fps:
            if fp is not None:
                distance = self._tanimoto_distance(target_fp, fp)
                if distance < min_distance:
                    min_distance = distance

        return min_distance

    def _calculate_history_novelty(self, target_fp) -> float:
        """
        Calculate historical novelty

        Nov_H(m) = (1/k) * Σ_{i=1}^{k} T_dist(m, m_(i))

        Using k-nearest neighbors average Tanimoto distance
        """
        # Get active history memory
        active_smiles, active_fps = self._get_active_history_memory()

        if not active_fps or all(fp is None for fp in active_fps):
            return 1.0  # History pool is empty, return maximum novelty

        # Calculate distance to all historical molecules
        distances = []
        for fp in active_fps:
            if fp is not None:
                distance = self._tanimoto_distance(target_fp, fp)
                distances.append(distance)

        if not distances:
            return 1.0

        # Get k nearest neighbors (if available molecules are fewer than k, use all)
        k = min(self.k_neighbors, len(distances))
        distances.sort()

        # Calculate average distance of k nearest neighbors
        avg_k_distances = sum(distances[:k]) / k

        return avg_k_distances

    def _calculate_repetition_penalty(self, target_fp) -> float:
        """
        Calculate repetition penalty

        Rep(m) = (1/k) * Σ_{i=1}^{k} I[T_sim(m, m_(i)) ≥ τ_rep]

        Calculate the proportion of k nearest neighbors with similarity exceeding threshold τ_rep
        """
        # Get active history memory
        active_smiles, active_fps = self._get_active_history_memory()

        if not active_fps or all(fp is None for fp in active_fps):
            return 0.0  # History pool is empty, no repetition

        # Calculate similarity to all historical molecules
        similarities = []
        for fp in active_fps:
            if fp is not None:
                similarity = self._tanimoto_similarity(target_fp, fp)
                similarities.append(similarity)

        if not similarities:
            return 0.0

        # Get similarity of k nearest neighbors (sorted by distance)
        # Note: We need to use distance to get true k-nearest neighbors
        distances = []
        for fp in active_fps:
            if fp is not None:
                distance = self._tanimoto_distance(target_fp, fp)
                distances.append(distance)
            else:
                distances.append(1.0)

        k = min(self.k_neighbors, len(distances))
        sorted_indices = np.argsort(distances)[:k]

        # Calculate proportion of k nearest neighbors with similarity exceeding threshold
        count_similar = 0
        for idx in sorted_indices:
            if similarities[idx] >= self.tau_rep:
                count_similar += 1

        repetition = count_similar / k
        return repetition

    def calculate_ier(self, target_smiles: str) -> float:
        """
        Calculate IER (Intrinsic-Extrinsic Reward)

        IER(m) = α * Nov_P(m) + (1-α) * Nov_H(m) - γ * Rep(m)

        Guarantee: -γ ≤ IER(m) ≤ 1

        Args:
            target_smiles: SMILES string of target molecule

        Returns:
            IER value (range: [-γ, 1])
        """
        self._stats['total_evaluations'] += 1

        # Get target molecule fingerprint
        target_fp = self._get_fingerprint(target_smiles)
        if target_fp is None:
            # Invalid molecule, return minimum value
            return -self.gamma

        # Calculate each component
        nov_p = self._calculate_pareto_novelty(target_fp)
        nov_h = self._calculate_history_novelty(target_fp)
        rep = self._calculate_repetition_penalty(target_fp)

        # Calculate IER
        ier = self.alpha * nov_p + (1 - self.alpha) * nov_h - self.gamma * rep

        # Update statistics
        self._update_stats(nov_p, nov_h, rep, ier)

        return ier

    def _update_stats(self, nov_p: float, nov_h: float, rep: float, ier: float):
        """Update statistics (for monitoring)"""
        n = self._stats['total_evaluations']
        # Sliding average update
        self._stats['avg_pareto_novelty'] = (
            (self._stats['avg_pareto_novelty'] * (n - 1) + nov_p) / n
        )
        self._stats['avg_history_novelty'] = (
            (self._stats['avg_history_novelty'] * (n - 1) + nov_h) / n
        )
        self._stats['avg_repetition'] = (
            (self._stats['avg_repetition'] * (n - 1) + rep) / n
        )
        self._stats['avg_ier'] = (
            (self._stats['avg_ier'] * (n - 1) + ier) / n
        )

    def calculate_ier_components(self, target_smiles: str) -> Dict[str, float]:
        """
        Calculate each component of IER (for analysis and debugging)

        Returns:
            Dictionary containing each component
        """
        target_fp = self._get_fingerprint(target_smiles)
        if target_fp is None:
            return {
                'Nov_P': 0.0,
                'Nov_H': 0.0,
                'Rep': 0.0,
                'IER': -self.gamma,
                'valid': False
            }

        nov_p = self._calculate_pareto_novelty(target_fp)
        nov_h = self._calculate_history_novelty(target_fp)
        rep = self._calculate_repetition_penalty(target_fp)
        ier = self.alpha * nov_p + (1 - self.alpha) * nov_h - self.gamma * rep

        return {
            'Nov_P': nov_p,
            'Nov_H': nov_h,
            'Rep': rep,
            'IER': ier,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'k_neighbors': self.k_neighbors,
            'tau_rep': self.tau_rep,
            'valid': True
        }

    def update_pools(
        self,
        pareto_pool: List[str] = None,
        history_pool: List[str] = None
    ):
        """Update molecule pools"""
        if pareto_pool is not None:
            self.pareto_pool = pareto_pool
            self.pareto_fps = self._calc_fingerprints(pareto_pool)

        if history_pool is not None:
            self.history_pool = history_pool
            # Maintain consistency with deque
            self._history_smiles_deque = deque(history_pool, maxlen=self.h_max)
            self._update_active_history_fps()

    def add_to_history(self, smiles: str):
        """Add molecule to history pool"""
        if smiles:
            # Add to deque (automatically handles bounded memory)
            self._history_smiles_deque.append(smiles)

            # Also update full history list (for saving)
            if smiles not in self.history_pool:
                self.history_pool.append(smiles)

            # Calculate and add fingerprint
            fp = self._get_fingerprint(smiles)
            self.active_history_fps.append(fp)

            # If active history is full, remove oldest fingerprint
            if len(self.active_history_fps) > self.h_max:
                self.active_history_fps = self.active_history_fps[-self.h_max:]

    def get_pool_stats(self) -> Dict:
        """Get molecule pool statistics"""
        return {
            'pareto_pool_size': len(self.pareto_pool),
            'history_pool_size': len(self.history_pool),
            'active_history_size': len(list(self._history_smiles_deque)),
            'h_max': self.h_max,
            'pareto_valid_fps': sum(1 for fp in self.pareto_fps if fp is not None),
            'active_history_valid_fps': sum(1 for fp in self.active_history_fps if fp is not None),
            'ier_parameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'k_neighbors': self.k_neighbors,
                'tau_rep': self.tau_rep
            },
            'evaluation_stats': self._stats
        }

    def get_active_history_memory(self) -> List[str]:
        """Get list of molecules in current active history memory"""
        return list(self._history_smiles_deque)

    def set_parameters(
        self,
        alpha: float = None,
        gamma: float = None,
        k_neighbors: int = None,
        h_max: int = None,
        tau_rep: float = None
    ):
        """
        Update IER parameters

        Args:
            alpha: Pareto/historical novelty weight [0,1]
            gamma: Repetition penalty intensity [0,∞)
            k_neighbors: Number of k-nearest neighbors
            h_max: Active history memory capacity
            tau_rep: Similarity threshold [0,1]
        """
        if alpha is not None:
            self.alpha = np.clip(alpha, 0.0, 1.0)
        if gamma is not None:
            self.gamma = max(0.0, gamma)
        if k_neighbors is not None:
            self.k_neighbors = max(1, k_neighbors)
        if h_max is not None:
            old_h_max = self.h_max
            self.h_max = max(1, h_max)
            # If new h_max is smaller, need to rebuild deque
            if self.h_max < old_h_max:
                self._history_smiles_deque = deque(
                    list(self._history_smiles_deque),
                    maxlen=self.h_max
                )
                self._update_active_history_fps()
        if tau_rep is not None:
            self.tau_rep = np.clip(tau_rep, 0.0, 1.0)

    def save_state(self, filepath: str):
        """Save IER evaluator state to file"""
        state = {
            'pareto_pool': self.pareto_pool,
            'history_pool': self.history_pool,
            'parameters': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'k_neighbors': self.k_neighbors,
                'h_max': self.h_max,
                'tau_rep': self.tau_rep
            },
            'stats': self._stats,
            'timestamp': time.asctime(time.localtime(time.time()))
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, filepath: str) -> 'IEREvaluator':
        """Load IER evaluator state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)

        evaluator = cls(
            pareto_pool=state.get('pareto_pool', []),
            history_pool=state.get('history_pool', []),
            alpha=state.get('parameters', {}).get('alpha', 0.5),
            gamma=state.get('parameters', {}).get('gamma', 0.3),
            k_neighbors=state.get('parameters', {}).get('k_neighbors', 5),
            h_max=state.get('parameters', {}).get('h_max', 1000),
            tau_rep=state.get('parameters', {}).get('tau_rep', 0.85)
        )
        evaluator._stats = state.get('stats', evaluator._stats)
        return evaluator

    def __repr__(self):
        return (
            f"IEREvaluator("
            f"pareto={len(self.pareto_pool)}, "
            f"history={len(self.history_pool)}, "
            f"α={self.alpha}, γ={self.gamma}, "
            f"k={self.k_neighbors}, H_max={self.h_max}, "
            f"τ_rep={self.tau_rep})"
        )
