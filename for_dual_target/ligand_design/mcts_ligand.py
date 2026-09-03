from math import *
import random
import random as pr
import numpy as np
from copy import deepcopy
import time

#from rdkit.Chem.QED import qed
from load_model import loaded_model
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node
from activity_cliff import ActivityCliffDetector
from pygmo import hypervolume
import copy

import os
import json
import errno

import argparse

from joblib import Parallel, delayed
import pdb

# ============================================================================
DISABLE_ACR_COMPONENT = False  # When True, ACS equals normalized docking only (ablation study)
ACS_DOCKING_WEIGHT = 0.5       # Weight of docking component in ACS
ACS_ACR_WEIGHT = 0.5           # Weight of ACR component in ACS
CLIFF_ALPHA1 = 0.5             # Activity difference threshold
CLIFF_ALPHA2 = 2.0             # Activity cliff index threshold
UCB_EXPLORATION_CONSTANT = 1.41421356237  # UCB exploration constant (default ≈ sqrt(2))
VISIT_PENALTY_COEFF = 0.01               # Visit count penalty coefficient
MAX_SELECTION_STEPS = 50                 # Maximum steps in selection phase
MAX_SMILES_LENGTH = 81                   # Upper limit approximation for molecule growth

# === IER (Intrinsic-Extrinsic Reward) Parameters ===
IER_ALPHA = 0.5              # Weight of Pareto novelty vs history novelty: IER = α*Nov_P + (1-α)*Nov_H - γ*Rep
IER_GAMMA = 0.3              # Repetition penalty strength coefficient
IER_K_NEIGHBORS = 5          # Number of k-nearest neighbors for Nov_H and Rep calculation
IER_H_MAX = 1000             # Maximum capacity of active history memory
IER_TAU_REP = 0.85           # Near-repetition similarity threshold

# Set random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class chemical:

    def __init__(self):

        self.position=['&']
        self.num_atom=8
        self.vl=['\n', '&', 'C', '(', 'c', '1', 'o', '=', 'O', 'N', 'F', '[C@@H]',
        'n', '-', '#', 'S', 'Cl', '[O-]', '[C@H]', '[NH+]', '[C@]', 's', 'Br', '/', '[nH]', '[NH3+]',
        '[NH2+]', '[C@@]', '[N+]', '[nH+]', '\\', '[S@]', '[N-]', '[n+]', '[S@@]', '[S-]',
        'I', '[n-]', 'P', '[OH+]', '[NH-]', '[P@@H]', '[P@@]', '[PH2]', '[P@]', '[P+]', '[S+]',
        '[o+]', '[CH2-]', '[CH-]', '[SH+]', '[O+]', '[s+]', '[PH+]', '[PH]', '[S@@+]']

    def Clone(self):

        st = chemical()
        st.position= self.position[:]
        return st

    def SelectPosition(self,m):
        self.position.append(m)

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class pareto:

    def __init__(self, front=[], size=0, avg=[], compounds=[], cliff_detector=None, ier_evaluator=None):
        self.front=front
        self.size=size
        self.avg=avg
        self.compounds=compounds
        # Activity cliff detector
        self.cliff_detector = cliff_detector
        # IER evaluator
        self.ier_evaluator = ier_evaluator
        # IER parameters (for initialization)
        self.ier_alpha = IER_ALPHA
        self.ier_gamma = IER_GAMMA
        self.ier_k_neighbors = IER_K_NEIGHBORS
        self.ier_h_max = IER_H_MAX
        self.ier_tau_rep = IER_TAU_REP

    def Dominated(self,m):
        if len(self.front) == 0:
            return False
        
        for p in self.front:
            flag = True
            for i in range(len(p)):
                if m[i]>=p[i]:
                    flag = False
            if(flag):
                return True
        
        return False

    def Update(self,scores,compound):
        # IER value was pre-calculated and stored in scores[2]
        # No recalculation here to avoid performance overhead
        
        del_list = []
        for k in range(len(self.front)):
            flag = True
            for i in range(len(self.front[k])):
                if(self.front[k][i]>=scores[i]):
                    flag = False
            if(flag):
                del_list.append(k-len(del_list))
        for i in range(len(del_list)):
            del self.front[del_list[i]]
            del self.compounds[del_list[i]]
        self.front.append(scores)
        self.compounds.append(compound)
        
        # Activity cliff detection and update
        if self.cliff_detector is not None:
            # Update high-activity memory
            self.cliff_detector.update_memory([compound], [scores])
            
            # Detect activity cliffs
            cliff_pairs = self.cliff_detector.detect_activity_cliffs([compound], [scores])
            if cliff_pairs:
                print(f"Detected {len(cliff_pairs)} activity cliff pairs")
                self.cliff_detector.add_cliff_pairs(cliff_pairs)
                
                # Record cliff information to main output file
                f = open(dataDir+"present/output.txt", 'a')
                print("Activity Cliffs detected:", file=f)
                for pair in cliff_pairs:
                    print(f"  {pair['smiles1']} (score: {pair['scores1']:.3f}) -> {pair['smiles2']} (score: {pair['scores2']:.3f}) ACI: {pair['ACI']:.3f}", file=f)
                f.close()
                
                # Save detailed cliff pair information to dedicated file
                self._save_cliff_pairs_detailed(cliff_pairs, compound, scores)
            
            # Calculate ACS score (after cliff detection) - uses average of EGFR and 3PP0 docking scores
            if len(scores) >= 6:  # Ensure ACS score position (index 5) exists
                scores[5] = _calculate_acs(scores[0], scores[1], self.cliff_detector, compound, compound_scores=scores)
                print(f"Calculated ACS score: {scores[5]:.3f} (EGFR: {scores[0]:.3f}, 3PP0: {scores[1]:.3f}, avg: {(scores[0]+scores[1])/2:.3f})")
        else:
            # If no cliff detector, use average of normalized docking scores as ACS
            if len(scores) >= 6:
                avg_dock = (scores[0] + scores[1]) / 2
                scores[5] = _calculate_acs(scores[0], scores[1], None, compound)
                print(f"Using average docking score as ACS: {scores[5]:.3f}")
        
        f = open(dataDir+"present/output.txt", 'a')
        
        print("pareto size:",len(self.front),file=f)
        print("Updated pareto front",self.front, file=f)
        print("Pareto Ligands",self.compounds,file=f)
        print("Time;",time.asctime( time.localtime(time.time()) ),file=f)
        f.close()
       
        print("pareto size:",len(self.front))
        print("Updated pareto front",self.front)
        
        self.avgcal()

    def avgcal(self):
        for i in range(len(self.avg)):
            self.avg[i] = 0
        for i in range(len(self.front)):
            for j in range(len(self.avg)):
                self.avg[j]+=self.front[i][j]/len(self.front)
    
    def get_cliff_molecules(self, n_samples=20):
        """Get activity cliff molecules for training"""
        if self.cliff_detector is None:
            return [], []
        return self.cliff_detector.get_cliff_molecules(n_samples)
    
    def get_high_activity_molecules(self, n_samples=20):
        """Get high-activity molecules for training"""
        if self.cliff_detector is None:
            return [], []
        return self.cliff_detector.get_high_activity_molecules(n_samples)
    
    def get_cliff_statistics(self):
        """Get activity cliff statistics"""
        if self.cliff_detector is None:
            return {}
        return self.cliff_detector.get_statistics()
    
    def _save_cliff_pairs_detailed(self, cliff_pairs, compound, scores):
        """Save detailed activity cliff pair information to dedicated file"""
        import time
        import json
        
        # Create cliff pair details
        cliff_details = {
            'timestamp': time.asctime(time.localtime(time.time())),
            'new_compound': compound,
            'new_scores': scores,
            'cliff_pairs': []
        }
        
        for pair in cliff_pairs:
            cliff_info = {
                'molecule1': {
                    'smiles': pair['smiles1'],
                    'docking_score': pair['scores1'],
                    'cliff_type': pair['cliff_type']
                },
                'molecule2': {
                    'smiles': pair['smiles2'],
                    'docking_score': pair['scores2'],
                    'cliff_type': pair['cliff_type']
                },
                'activity_cliff_index': pair['ACI'],
                'activity_difference': abs(pair['scores1'] - pair['scores2']),
                'cliff_direction': pair['cliff_type']
            }
            cliff_details['cliff_pairs'].append(cliff_info)
        
        # Save to JSON file
        cliff_file = dataDir + "present/activity_cliffs.json"
        try:
            # If file exists, read existing data
            if os.path.exists(cliff_file):
                with open(cliff_file, 'r') as f:
                    all_cliffs = json.load(f)
            else:
                all_cliffs = {'cliff_events': []}
            
            # Add new cliff events
            all_cliffs['cliff_events'].append(cliff_details)
            
            # Save updated data
            with open(cliff_file, 'w') as f:
                json.dump(all_cliffs, f, indent=2)
                
        except Exception as e:
            print(f"Error saving cliff pair information: {e}")
        
        # Also save to text file for viewing
        cliff_txt_file = dataDir + "present/activity_cliffs.txt"
        with open(cliff_txt_file, 'a') as f:
            f.write(f"\n=== Activity Cliff Pair Detection - {time.asctime(time.localtime(time.time()))} ===\n")
            f.write(f"New molecule: {compound}\n")
            f.write(f"New molecule scores: {scores}\n")
            f.write(f"Detected {len(cliff_pairs)} activity cliff pairs:\n\n")
            
            for i, pair in enumerate(cliff_pairs, 1):
                f.write(f"Cliff pair {i}:\n")
                f.write(f"  Molecule 1: {pair['smiles1']}\n")
                f.write(f"  Molecule 1 score: {pair['scores1']:.3f}\n")
                f.write(f"  Molecule 2: {pair['smiles2']}\n")
                f.write(f"  Molecule 2 score: {pair['scores2']:.3f}\n")
                f.write(f"  Activity Cliff Index (ACI): {pair['ACI']:.3f}\n")
                f.write(f"  Activity difference: {abs(pair['scores1'] - pair['scores2']):.3f}\n")
                f.write(f"  Cliff type: {pair['cliff_type']}\n")
                f.write(f"  Cliff direction: {'high-to-low' if pair['cliff_type'] == 'high_to_low' else 'low-to-high'}\n")
                f.write("-" * 50 + "\n")
            f.write("\n")
    
    def _save_cliff_statistics(self, cliff_stats):
        """Save activity cliff statistics"""
        import time
        import json
        
        # Create statistics info
        stats_info = {
            'timestamp': time.asctime(time.localtime(time.time())),
            'high_activity_count': cliff_stats.get('high_activity_count', 0),
            'cliff_count': cliff_stats.get('cliff_count', 0),
            'avg_high_activity_score': cliff_stats.get('avg_high_activity_score', 0),
            'avg_cliff_score': cliff_stats.get('avg_cliff_score', 0)
        }
        
        # Save to JSON file
        stats_file = dataDir + "present/cliff_statistics.json"
        try:
            # If file exists, read existing data
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    all_stats = json.load(f)
            else:
                all_stats = {'statistics_history': []}
            
            # Add new statistics
            all_stats['statistics_history'].append(stats_info)
            
            # Save updated data
            with open(stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
                
        except Exception as e:
            print(f"Error saving statistics: {e}")
        
        # Also save to text file
        stats_txt_file = dataDir + "present/cliff_statistics.txt"
        with open(stats_txt_file, 'a') as f:
            f.write(f"{time.asctime(time.localtime(time.time()))} | "
                   f"High-activity molecules: {cliff_stats.get('high_activity_count', 0)} | "
                   f"Cliff molecules: {cliff_stats.get('cliff_count', 0)} | "
                   f"Avg high-activity score: {cliff_stats.get('avg_high_activity_score', 0):.3f} | "
                   f"Avg cliff score: {cliff_stats.get('avg_cliff_score', 0):.3f}\n")
    
    def _generate_final_cliff_report(self):
        """Generate final activity cliff pair summary report"""
        import time
        import json
        
        if self.cliff_detector is None:
            return
        
        # Get all cliff molecules
        cliff_smiles, cliff_scores = self.cliff_detector.get_cliff_molecules(1000)  # Get all cliff molecules
        high_activity_smiles, high_activity_scores = self.cliff_detector.get_high_activity_molecules(1000)
        
        # Generate summary report
        report = {
            'generation_time': time.asctime(time.localtime(time.time())),
            'summary': {
                'total_high_activity_molecules': len(high_activity_smiles),
                'total_cliff_molecules': len(cliff_smiles),
                'cliff_detection_parameters': {
                    'alpha1': self.cliff_detector.alpha1,
                    'alpha2': self.cliff_detector.alpha2,
                    'max_memory_size': self.cliff_detector.max_memory_size
                }
            },
            'high_activity_molecules': [
                {'smiles': smiles, 'scores': scores} 
                for smiles, scores in zip(high_activity_smiles, high_activity_scores)
            ],
            'cliff_molecules': [
                {'smiles': smiles, 'scores': scores} 
                for smiles, scores in zip(cliff_smiles, cliff_scores)
            ]
        }
        
        # Save summary report
        report_file = dataDir + "present/final_cliff_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate readable text report
        txt_report_file = dataDir + "present/final_cliff_report.txt"
        with open(txt_report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Activity Cliff Pair Detection Final Summary Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generation time: {time.asctime(time.localtime(time.time()))}\n\n")
            
            f.write("Detection parameters:\n")
            f.write(f"  Activity difference threshold (alpha1): {self.cliff_detector.alpha1}\n")
            f.write(f"  Activity cliff index threshold (alpha2): {self.cliff_detector.alpha2}\n")
            f.write(f"  Max memory size: {self.cliff_detector.max_memory_size}\n\n")
            
            f.write("Statistics:\n")
            f.write(f"  Total high-activity molecules: {len(high_activity_smiles)}\n")
            f.write(f"  Total cliff molecules: {len(cliff_smiles)}\n\n")
            
            f.write("High-activity molecule list:\n")
            f.write("-" * 40 + "\n")
            for i, (smiles, scores) in enumerate(zip(high_activity_smiles, high_activity_scores), 1):
                f.write(f"{i:3d}. {smiles}\n")
                f.write(f"     Scores: {scores}\n")
            
            f.write("\nCliff molecule list:\n")
            f.write("-" * 40 + "\n")
            for i, (smiles, scores) in enumerate(zip(cliff_smiles, cliff_scores), 1):
                f.write(f"{i:3d}. {smiles}\n")
                f.write(f"     Scores: {scores}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Final cliff pair summary report saved to: {txt_report_file}")
    
    def __len__(self):
        """Return the size of Pareto front"""
        return len(self.front)
    
    @staticmethod
    def from_dict(_filename):
        # should check _filename exists
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_pareto = pareto(front = _set_json['front'], size=_set_json['size'], avg=_set_json['avg'], compounds=_set_json['compounds'])
        _set_file.close()
        # Recreate cliff_detector
        new_pareto.cliff_detector = ActivityCliffDetector(alpha1=0.5, alpha2=2.0, max_memory_size=1000)
        print("Loaded Pareto Fronts")
        return new_pareto
    
    def to_dict(self):
        """Convert to serializable dict, excluding cliff_detector"""
        return {
            'front': self.front,
            'size': self.size,
            'avg': self.avg,
            'compounds': self.compounds
        }

class Node:

    def __init__(self, position = None,  parent = None, state = None, childNodes=[], child=None, wins=[0,0,0,0,0,0], visits=0, nonvisited_atom=None, type_node= [], depth=0):
        #super().__init__()
        #self.__dict__ = self
        self.position = position
        self.parentNode = parent
        self.childNodes = childNodes
        self.child=child
        self.wins = wins
        self.visits = visits
        self.nonvisited_atom=state.Getatom() if nonvisited_atom is None else nonvisited_atom
        self.type_node=type_node
        self.depth=depth

        self.cached_hv = None  # Cached hypervolume
        self._last_pareto_len = -1  # Previous Pareto front length
        self._last_wins_snapshot = None  # Previous wins snapshot

    def get_cached_hv(self, pareto_front):
        """Get cached hypervolume value to avoid recalculation"""
        front_snapshot = len(pareto_front)
        wins_snapshot = tuple(self.wins)
        if (
            self.cached_hv is None
            or self._last_pareto_len != front_snapshot
            or self._last_wins_snapshot != wins_snapshot
        ):
            self.cached_hv = self.hvcal(pareto_front, self.wins)
            self._last_pareto_len = front_snapshot
            self._last_wins_snapshot = wins_snapshot
        return self.cached_hv

    def Selectnode(self,pareto_front):
        w=[]
        for i in range(len(self.childNodes)):
            ucb=[]
            for win in self.childNodes[i].wins:
                # Both visit penalty and UCB exploration constant are configurable
                visit_penalty = VISIT_PENALTY_COEFF * sqrt(self.childNodes[i].visits)
                exploration_term = UCB_EXPLORATION_CONSTANT * sqrt(2*log(self.visits)/self.childNodes[i].visits)
                ucb.append(win/self.childNodes[i].visits + exploration_term - visit_penalty)
            w.append(self.childNodes[i].wcal(pareto_front,ucb))
        m = np.amax(w)
        indices = np.nonzero(w == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]

        return s

    def wcal(self,pareto,ucb):
        dominated = pareto.Dominated(self.wins)
        if dominated:
            delta_hv = self._relaxed_hv_delta(pareto, ucb)
            return delta_hv
        else:
            return self.get_cached_hv(pareto)
    
    def _relaxed_hv_delta(self, pareto, ucb):
        """Calculate relaxed HV delta for dominated points using ε-relaxation."""
        try:
            front = pareto.front
            if len(front) == 0:
                return 0.0

            eps = _estimate_hv_epsilon(front)
            if eps <= 0:
                return 0.0

            # Convert Pareto front to minimization form required by HV
            dominated_front = []
            for point in front:
                dominated_front.append([-v if v > 0 else -1e-17 for v in point])
            dominated_front.append([-v if v > 0 else -1e-17 for v in ucb])

            # Difference between full HV and HV without ucb
            full_hv = float(hypervolume(dominated_front).compute([0] * len(ucb)))
            no_candidate_hv = float(hypervolume(dominated_front[:-1]).compute([0] * len(ucb)))
            relaxed = min((full_hv - no_candidate_hv) / eps, full_hv)
            return max(relaxed, 0.0)
        except Exception:
            return 0.0


    


    def hvcal(self,pareto,ucb):
        if len(pareto.front) == 0:
            return 0
        _pareto_temp = copy.deepcopy(pareto.front)
        _pareto_temp.append(ucb)
        for i in range(len(_pareto_temp)):
            for j in range(len(_pareto_temp[0])):
                if(_pareto_temp[i][j]>0):
                    _pareto_temp[i][j] = -_pareto_temp[i][j]
                else:
                    _pareto_temp[i][j] = -0.00000000000000001
        hv = hypervolume(_pareto_temp)
        # Dual target: 6-dimensional Pareto front [EGFR_docking, 3PP0_docking, QED, IER, SA_score, ACS]
        ref_point = [0,0,0,0,0,0]
        hvnum = 0
        try:
            hvnum = hv.compute(ref_point)
        except:
            f = open("./data/present/hverror_output.txt", 'a')
            print(time.asctime( time.localtime(time.time()) ),file=f)
            print(pareto.front,file=f)
            f.close()
        return hvnum

    def Addnode(self, m, s):

        n = Node(position = m, parent = self, state = s)
        if not n in self.childNodes:
            self.childNodes.append(n)

    def simulation(self,state):
        predicted_smile=predict_smile(model,state)
        input_smile=make_input_smile(predicted_smile)
        logp = []
        valid_smile = input_smile
        all_smile = input_smile

        return logp,valid_smile,all_smile

    def preprocess_todict(self):
        self.parentNode = None
        for cn in self.childNodes:
            print(cn)
            print(cn.depth)
            print(cn.childNodes)
            if self != cn:
                cn.preprocess_todict()
            else:
                self.childNodes.remove(cn)
        return self

    def preprocess_fromdict(self):
        for cn in self.childNodes:
            cn.parentNode = self
            cn.preprocess_fromdict()
    

    def Update(self, result):

        self.visits += 1
        for i in range(len(self.wins)):
            self.wins[i]+=result[i]
        return self

    @staticmethod
    def from_dict(_filename):
        #must check if _filename file exists
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_root = Node(position =_set_json['position'], parentNode=None, childNodes=None, child=_set_json['child'], visits=_set_json['visits'], nonvisited_atom=_set_json['nonvisited_atom'], type_node=_set_json['type_node'], depth=_set_json['depth'])
        _set_file.close()
        print("Loaded Node")
        return new_root

def _sigmoidnormalize(score:float)-> float:
    """
    Normalize docking score to [0,1] range using sigmoid function
    """
    threshold = -5
    normalized = 1 / (1+ np.exp(score - threshold))
    return round(normalized, 3)

def _linearnormalize(score: float)-> float:
    """
    Linear normalization of docking score to [0,1] range
    Lower (more negative) docking scores are better
    """
    # Assume docking score range is roughly [-15, 0], map to [0,1]
    min_score = -15
    max_score = 0
    normalized = (score - min_score) / (max_score - min_score)
    normalized = max(0, min(1, normalized))  # Ensure within [0,1] range
    return round(normalized, 3)

def _sbmolgennormalize(score:float)-> float:
    """
    Normalize docking score to [0,1] range
    Lower (more negative) docking scores are better, convert to [0,1] range
    """
    base_dock_score = 0
    # Raw formula may return negative values, need to map to [0,1]
    raw_score = -((score - base_dock_score)*0.1)/(1+abs((score - base_dock_score)*0.1))
    # Use sigmoid function to map any range to [0,1]
    normalized = 1 / (1 + np.exp(-raw_score))
    return round(normalized, 3)

def _sa_score_normalize(score:float)-> float:
    """SA score normalization: lower SA score is better, range 1-10, convert to 0-1"""
    return round(1 - score/10, 3)

def _logp_normalize(score:float)-> float:
    logp_center = 2.5
    return round(1 - pow((0.5*(score-logp_center)),2), 3)

def _estimate_hv_epsilon(front):
    """Estimate ε-relaxation scale from Pareto front IQR across dimensions."""
    try:
        front = np.asarray(front, dtype=float)
        if front.ndim != 2 or front.shape[0] < 2:
            return 0.0

        q75, q25 = np.percentile(front, [75, 25], axis=0)
        iqr = q75 - q25
        return max(float(np.mean(iqr)) * 0.1, 0.0)
    except Exception:
        return 0.0

def _calculate_acs(docking_score_egfr: float, docking_score_3pp0: float, cliff_detector, smiles: str, compound_scores=None) -> float:
    """
    Calculate ACS (Activity Cliff Score) for dual-target molecule generation.
    Uses average of EGFR and 3PP0 docking scores.
    
    Args:
        docking_score_egfr: EGFR docking score (original/EGFR target)
        docking_score_3pp0: 3PP0 docking score (secondary target)
        cliff_detector: Activity cliff detector instance
        smiles: SMILES string of the molecule
        compound_scores: Full score vector [EGFR, 3PP0, QED, IER, SA, ACS]
    
    Returns:
        ACS score normalized to [0,1] range
    """
    # Use average of two docking scores for dual-target optimization
    avg_docking_score = (docking_score_egfr + docking_score_3pp0) / 2.0
    normalized_docking = _sbmolgennormalize(avg_docking_score)
    
    if cliff_detector is None or DISABLE_ACR_COMPONENT:
        return normalized_docking
    
    # Create averaged compound scores for ACR calculation
    if compound_scores is not None and len(compound_scores) >= 2:
        avg_scores_for_acr = [avg_docking_score] + list(compound_scores[1:])
    else:
        avg_scores_for_acr = [avg_docking_score]
    
    acr = _calculate_acr(smiles, cliff_detector, compound_scores=avg_scores_for_acr)
    normalized_acr = (acr + 1) / 2
    acs = ACS_DOCKING_WEIGHT * normalized_docking + ACS_ACR_WEIGHT * normalized_acr
    acs = max(0, min(1, acs))
    return round(acs, 3)

def _calculate_acr(smiles: str, cliff_detector, compound_scores=None) -> float:
    if cliff_detector is None:
        return 0.0
    
    compound_scores = compound_scores or [None]
    docking_score = compound_scores[0] if isinstance(compound_scores, list) and len(compound_scores) > 0 else compound_scores
    
    if docking_score is None or docking_score < -20:
        return 0.0
    
    try:
        cliff_neighbors = cliff_detector.get_high_activity_cliff_neighbors(
            [smiles],
            [compound_scores],
            beta1=CLIFF_ALPHA1,
            beta2=CLIFF_ALPHA2
        )
    except Exception as e:
        print(f"Error in _calculate_acr: {e}")
        return 0.0
    
    if not cliff_neighbors:
        return 0.0
    
    neighbors = cliff_neighbors[0]['neighbors']
    if not neighbors:
        return 0.0
    
    acr_values = []
    for neighbor in neighbors:
        delta = neighbor['delta']
        beta1 = CLIFF_ALPHA1
        term = -delta / beta1
        acr_values.append(term)
    
    acr = sum(acr_values) / len(acr_values)
    return float(np.clip(acr, -1.0, 1.0))

def MCTS(root, pareto=pareto(), budget=3600*240, CostPerMolecule=False, enable_activity_cliff=True, enable_ier=True, max_docking_calls=20000):
    global DISABLE_ACR_COMPONENT, ACS_DOCKING_WEIGHT, ACS_ACR_WEIGHT, CLIFF_ALPHA1, CLIFF_ALPHA2
    global UCB_EXPLORATION_CONSTANT, VISIT_PENALTY_COEFF
    global IER_ALPHA, IER_GAMMA, IER_K_NEIGHBORS, IER_H_MAX, IER_TAU_REP
    global REWARD
    REWARD = "normal"  # Default value, will be overwritten if config exists
    total_cost = 0
    total_docking_calls = 0
    rootnode = Node(state = root)
    state = root.Clone()
    
    print(f"Search termination condition: Vina docking calls reach {max_docking_calls}")
    
    if enable_activity_cliff and pareto.cliff_detector is None:
        pareto.cliff_detector = ActivityCliffDetector(alpha1=CLIFF_ALPHA1, alpha2=CLIFF_ALPHA2, max_memory_size=1000)
        print("Activity cliff detector initialized")
    
    if enable_ier and pareto.ier_evaluator is None:
        from ligand_design.ier_evaluator import IEREvaluator
        pareto.ier_evaluator = IEREvaluator(
            pareto_pool=pareto.compounds,
            alpha=pareto.ier_alpha,
            gamma=pareto.ier_gamma,
            k_neighbors=pareto.ier_k_neighbors,
            h_max=pareto.ier_h_max,
            tau_rep=pareto.ier_tau_rep
        )
        print(f"IER evaluator initialized: alpha={pareto.ier_alpha}, gamma={pareto.ier_gamma}, "
              f"k={pareto.ier_k_neighbors}, H_max={pareto.ier_h_max}, tau_rep={pareto.ier_tau_rep}")

    valid_compound=[]
    all_simulated_compound=[]
    desired_compound=[]
    desired_activity=[]
    depth=[]
    min_score=1000
    score_distribution=[]
    min_score_distribution=[]
    dock_score=[]
    sascore=[]
    qedscore=[]
    default_reward = [[0,0,0,0,0,0]]
    # Dual target: 6-dimensional penalty reward
    penalty_reward = [-1., -1., -1., -1., -1., -1.]
    
    if os.path.exists(dataDir+'/input/python_config.json') :
        config = json.load(open(dataDir+'/input/python_config.json'))
        REWARD = config['reward']
        DISABLE_ACR_COMPONENT = config.get('disable_acr_component', DISABLE_ACR_COMPONENT)
        ACS_DOCKING_WEIGHT = config.get('acs_docking_weight', ACS_DOCKING_WEIGHT)
        ACS_ACR_WEIGHT = config.get('acs_acr_weight', ACS_ACR_WEIGHT)
        CLIFF_ALPHA1 = config.get('activity_cliff_alpha1', CLIFF_ALPHA1)
        CLIFF_ALPHA2 = config.get('activity_cliff_alpha2', CLIFF_ALPHA2)
        UCB_EXPLORATION_CONSTANT = config.get('ucb_exploration_constant', UCB_EXPLORATION_CONSTANT)
        VISIT_PENALTY_COEFF = config.get('visit_penalty_coefficient', VISIT_PENALTY_COEFF)

        IER_ALPHA = config.get('ier_alpha', IER_ALPHA)
        IER_GAMMA = config.get('ier_gamma', IER_GAMMA)
        IER_K_NEIGHBORS = config.get('ier_k_neighbors', IER_K_NEIGHBORS)
        IER_H_MAX = config.get('ier_h_max', IER_H_MAX)
        IER_TAU_REP = config.get('ier_tau_rep', IER_TAU_REP)

        if hasattr(pareto, 'ier_evaluator') and pareto.ier_evaluator is not None:
            pareto.ier_evaluator.set_parameters(
                alpha=IER_ALPHA,
                gamma=IER_GAMMA,
                k_neighbors=IER_K_NEIGHBORS,
                h_max=IER_H_MAX,
                tau_rep=IER_TAU_REP
            )

        pareto.ier_alpha = IER_ALPHA
        pareto.ier_gamma = IER_GAMMA
        pareto.ier_k_neighbors = IER_K_NEIGHBORS
        pareto.ier_h_max = IER_H_MAX
        pareto.ier_tau_rep = IER_TAU_REP

    mcts_start_time = time.time()

    # === IER Component Records ===
    ier_history = {
        'iterations': [],
        'timestamps': [],
        'num_pareto': [],
        'num_history': [],
        'Nov_P_mean': [],
        'Nov_H_mean': [],
        'Rep_mean': [],
        'IER_mean': [],
        'Nov_P_std': [],
        'Nov_H_std': [],
        'Rep_std': [],
        'IER_std': [],
        'pareto_pool_size': [],
        'history_pool_size': [],
        'active_history_size': []
    }

    def _record_ier_snapshot():
        """Record IER component snapshot"""
        if pareto.ier_evaluator is None:
            return

        try:
            stats = pareto.ier_evaluator.get_pool_stats()
            eval_stats = stats.get('evaluation_stats', {})

            ier_history['iterations'].append(_iter_count)
            ier_history['timestamps'].append(time.asctime(time.localtime(time.time())))
            ier_history['num_pareto'].append(stats.get('pareto_pool_size', 0))
            ier_history['num_history'].append(stats.get('history_pool_size', 0))
            ier_history['Nov_P_mean'].append(eval_stats.get('avg_pareto_novelty', 0.0))
            ier_history['Nov_H_mean'].append(eval_stats.get('avg_history_novelty', 0.0))
            ier_history['Rep_mean'].append(eval_stats.get('avg_repetition', 0.0))
            ier_history['IER_mean'].append(eval_stats.get('avg_ier', 0.0))
            ier_history['Nov_P_std'].append(0.0)
            ier_history['Nov_H_std'].append(0.0)
            ier_history['Rep_std'].append(0.0)
            ier_history['IER_std'].append(0.0)
            ier_history['pareto_pool_size'].append(stats.get('pareto_pool_size', 0))
            ier_history['active_history_size'].append(stats.get('active_history_size', 0))

            # Write to CSV
            ier_csv_path = os.path.join(dataDir, 'present', 'ier_history.csv')
            write_header = not os.path.exists(ier_csv_path)
            with open(ier_csv_path, 'a') as f:
                if write_header:
                    f.write('iteration,timestamp,num_pareto,num_history,Nov_P_mean,Nov_H_mean,Rep_mean,IER_mean,pareto_pool_size,active_history_size\n')
                f.write(f"{_iter_count},{time.time()},{stats.get('pareto_pool_size', 0)},{stats.get('history_pool_size', 0)},"
                        f"{eval_stats.get('avg_pareto_novelty', 0.0):.6f},{eval_stats.get('avg_history_novelty', 0.0):.6f},"
                        f"{eval_stats.get('avg_repetition', 0.0):.6f},{eval_stats.get('avg_ier', 0.0):.6f},"
                        f"{stats.get('pareto_pool_size', 0)},{stats.get('active_history_size', 0)}\n")

            # Write to JSON
            ier_json_path = os.path.join(dataDir, 'present', 'ier_history.json')
            with open(ier_json_path, 'w') as f:
                json.dump(ier_history, f, indent=2)

        except Exception as e:
            print(f"Error recording IER snapshot: {e}")

    def _write_metrics_snapshot():
        """Write a metric snapshot to present/metrics.(json|csv) based on current state"""
        try:
            # Calculate hypervolume (based on current Pareto front)
            hv_value = 0.0
            if len(pareto.front) > 0:
                _front = copy.deepcopy(pareto.front)
                for i in range(len(_front)):
                    for j in range(len(_front[0])):
                        if _front[i][j] > 0:
                            _front[i][j] = -_front[i][j]
                        else:
                            _front[i][j] = -1e-17
                try:
                    hv = hypervolume(_front)
                    ref_point = [0] * len(_front[0])
                    hv_value = float(hv.compute(ref_point))
                except Exception as _:
                    hv_value = 0.0

            # Number of generated molecules (based on Pareto set)
            num_generated = len(pareto.compounds)

            # Calculate total elapsed time and efficiency
            total_elapsed_sec = total_cost if CostPerMolecule == 0 else (time.time() - mcts_start_time)
            molecules_per_hour = (num_generated / (total_elapsed_sec / 3600.0)) if total_elapsed_sec > 0 else 0.0

            # Calculate molecular diversity and novelty (based on mean pairwise Tanimoto distance)
            diversity_mean = 0.0
            diversity_count = 0
            try:
                from rdkit import Chem, DataStructs
                from rdkit.Chem import AllChem
                smiles_list = list(dict.fromkeys(pareto.compounds))  # Deduplicate while preserving order
                fps = []
                for s in smiles_list:
                    mol = Chem.MolFromSmiles(s)
                    if mol is None:
                        fps.append(None)
                    else:
                        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
                dist_sum = 0.0
                for i in range(len(fps)):
                    if fps[i] is None:
                        continue
                    for j in range(i + 1, len(fps)):
                        if fps[j] is None:
                            continue
                        sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                        dist = 1.0 - sim
                        dist_sum += dist
                        diversity_count += 1
                if diversity_count > 0:
                    diversity_mean = dist_sum / diversity_count
            except Exception as _:
                diversity_mean = 0.0

            novelty_mean = diversity_mean

            # Pareto front coverage (optional): calculate if baseline file present/pareto_baseline.json exists
            coverage = None
            try:
                baseline_path = os.path.join(dataDir, 'present', 'pareto_baseline.json')
                if os.path.exists(baseline_path):
                    with open(baseline_path, 'r') as bf:
                        baseline = json.load(bf)
                    baseline_front = baseline.get('front', [])
                    cur_front = pareto.front

                    def dominated_by_cur(r):
                        for c in cur_front:
                            ge_all = True
                            gt_any = False
                            for a, b in zip(c, r):
                                if a < b:
                                    ge_all = False
                                    break
                                if a > b:
                                    gt_any = True
                            if ge_all and gt_any:
                                return True
                        return False

                    if len(baseline_front) > 0:
                        dominated_num = sum(1 for r in baseline_front if dominated_by_cur(r))
                        coverage = dominated_num / float(len(baseline_front))
            except Exception as _:
                coverage = None

            # Assemble and write
            metrics = {
                'timestamp': time.asctime(time.localtime(time.time())),
                'num_generated_molecules': num_generated,
                'total_docking_calls': total_docking_calls,
                'hypervolume': hv_value,
                'total_elapsed_seconds': total_elapsed_sec,
                'molecules_per_hour': molecules_per_hour,
                'diversity_mean_tanimoto_distance': diversity_mean,
                'novelty_mean_tanimoto_distance': novelty_mean,
                'pareto_coverage_against_baseline': coverage
            }

            os.makedirs(os.path.join(dataDir, 'present'), exist_ok=True)

            # JSON (overwrite current snapshot)
            try:
                with open(os.path.join(dataDir, 'present', 'metrics.json'), 'w') as jf:
                    json.dump(metrics, jf, indent=2)
            except Exception as e_json1:
                # Fallback: string concatenation path
                try:
                    with open(dataDir + 'present/metrics.json', 'w') as jf2:
                        json.dump(metrics, jf2, indent=2)
                except Exception as e_json2:
                    with open(os.path.join(dataDir, 'present', 'metrics_error.txt'), 'a') as ef:
                        ef.write(f"{time.asctime(time.localtime(time.time()))} Write metrics.json failed: {e_json1} | Fallback also failed: {e_json2}\n")

            # Append to CSV (write header if not exists)
            csv_path = os.path.join(dataDir, 'present', 'metrics.csv')
            header = ['timestamp', 'num_generated_molecules', 'hypervolume', 'total_elapsed_seconds', 'molecules_per_hour', 'diversity_mean_tanimoto_distance', 'novelty_mean_tanimoto_distance', 'pareto_coverage_against_baseline']
            line = [
                metrics['timestamp'], num_generated, hv_value, total_elapsed_sec, molecules_per_hour,
                diversity_mean, novelty_mean, ('' if coverage is None else coverage)
            ]
            write_header = not os.path.exists(csv_path)
            try:
                with open(csv_path, 'a') as cf:
                    if write_header:
                        cf.write(','.join(header) + '\n')
                    cf.write(','.join(str(x) for x in line) + '\n')
            except Exception as e_csv1:
                # Fallback: string concatenation path
                try:
                    csv_path2 = dataDir + 'present/metrics.csv'
                    write_header2 = not os.path.exists(csv_path2)
                    with open(csv_path2, 'a') as cf2:
                        if write_header2:
                            cf2.write(','.join(header) + '\n')
                        cf2.write(','.join(str(x) for x in line) + '\n')
                except Exception as e_csv2:
                    with open(os.path.join(dataDir, 'present', 'metrics_error.txt'), 'a') as ef:
                        ef.write(f"{time.asctime(time.localtime(time.time()))} Write metrics.csv failed: {e_csv1} | Fallback also failed: {e_csv2}\n")
        except Exception as e:
            print(f"Error writing metric snapshot: {e}")

    # Iteration counter for controlling metrics write frequency
    _iter_count = 0

    while total_docking_calls < max_docking_calls:
        start_time = time.time()
        node = rootnode
        state = root.Clone()
        node_pool=[]
        
        for step in range(MAX_SELECTION_STEPS):
            if node.position == '\n' or not node.childNodes:
                break
            if len(state.position) >= MAX_SMILES_LENGTH:
                break
            new_node = node.Selectnode(pareto)
            if new_node == node:
                node = new_node
                break
            node = new_node
            state.SelectPosition(node.position)
        
        if node.position == '\n':
            print("end with \\n")
            while node != None:
                node.Update(penalty_reward)
                node = node.parentNode
            continue
        if len(state.position)>= 81:
            print("position bigger than 81")
            while node != None:
                node.Update(penalty_reward)
                node = node.parentNode
            continue
        
        expanded=expanded_node(model,state.position,val)
        nodeadded=node_to_add(expanded,val)
        all_posible=chem_kn_simulation(model,state.position,val,nodeadded)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)


        node_index,scores,valid_smile, docking_count=check_node_type(new_compound,dataDir)
        total_docking_calls += docking_count
        f = open(dataDir+"present/ligands.txt", 'a')
        for p in valid_smile:
            print(p,file=f)
        f.close()

        if len(node_index)==0:
            while node != None:
                node.Update(default_reward[0])
                node = node.parentNode
            continue
        re=[]
        for i in range(len(node_index)):
            m=node_index[i]
            newflag = True
            for j in range(len(node.childNodes)):
                if(node.childNodes[j].position == nodeadded[m]):
                    newflag = False
                    node_pool.append(node.childNodes[j])
            if newflag:
                node.Addnode(nodeadded[m],state)
                if len(node.childNodes) >0:
                    node_pool.append(node.childNodes[-1])
            
            f = open(dataDir+"present/depth.txt", 'a')
            print(len(state.position),file=f)
            base_dock_score = 0
            # Dual target: Normalize both EGFR and 3PP0 docking scores
            if REWARD == "normal":
                    scores[i][0]= _sbmolgennormalize(scores[i][0])  # EGFR docking score
                    scores[i][1]= _sbmolgennormalize(scores[i][1])  # 3PP0 docking score
            elif REWARD == "sigmoid":
                    scores[i][0]= _sigmoidnormalize(scores[i][0])
                    scores[i][1]= _sigmoidnormalize(scores[i][1])
            elif REWARD == "nonormal":
                    scores[i][0]= _linearnormalize(scores[i][0])
                    scores[i][1]= _linearnormalize(scores[i][1])
            
            scores[i][4] = _sa_score_normalize(scores[i][4])  # SA score
            scores[i][5] = 0.0  # ACS placeholder (calculated later in pareto.Update)

            if pareto.ier_evaluator is not None:
                ier_components = pareto.ier_evaluator.calculate_ier_components(valid_smile[i])
                scores[i][3] = ier_components['IER']  # IER at position 3
                print(f"IER calculation: Nov_P={ier_components['Nov_P']:.3f}, "
                      f"Nov_H={ier_components['Nov_H']:.3f}, "
                      f"Rep={ier_components['Rep']:.3f}, "
                      f"IER={scores[i][3]:.3f}")
                pareto.ier_evaluator.update_pools(pareto_pool=pareto.compounds + [valid_smile[i]])
                pareto.ier_evaluator.add_to_history(valid_smile[i])

            if pareto.Dominated(scores[i]) == False:
                pareto.Update(scores[i],valid_smile[i])
                print("Time: ",time.asctime( time.localtime(time.time()) ))

            re.append(scores[i])

        f = open(dataDir+"present/scores.txt", 'a')
        for s in scores:
            print(s, file=f)
        f.close()

        for i in range(len(node_pool)):
            node=node_pool[i]
            while node != None:
                node.Update(re[i])
                node = node.parentNode

        print("End Search Epoch: ", time.asctime( time.localtime(time.time()) ))
        pareto_file = open(dataDir+'present/pareto.json', 'w')
        json.dump(pareto.to_dict(), pareto_file, indent=4, separators=(',', ': '))
        pareto_file.close()

        if CostPerMolecule==0:
            cost = time.time() - start_time
        else:
            cost = CostPerMolecule * len(valid_smile)
        total_cost += cost
        print(f"Docking: {total_docking_calls}/{max_docking_calls}")
        
        if enable_activity_cliff and pareto.cliff_detector is not None:
            cliff_stats = pareto.get_cliff_statistics()
            print(f"Activity cliff statistics: High-activity molecules={cliff_stats.get('high_activity_count', 0)}, "
                  f"Cliff molecules={cliff_stats.get('cliff_count', 0)}, "
                  f"Avg high-activity score={cliff_stats.get('avg_high_activity_score', 0):.3f}")
            pareto._save_cliff_statistics(cliff_stats)

        _iter_count += 1
        if _iter_count % 5 == 0:
            _write_metrics_snapshot()
            if enable_ier and pareto.ier_evaluator is not None:
                _record_ier_snapshot()

    print("pareto front",pareto.compounds)
    print("pareto front scores",pareto.front)
    print(f"Total molecular docking calls: {total_docking_calls}")
    
    if enable_activity_cliff and pareto.cliff_detector is not None:
        cliff_stats = pareto.get_cliff_statistics()
        print(f"Final activity cliff statistics: High-activity molecules={cliff_stats.get('high_activity_count', 0)}, "
              f"Cliff molecules={cliff_stats.get('cliff_count', 0)}")
        
        # Save final statistics
        pareto._save_cliff_statistics(cliff_stats)
        
        # Save activity cliff memory
        pareto.cliff_detector.save_memory(dataDir + 'present/activity_cliff_memory.json')
        print("Activity cliff memory saved")
        
        # Generate final cliff pair summary report
        pareto._generate_final_cliff_report()


    # ============== Metrics statistics and writing (final write once more) ==============
    try:
        # Calculate hypervolume (based on current Pareto front)
        hv_value = 0.0
        if len(pareto.front) > 0:
            # Convert front copy to pygmo hypervolume expected "maximization to minimization" form
            _front = copy.deepcopy(pareto.front)
            for i in range(len(_front)):
                for j in range(len(_front[0])):
                    if _front[i][j] > 0:
                        _front[i][j] = -_front[i][j]
                    else:
                        _front[i][j] = -1e-17
            try:
                hv = hypervolume(_front)
                ref_point = [0] * len(_front[0])
                hv_value = float(hv.compute(ref_point))
            except Exception as _:
                hv_value = 0.0

        # Number of generated molecules (based on Pareto set)
        num_generated = len(pareto.compounds)

        # Calculate total elapsed time and efficiency
        total_elapsed_sec = total_cost if CostPerMolecule == 0 else (time.time() - mcts_start_time)
        molecules_per_hour = (num_generated / (total_elapsed_sec / 3600.0)) if total_elapsed_sec > 0 else 0.0

        # Calculate molecular diversity and novelty (based on mean pairwise Tanimoto distance)
        diversity_mean = 0.0
        diversity_count = 0
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            smiles_list = list(dict.fromkeys(pareto.compounds))  # Deduplicate while preserving order
            fps = []
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    fps.append(None)
                else:
                    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            dist_sum = 0.0
            for i in range(len(fps)):
                if fps[i] is None:
                    continue
                for j in range(i + 1, len(fps)):
                    if fps[j] is None:
                        continue
                    sim = DataStructs.FingerprintSimilarity(fps[i], fps[j])
                    dist = 1.0 - sim
                    dist_sum += dist
                    diversity_count += 1
            if diversity_count > 0:
                diversity_mean = dist_sum / diversity_count
        except Exception as _:
            diversity_mean = 0.0

        # As a simple proxy for "novelty": average distance to current set (same set-level metric as diversity)
        novelty_mean = diversity_mean

        # Pareto front coverage (optional): calculate if baseline file present/pareto_baseline.json exists
        coverage = None
        try:
            baseline_path = os.path.join(dataDir, 'present', 'pareto_baseline.json')
            if os.path.exists(baseline_path):
                with open(baseline_path, 'r') as bf:
                    baseline = json.load(bf)
                baseline_front = baseline.get('front', [])
                cur_front = pareto.front

                def dominated_by_cur(r):
                    for c in cur_front:
                        ge_all = True
                        gt_any = False
                        for a, b in zip(c, r):
                            if a < b:
                                ge_all = False
                                break
                            if a > b:
                                gt_any = True
                            if ge_all and gt_any:
                                return True
                    return False

                if len(baseline_front) > 0:
                    dominated_num = sum(1 for r in baseline_front if dominated_by_cur(r))
                    coverage = dominated_num / float(len(baseline_front))
        except Exception as _:
            coverage = None

        # Assemble and write
        metrics = {
            'timestamp': time.asctime(time.localtime(time.time())),
            'num_generated_molecules': num_generated,
            'total_docking_calls': total_docking_calls,
            'hypervolume': hv_value,
            'total_elapsed_seconds': total_elapsed_sec,
            'molecules_per_hour': molecules_per_hour,
            'diversity_mean_tanimoto_distance': diversity_mean,
            'novelty_mean_tanimoto_distance': novelty_mean,
            'pareto_coverage_against_baseline': coverage
        }

        os.makedirs(os.path.join(dataDir, 'present'), exist_ok=True)

        # JSON
        try:
            with open(os.path.join(dataDir, 'present', 'metrics.json'), 'w') as jf:
                json.dump(metrics, jf, indent=2)
        except Exception as e_json1:
            try:
                with open(dataDir + 'present/metrics.json', 'w') as jf2:
                    json.dump(metrics, jf2, indent=2)
            except Exception as e_json2:
                with open(os.path.join(dataDir, 'present', 'metrics_error.txt'), 'a') as ef:
                    ef.write(f"{time.asctime(time.localtime(time.time()))} Final write metrics.json failed: {e_json1} | Fallback also failed: {e_json2}\n")

        # Append to CSV (write header if not exists)
        csv_path = os.path.join(dataDir, 'present', 'metrics.csv')
        header = ['timestamp', 'num_generated_molecules', 'hypervolume', 'total_elapsed_seconds', 'molecules_per_hour', 'diversity_mean_tanimoto_distance', 'novelty_mean_tanimoto_distance', 'pareto_coverage_against_baseline']
        line = [
            metrics['timestamp'], num_generated, hv_value, total_elapsed_sec, molecules_per_hour,
            diversity_mean, novelty_mean, ('' if coverage is None else coverage)
        ]
        write_header = not os.path.exists(csv_path)
        try:
            with open(csv_path, 'a') as cf:
                if write_header:
                    cf.write(','.join(header) + '\n')
                cf.write(','.join(str(x) for x in line) + '\n')
        except Exception as e_csv1:
            try:
                csv_path2 = dataDir + 'present/metrics.csv'
                write_header2 = not os.path.exists(csv_path2)
                with open(csv_path2, 'a') as cf2:
                    if write_header2:
                        cf2.write(','.join(header) + '\n')
                    cf2.write(','.join(str(x) for x in line) + '\n')
            except Exception as e_csv2:
                with open(os.path.join(dataDir, 'present', 'metrics_error.txt'), 'a') as ef:
                    ef.write(f"{time.asctime(time.localtime(time.time()))} Final write metrics.csv failed: {e_csv1} | Fallback also failed: {e_csv2}\n")
        print(f"Metrics written: {csv_path}")
    except Exception as e:
        print(f"Error writing metrics: {e}")

    return valid_compound


def UCTchemical(budget=3600*240, CostPerMolecule=False, enable_activity_cliff=True, enable_ier=True, max_docking_calls=20000):
    state = chemical()
    pareto_front = pareto() if isLoadTree is False else pareto.from_dict(pareto_locate)
    best = MCTS(root = state,pareto=pareto_front, budget = budget, CostPerMolecule = CostPerMolecule, enable_activity_cliff=enable_activity_cliff, enable_ier=enable_ier, max_docking_calls=max_docking_calls)


    return best


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='search molecular')

    parser.add_argument('dataDir',help='path to data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    if os.path.exists(dataDir+'/input/python_config.json') :
        config = json.load(open(dataDir+'input/python_config.json'))
        isLoadTree = config['isLoadTree']
        pareto_locate = dataDir+'present/pareto.json'
        budget = config.get('max_docking_calls', 20000)
        CostPerMolecule = config['CostPerMolecule']
        rnnModelDir = config['whereisRNNmodelDir']
        rnnModelFile = config.get('rnnModelFile', 'model.h5')
        randomSeed = config.get('randomSeed', 42)
        
        # Set random seed
        set_random_seed(randomSeed)
        print(f"Random seed set to: {randomSeed}")
        print(f"RNN Model: {rnnModelDir}{rnnModelFile}")
    else :
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),dataDir+'/input/python_config.json')
    smile_old=zinc_data_with_bracket_original()
    val,smile=zinc_processed_with_bracket(smile_old)
    model=loaded_model(rnnModelDir, rnnModelFile)
    enable_ier = config.get('enable_ier', True)
    enable_activity_cliff = config.get('enable_activity_cliff', True)
    max_docking_calls = config.get('max_docking_calls', 20000)
    
    valid_compound=UCTchemical(budget=budget, CostPerMolecule=CostPerMolecule, 
                              enable_activity_cliff=enable_activity_cliff, enable_ier=enable_ier,
                              max_docking_calls=max_docking_calls)
