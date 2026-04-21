from math import *
import random
import random as pr
import numpy as np
from copy import deepcopy
import time
import argparse

from load_model import loaded_model
from make_smile import zinc_data_with_bracket_original, zinc_processed_with_bracket
from add_node_type import chem_kn_simulation, make_input_smile,predict_smile,check_node_type,node_to_add,expanded_node
from activity_cliff import ActivityCliffDetector
import copy
import os
import json
import errno
import argparse
from joblib import Parallel, delayed
import pdb

def _hv_prepare_points_for_minimization(front):
    pts = copy.deepcopy(front)
    if len(pts) == 0:
        return pts
    for i in range(len(pts)):
        for j in range(len(pts[0])):
            if pts[i][j] > 0:
                pts[i][j] = -pts[i][j]
            else:
                pts[i][j] = -1e-17
    return pts

def _hv_filter_nondominated_min(points):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts
    if pts.ndim != 2:
        pts = pts.reshape((-1, 1))
    pts = np.unique(pts, axis=0)
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        pi = pts[i]
        dominated_by = np.all(pts <= pi, axis=1) & np.any(pts < pi, axis=1)
        dominated_by[i] = False
        if np.any(dominated_by):
            keep[i] = False
    return pts[keep]

def _hv_2d_min(points, ref_point):
    pts = _hv_filter_nondominated_min(points)
    if pts.shape[0] == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    rx, ry = float(ref_point[0]), float(ref_point[1])
    hv = 0.0
    y_prev = ry
    for i in range(pts.shape[0]):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if y < y_prev:
            hv += max(0.0, rx - x) * (y_prev - y)
            y_prev = y
    return hv

def _hv_wfg_min(points, ref_point):
    pts = _hv_filter_nondominated_min(points)
    if pts.shape[0] == 0:
        return 0.0
    ref = np.asarray(ref_point, dtype=float).reshape((-1,))
    m = pts.shape[1]
    if m == 1:
        return float(ref[0] - np.min(pts[:, 0]))
    if m == 2:
        return _hv_2d_min(pts, ref)
    order = np.argsort(pts[:, m - 1])[::-1]
    pts = pts[order]
    hv = 0.0
    z_prev = float(ref[m - 1])
    for i in range(pts.shape[0]):
        z = float(pts[i, m - 1])
        if z < z_prev:
            slice_height = z_prev - z
            hv += slice_height * _hv_wfg_min(pts[i:, :m - 1], ref[:m - 1])
            z_prev = z
    return hv

def _hv_value_from_front_max(front_max):
    if len(front_max) == 0:
        return 0.0
    pts = _hv_prepare_points_for_minimization(front_max)
    ref_point = [0.0] * len(pts[0])
    return float(_hv_wfg_min(pts, ref_point))

def _hv_iqr_shift_from_front_max(front_max, beta):
    if len(front_max) == 0:
        return []
    b = float(beta)
    if b < 0.01:
        b = 0.01
    if b > 0.1:
        b = 0.1
    pts = np.asarray(front_max, dtype=float)
    if pts.ndim != 2:
        pts = pts.reshape((-1, 1))
    q25 = np.percentile(pts, 25, axis=0)
    q75 = np.percentile(pts, 75, axis=0)
    iqr = q75 - q25
    shift = iqr * b
    return shift.tolist()

def _hv_shift_point_max(point, shift):
    if len(shift) == 0:
        return list(point)
    out = []
    for v, s in zip(point, shift):
        vv = float(v) + float(s)
        if vv < 0.0:
            vv = 0.0
        if vv > 1.0:
            vv = 1.0
        out.append(vv)
    if len(point) > len(out):
        out.extend(point[len(out):])
    return out

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
        self.cliff_detector = cliff_detector
        self.ier_evaluator = ier_evaluator
        self.hv_cache_value = None
        self.hv_cache_size = -1
        self.hv_relax_shift_cache = None
        self.hv_relax_shift_cache_size = -1

    def get_hv_value(self):
        if self.hv_cache_value is None or self.hv_cache_size != len(self.front):
            self.hv_cache_value = _hv_value_from_front_max(self.front)
            self.hv_cache_size = len(self.front)
        return self.hv_cache_value

    def get_hv_relax_shift(self):
        if (self.hv_relax_shift_cache is None) or (self.hv_relax_shift_cache_size != len(self.front)):
            self.hv_relax_shift_cache = _hv_iqr_shift_from_front_max(self.front, HV_RELAX_IQR_SCALE)
            self.hv_relax_shift_cache_size = len(self.front)
        return self.hv_relax_shift_cache

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
        if self.ier_evaluator is not None:
            ier_value = self.ier_evaluator.calculate_ier(compound)
            
            if len(scores) > 2:
                scores[2] = ier_value
            else:
                while len(scores) < 5:
                    scores.append(0.0)
                scores[2] = ier_value
            
            self.ier_evaluator.update_pools(pareto_pool=self.compounds + [compound])
            self.ier_evaluator.add_to_history(compound)
        
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
        self.hv_cache_value = None
        self.hv_cache_size = -1
        self.hv_relax_shift_cache = None
        self.hv_relax_shift_cache_size = -1
        
        if self.cliff_detector is not None:
            self.cliff_detector.update_memory([compound], [scores])
            
            cliff_pairs = self.cliff_detector.detect_activity_cliffs([compound], [scores])
            if cliff_pairs:
               
                self.cliff_detector.add_cliff_pairs(cliff_pairs)
                
                f = open(dataDir+"output.txt", 'a')                 
                print("Activity Cliffs detected:", file=f)
                for pair in cliff_pairs:
                    print(f"  {pair['smiles1']} (score: {pair['scores1']:.3f}) -> {pair['smiles2']} (score: {pair['scores2']:.3f}) ACI: {pair['ACI']:.3f}", file=f)
                f.close()
                
                self._save_cliff_pairs_detailed(cliff_pairs, compound, scores)
            
            if len(scores) >= 5:
                normalized_docking = _sbmolgennormalize(scores[0])
                if DISABLE_ACR_COMPONENT:
                    scores[4] = normalized_docking
                else:
                    acr = _calculate_acr(compound, self.cliff_detector)
                    normalized_acr = (acr + 1) / 2
                    acs = ACS_DOCKING_WEIGHT * normalized_docking + ACS_ACR_WEIGHT * normalized_acr
                    acs = max(0, min(1, acs))
                    scores[4] = round(acs, 3)
                
        else:
            if len(scores) >= 5:
                scores[4] = _sbmolgennormalize(scores[0])
                
                f = open(dataDir+"output.txt", 'a') 
        
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
        if self.cliff_detector is None:
            return [], []
        return self.cliff_detector.get_cliff_molecules(n_samples)
    
    def get_high_activity_molecules(self, n_samples=20):
        if self.cliff_detector is None:
            return [], []
        return self.cliff_detector.get_high_activity_molecules(n_samples)
    
    def get_cliff_statistics(self):
        if self.cliff_detector is None:
            return {}
        return self.cliff_detector.get_statistics()
    
    def _save_cliff_pairs_detailed(self, cliff_pairs, compound, scores):
        import time
        import json
        
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
        
        cliff_file = dataDir + "activity_cliffs.json"
        try:
            if os.path.exists(cliff_file):
                with open(cliff_file, 'r') as f:
                    all_cliffs = json.load(f)
            else:
                all_cliffs = {'cliff_events': []}
            
            all_cliffs['cliff_events'].append(cliff_details)
            
            with open(cliff_file, 'w') as f:
                json.dump(all_cliffs, f, indent=2)
                
        except Exception as e:
            print(f"error: {e}")
        
        cliff_txt_file = dataDir + "activity_cliffs.txt"
        with open(cliff_txt_file, 'a') as f:
           
            
            for i, pair in enumerate(cliff_pairs, 1):
                
                f.write("-" * 50 + "\n")
            f.write("\n")
    
    def _save_cliff_statistics(self, cliff_stats):
        import time
        import json
        
        stats_info = {
            'timestamp': time.asctime(time.localtime(time.time())),
            'high_activity_count': cliff_stats.get('high_activity_count', 0),
            'cliff_count': cliff_stats.get('cliff_count', 0),
            'avg_high_activity_score': cliff_stats.get('avg_high_activity_score', 0),
            'avg_cliff_score': cliff_stats.get('avg_cliff_score', 0)
        }
        
        stats_file = dataDir + "present/cliff_statistics.json"
        try:
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    all_stats = json.load(f)
            else:
                all_stats = {'statistics_history': []}
            
            all_stats['statistics_history'].append(stats_info)
            
            with open(stats_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
                
        except Exception as e:
            
        
        #stats_txt_file = dataDir + "present/cliff_statistics.txt"
        #with open(stats_txt_file, 'a') as f:
            f.write(f"{time.asctime(time.localtime(time.time()))} | ")
    
    def _generate_final_cliff_report(self):
        import time
        import json
        
        if self.cliff_detector is None:
            return
        
        cliff_smiles, cliff_scores = self.cliff_detector.get_cliff_molecules(1000)
        high_activity_smiles, high_activity_scores = self.cliff_detector.get_high_activity_molecules(1000)
        
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
        
        report_file = dataDir + "final_cliff_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        txt_report_file = dataDir + "final_cliff_report.txt"
        with open(txt_report_file, 'w') as f:
            f.write("=" * 80 + "\n")

            f.write("=" * 80 + "\n")
           
            f.write("-" * 40 + "\n")
            for i, (smiles, scores) in enumerate(zip(high_activity_smiles, high_activity_scores), 1):
                f.write(f"{i:3d}. {smiles}\n")
                f.write(f"     score: {scores}\n")
            
            f.write("-" * 40 + "\n")
            for i, (smiles, scores) in enumerate(zip(cliff_smiles, cliff_scores), 1):
                f.write(f"{i:3d}. {smiles}\n")
                f.write(f"     score: {scores}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"score: {txt_report_file}")
    
    def __len__(self):
        return len(self.front)
    
    @staticmethod
    def from_dict(_filename):
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_pareto = pareto(front = _set_json['front'], size=_set_json['size'], avg=_set_json['avg'], compounds=_set_json['compounds'])
        _set_file.close()
        new_pareto.cliff_detector = ActivityCliffDetector(alpha1=0.5, alpha2=2.0, max_memory_size=1000)
        print("Loaded Pareto Fronts")
        return new_pareto
    
    def to_dict(self):
        return {
            'front': self.front,
            'size': self.size,
            'avg': self.avg,
            'compounds': self.compounds
        }

class Node:

    def __init__(self, position = None,  parent = None, state = None, childNodes=[], child=None, wins=[0,0,0,0,0], visits=0, nonvisited_atom=None, type_node= [], depth=0):
        self.position = position
        self.parentNode = parent
        self.childNodes = childNodes
        self.child=child
        self.wins = wins
        self.visits = visits
        self.nonvisited_atom=state.Getatom() if nonvisited_atom is None else nonvisited_atom
        self.type_node=type_node
        self.depth=depth

        self.cached_hv = None
        self.last_hv_update = 0

    def get_cached_hv(self, pareto_front):
        if self.cached_hv is None or self.last_hv_update < len(pareto_front):
           self.cached_hv = self.hvcal(pareto_front, self.wins)
           self.last_hv_update = len(pareto_front)
        return self.cached_hv

    def Selectnode(self,pareto_front):

        w=[]
        for i in range(len(self.childNodes)):
            ucb=[]
            for win in self.childNodes[i].wins:
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
        dominated = pareto.Dominated(ucb)
        if dominated and HV_RELAX_ENABLED:
            hv_front = pareto.get_hv_value()
            shift = pareto.get_hv_relax_shift()
            relaxed = _hv_shift_point_max(ucb, shift)
            hv_relaxed = _hv_value_from_front_max(pareto.front + [relaxed])
            soft = hv_relaxed - hv_front
            if soft < 0.0:
                soft = 0.0
            return hv_front + HV_RELAX_SOFT_WEIGHT * soft
        hv = self.get_cached_hv(pareto)
        if dominated:
            return hv - self.distance(pareto,ucb)
        return hv

    def distance(self, pareto, ucb):
        avg = pareto.avg
        distance = 0
        for i in range(len(avg)):
            distance += pow(avg[i]-ucb[i])

        return sqrt(distance)


    def hvcal(self,pareto,ucb):
        if len(pareto.front) == 0:
            return 0
        _pareto_temp = copy.deepcopy(pareto.front)
        _pareto_temp.append(ucb)
        hvnum = 0
        _pareto_temp = _hv_prepare_points_for_minimization(_pareto_temp)
        ref_point = [0.0] * len(_pareto_temp[0])
        try:
            hvnum = _hv_wfg_min(_pareto_temp, ref_point)
        except:
            f = open("./hverror_output.txt", 'a')
            print(time.asctime( time.localtime(time.time()) ),file=f)
            print(pareto.front,file=f)
            f.close()
        return hvnum

    def Addnode(self, m, s):

        n = Node(position = m, parent = self, state = s)
        if not n in self.childNodes:
            self.childNodes.append(n)
            pass

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
                pass
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
        _set_file = open(_filename,'r')
        _set_json = json.load(_set_file)
        new_root = Node(position =_set_json['position'], parentNode=None, childNodes=None, child=_set_json['child'], visits=_set_json['visits'], nonvisited_atom=_set_json['nonvisited_atom'], type_node=_set_json['type_node'], depth=_set_json['depth'])
        _set_file.close()
        while True:
            new_root.childNodes 
        print("Loaded Pareto Fronts")
        return new_pareto
        self.position = position
        self.parentNode = parent
        self.childNodes = childNodes
        self.child=child
        self.wins = wins
        self.visits = visits
        self.nonvisited_atom=state.Getatom() if nonvisited_atom is None else nonvisited_atom
        self.type_node=type_node
        self.depth=depth

def _sigmoidnormalize(score:float)-> float:
   
    normalized = 1 / (1+ np.exp(score - threshold))
    return round(normalized, 3)

def _linearnormalize(score: float)-> float:
   
    normalized = (score - min_score) / (max_score - min_score)
    normalized = max(0, min(1, normalized))  
    return round(normalized, 3)

def _sbmolgennormalize(score:float)-> float:

    base_dock_score = 0
    raw_score = -((score - base_dock_score)*0.1)/(1+abs((score - base_dock_score)*0.1))

    normalized = 1 / (1 + np.exp(-raw_score))
    return round(normalized, 3)

def _sa_score_normalize(score:float)-> float:
    return round(1 - score/10, 3)


def _calculate_acs(docking_score: float, cliff_detector, smiles: str) -> float:
    
    normalized_docking = _sbmolgennormalize(docking_score)
    
    if cliff_detector is None or DISABLE_ACR_COMPONENT:
        return normalized_docking
    
    acr = _calculate_acr(smiles, cliff_detector)
    normalized_acr = (acr + 1) / 2
    acs = ACS_DOCKING_WEIGHT * normalized_docking + ACS_ACR_WEIGHT * normalized_acr
    acs = max(0, min(1, acs))
    return round(acs, 3)

def _calculate_acr(smiles: str, cliff_detector) -> float:
   
    if cliff_detector is None or len(cliff_detector.cliff_memory) == 0:
        return 0.0
    
    cliff_molecules = cliff_detector.cliff_memory['smiles'].tolist()
    
    if smiles not in cliff_molecules:
        return 0.0
    
    molecule_row = cliff_detector.cliff_memory[cliff_detector.cliff_memory['smiles'] == smiles]
    if len(molecule_row) == 0:
        return 0.0


def MCTS(root, pareto=pareto(), budget=3600*240, CostPerMolecule=False, enable_activity_cliff=True, enable_ier=True):
   
    global DISABLE_ACR_COMPONENT, ACS_DOCKING_WEIGHT, ACS_ACR_WEIGHT, CLIFF_ALPHA1, CLIFF_ALPHA2
    global UCB_EXPLORATION_CONSTANT, VISIT_PENALTY_COEFF
    total_cost = 0
    rootnode = Node(state = root)
    state = root.Clone()
    
    if enable_activity_cliff and pareto.cliff_detector is None:
        pareto.cliff_detector = ActivityCliffDetector(alpha1=CLIFF_ALPHA1, alpha2=CLIFF_ALPHA2, max_memory_size=1000)
      
    if enable_ier and pareto.ier_evaluator is None:
        from ligand_design.ier_evaluator import IEREvaluator
        pareto.ier_evaluator = IEREvaluator(pareto_pool=pareto.compounds)
    
  
    valid_compound=[]

    default_reward = [[0,0,0,0,0]]
    penalty_reward = [-1. , -1. , -1. , -1. , -1.]
    

    if os.path.exists(dataDir+'config.json') :
        config = json.load(open(dataDir+'config.json'))
        REWARD = config['reward'] 
        
        DISABLE_ACR_COMPONENT = config.get('disable_acr_component', DISABLE_ACR_COMPONENT)
        ACS_DOCKING_WEIGHT = config.get('acs_docking_weight', ACS_DOCKING_WEIGHT)
        ACS_ACR_WEIGHT = config.get('acs_acr_weight', ACS_ACR_WEIGHT)
        CLIFF_ALPHA1 = config.get('activity_cliff_alpha1', CLIFF_ALPHA1)
        CLIFF_ALPHA2 = config.get('activity_cliff_alpha2', CLIFF_ALPHA2)
        
        UCB_EXPLORATION_CONSTANT = config.get('ucb_exploration_constant', UCB_EXPLORATION_CONSTANT)
        VISIT_PENALTY_COEFF = config.get('visit_penalty_coefficient', VISIT_PENALTY_COEFF)


    mcts_start_time = time.time()

    def _write_metrics_snapshot():
        try:
            
            hv_value = 0.0
            if len(pareto.front) > 0:
                _front = _hv_prepare_points_for_minimization(pareto.front)
                try:
                    ref_point = [0] * len(_front[0])
                    hv_value = float(_hv_wfg_min(_front, ref_point))
                except Exception as _:
                    hv_value = 0.0

           
            num_generated = len(pareto.compounds)

            
            total_elapsed_sec = total_cost if CostPerMolecule == 0 else (time.time() - mcts_start_time)
            molecules_per_hour = (num_generated / (total_elapsed_sec / 3600.0)) if total_elapsed_sec > 0 else 0.0

            
            diversity_mean = 0.0
            diversity_count = 0
            try:
                from rdkit import Chem, DataStructs
                from rdkit.Chem import AllChem
                smiles_list = list(dict.fromkeys(pareto.compounds))  
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
            
            coverage = None
            try:
                baseline_path = os.path.join(dataDir, 'pareto_baseline.json')   
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

                
            metrics = {
                'timestamp': time.asctime(time.localtime(time.time())),
                'num_generated_molecules': num_generated,
                'hypervolume': hv_value,
                'total_elapsed_seconds': total_elapsed_sec,
                'diversity_mean_tanimoto_distance': diversity_mean,
                'novelty_mean_tanimoto_distance': novelty_mean,
                'pareto_coverage_against_baseline': coverage
            }

            os.makedirs(os.path.join(dataDir), exist_ok=True)

         
            try:
                with open(os.path.join(dataDir, 'metrics.json'), 'w') as jf:
                    json.dump(metrics, jf, indent=2)
            except Exception as e_json1:
              
                try:
                    with open(dataDir + 'metrics.json', 'w') as jf2:
                        json.dump(metrics, jf2, indent=2)
                except Exception as e_json2:
                    with open(os.path.join(dataDir, 'metrics_error.txt'), 'a') as ef:
                        ef.write(f"{time.asctime(time.localtime(time.time()))} 写metrics.json失败: {e_json1} | 备用也失败: {e_json2}\n")

         
            csv_path = os.path.join(dataDir, 'metrics.csv') 
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
                    csv_path2 = os.path.join(dataDir, 'metrics.csv')
                    write_header2 = not os.path.exists(csv_path2)
                    with open(csv_path2, 'a') as cf2:
                        if write_header2:
                            cf2.write(','.join(header) + '\n')
                        cf2.write(','.join(str(x) for x in line) + '\n')
                except Exception as e_csv2:
                    with open(os.path.join(dataDir, 'metrics_error.txt'), 'a') as ef:
                        ef.write(f"{time.asctime(time.localtime(time.time()))} 写metrics.csv失败: {e_csv1} | 备用也失败: {e_csv2}\n")
        except Exception as e:
            print(f"error: {e}")


    while total_cost < budget:
        start_time = time.time()
        node = rootnode 
        state = root.Clone() 
        node_pool=[]
        
        while node.childNodes!=[]:
            if not int(pow(node.visits +1, 0.5))==int(pow(node.visits, 0.5)):
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
        if len(state.position)>= 70:
            
            print("position bigger than 70")
            while node != None:
                node.Update(penalty_reward)
                node = node.parentNode
            continue
        

        expanded=expanded_node(model,state.position,val)
        nodeadded=node_to_add(expanded,val)
        all_posible=chem_kn_simulation(model,state.position,val,nodeadded)
        generate_smile=predict_smile(all_posible,val)
        new_compound=make_input_smile(generate_smile)


        
               # node_index,scores,valid_smile=check_node_type(new_compound,dataDir)
        f = open(os.path.join(dataDir, "ligands.txt"), 'a')
        for p in valid_smile:
            print(p,file=f)
        f.close()
        
        f = open(os.path.join(dataDir, "scores.txt"), 'a')     
        for s in scores:
            print(s,file=f)
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

                f = open(dataDir+"depth.txt", 'a')
            print(len(state.position),file=f)
            
            base_dock_score = 0
            if REWARD == "normal":
                    scores[i][0]= _sbmolgennormalize(scores[i][0])
            elif "sigmoid":
                    scores[i][0]= _sigmoidnormalize(scores[i][0])
            elif "nonormal":
                    scores[i][0]= _linearnormalize(scores[i][0])
        
            scores[i][3] = _sa_score_normalize(scores[i][3])
            
            scores[i][4] = ACS_normalize(scores[i][4])

            if pareto.Dominated(scores[i]) == False:
                pareto.Update(scores[i],valid_smile[i])
               
                _write_metrics_snapshot()
                print("Time: ",time.asctime( time.localtime(time.time()) ))

            re.append(scores[i])
            

        for i in range(len(node_pool)):

            node=node_pool[i]
            while node != None:
                node.Update(re[i])
                node = node.parentNode
       

        print("End Search Epoch: ", time.asctime( time.localtime(time.time()) ))
        pareto_file = open(os.path.join(dataDir, 'pareto.json'), 'w')
        json.dump(pareto.to_dict(), pareto_file, indent=4, separators=(',', ': '))
        pareto_file.close()


        if CostPerMolecule==0:
            cost = time.time() - start_time
        else:
            cost = CostPerMolecule * len(valid_smile)
        total_cost += cost
        print("Total Cost: ", total_cost, "Budget: ", budget)
        
        if enable_activity_cliff and pareto.cliff_detector is not None:
            cliff_stats = pareto.get_cliff_statistics()
        
            pareto._save_cliff_statistics(cliff_stats)
    
 
    if enable_activity_cliff and pareto.cliff_detector is not None:
        cliff_stats = pareto.get_cliff_statistics()
   
        pareto._save_cliff_statistics(cliff_stats)
        pareto.cliff_detector.save_memory(os.path.join(dataDir, 'activity_cliff_memory.json'))          
       
        pareto._generate_final_cliff_report()

    try:
        
        hv_value = 0.0
        if len(pareto.front) > 0:
            _front = _hv_prepare_points_for_minimization(pareto.front)
            try:
                ref_point = [0] * len(_front[0])
                hv_value = float(_hv_wfg_min(_front, ref_point))
            except Exception as _:
                hv_value = 0.0

        num_generated = len(pareto.compounds)

        total_elapsed_sec = total_cost if CostPerMolecule == 0 else (time.time() - mcts_start_time)
        molecules_per_hour = (num_generated / (total_elapsed_sec / 3600.0)) if total_elapsed_sec > 0 else 0.0

        diversity_mean = 0.0
        diversity_count = 0
        try:
            from rdkit import Chem, DataStructs
            from rdkit.Chem import AllChem
            smiles_list = list(dict.fromkeys(pareto.compounds))
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
        coverage = None
        try:
            baseline_path = os.path.join(dataDir, 'pareto_baseline.json')
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

        metrics = {
            'timestamp': time.asctime(time.localtime(time.time())),
            'num_generated_molecules': num_generated,
            'hypervolume': hv_value,
            'total_elapsed_seconds': total_elapsed_sec,
            'diversity_mean_tanimoto_distance': diversity_mean,
            'novelty_mean_tanimoto_distance': novelty_mean,
            'pareto_coverage_against_baseline': coverage
        }

        os.makedirs(os.path.join(dataDir, 'present'), exist_ok=True)

        try:
            with open(os.path.join(dataDir, 'metrics.json'), 'w') as jf:
                json.dump(metrics, jf, indent=2)
        except Exception as e_json1:
            try:
                with open(dataDir + 'metrics.json', 'w') as jf2:
                    json.dump(metrics, jf2, indent=2)
            except Exception as e_json2:
                with open(os.path.join(dataDir, 'present', 'metrics_error.txt'), 'a') as ef:
                    ef.write(f"{time.asctime(time.localtime(time.time()))} error: {e_json1} | error: {e_json2}\n")


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
                    ef.write(f"{time.asctime(time.localtime(time.time()))} error: {e_csv1} | error: {e_csv2}\n")
        
    except Exception as e:
        print(f"error: {e}")

    return valid_compound


def UCTchemical(budget=3600*3600, CostPerMolecule=False, enable_activity_cliff=True, enable_ier=True):
    state = chemical()
    pareto_front = pareto() if isLoadTree is False else pareto.from_dict(pareto_locate)
    best = MCTS(root = state,pareto=pareto_front, budget = budget, CostPerMolecule = CostPerMolecule, enable_activity_cliff=enable_activity_cliff, enable_ier=enable_ier)


    return best


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='search molecular')

    parser.add_argument('dataDir',help='path to data dir')
    args = parser.parse_args()

    dataDir = args.dataDir

    pareto_locate = dataDir+'pareto.json'
    config = json.load(open(dataDir+'input/python_config.json'))
    isLoadTree = config['isLoadTree']
    pareto_locate = dataDir+'present/pareto.json'
    budget = config['limitBudget']
    CostPerMolecule = config['CostPerMolecule']
    rnnModelDir = config['whereisRNNmodelDir']
    randomSeed = config.get('randomSeed', 42)  
    
    set_random_seed(randomSeed)
    print(f"Random seed set to: {randomSeed}")
else :
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),dataDir+'/input/python_config.json')
    
    smile_old=zinc_data_with_bracket_original()
    val,smile=zinc_processed_with_bracket(smile_old)
  
    model=loaded_model(rnnModelDir)
    
    enable_ier = config.get('enable_ier', True)  
    enable_activity_cliff = config.get('enable_activity_cliff', True)  
    
    valid_compound=UCTchemical(budget=budget, CostPerMolecule=CostPerMolecule, 
                              enable_activity_cliff=enable_activity_cliff, enable_ier=enable_ier)
