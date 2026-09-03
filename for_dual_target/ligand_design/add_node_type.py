from math import *
import numpy as np
import pandas as pd
import traceback
import time
import subprocess
#from load_model import loaded_model
from keras.utils import pad_sequences
from rdkit import Chem
from rdkit import RDLogger
from rdkit.RDLogger import logger
from rdkit.Chem import QED
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
import networkx as nx

# Try to import sascorer (use approximate calculation if unavailable)
try:
    import sascorer
    HAS_SASCORER = True
except ImportError:
    HAS_SASCORER = False
    print("[WARN] sascorer not available")

from joblib import load

import os
import json
import re
def expanded_node(model,state,val):

    all_nodes=[]

    end="\n"

    #position=[]
    position=[]
    position.extend(state)
    total_generated=[]
    new_compound=[]
    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= pad_sequences(x, maxlen=82, dtype='int32',padding='post', truncating='pre', value=0.)

    for i in range(30):
        predictions=model.predict_on_batch(x_pad)
        preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
        preds = np.log(preds) / 1.0
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_probas = np.random.multinomial(1, preds, 1)
        next_int=np.argmax(next_probas)
        all_nodes.append(next_int)

    all_nodes=list(set(all_nodes))

    return all_nodes


def node_to_add(all_nodes,val):
    added_nodes=[]
    for i in range(len(all_nodes)):
        added_nodes.append(val[all_nodes[i]])

    return added_nodes



def chem_kn_simulation(model,state,val,added_nodes):
    all_posible=[]

    end="\n"
    for i in range(len(added_nodes)):
        position=[]
        position.extend(state)
        position.append(added_nodes[i])
        total_generated=[]
        new_compound=[]
        get_int_old=[]
        for j in range(len(position)):
            get_int_old.append(val.index(position[j]))

        get_int=get_int_old

        x=np.reshape(get_int,(1,len(get_int)))
        x_pad= pad_sequences(x, maxlen=82, dtype='int32',padding='post', truncating='pre', value=0.)
        while not get_int[-1] == val.index(end):
            predictions=model.predict_on_batch(x_pad)
            preds=np.asarray(predictions[0][len(get_int)-1]).astype('float64')
            preds = np.log(preds) / 1.0
            preds = np.exp(preds) / np.sum(np.exp(preds))
            next_probas = np.random.multinomial(1, preds, 1)
            next_int=np.argmax(next_probas)
            a=predictions[0][len(get_int)-1]
            next_int_test=sorted(range(len(a)), key=lambda i: a[i])[-10:]
            get_int.append(next_int)
            x=np.reshape(get_int,(1,len(get_int)))
            x_pad = pad_sequences(x, maxlen=82, dtype='int32',padding='post', truncating='pre', value=0.)
            if len(get_int)>81:
                break
        total_generated.append(get_int)
        all_posible.extend(total_generated)

    return all_posible



def predict_smile(all_posible,val):
    new_compound=[]
    for i in range(len(all_posible)):
        total_generated=all_posible[i]
        generate_smile=[]
        for j in range(len(total_generated)-1):
            generate_smile.append(val[total_generated[j]])
        generate_smile.remove("&")
        new_compound.append(generate_smile)

    return new_compound


def make_input_smile(generate_smile):
    new_compound=[]
    for i in range(len(generate_smile)):
        middle=[]
        for j in range(len(generate_smile[i])):
            middle.append(generate_smile[i][j])
        com=''.join(middle)
        new_compound.append(com)

    return new_compound



def check_node_type(new_compound,dataDir):
    isUseeToxPred = False
    if os.path.exists(dataDir+'input/python_config.json') :
        config = json.load(open(dataDir+'input/python_config.json','r'))
        proteinName = config['proteinName']
        # Dual target: also get 3PP0 protein name
        proteinName_3PP0 = config.get('proteinName_3PP0', '3PP0')
        isUseeToxPred = config['isUseeToxPred']
        saThreshold = config['saThreshold']
        if isUseeToxPred:
            eToxPredModel = load("./ligand_design/etoxpred_best_model.joblib")

    node_index=[]
    valid_compound=[]
    all_smile=[]
    distance=[]

    docking_call_count = 0

    scores=[]
    
    for i in range(len(new_compound)):
        new_compound[i] = re.sub('\n','',new_compound[i])
        # Dual target: 6-dimensional score vector [EGFR_docking, 3PP0_docking, QED, IER, SA_score, ACS]
        score = [0.,0.,0.,0.,0.,0.]
        if len(new_compound[i]) == 0:
            continue
        assert len(new_compound[i]) >0
        with open(dataDir+"./output/allproducts.txt","a") as f:
            f.write(new_compound[i]+"\n")
        try:
            RDLogger.DisableLog('rdApp.*')
            ko = Chem.MolFromSmiles(new_compound[i])
        finally:
            RDLogger.EnableLog('rdApp.*')
        if ko == None:
            continue
        assert ko != None 

        if HAS_SASCORER:
            SA_score = sascorer.calculateScore(ko)
        else:
            SA_score = 5.0  # Default value when sascorer unavailable
        score[4] = SA_score
        score[5] = 0.0  # ACS placeholder

        cycle_list = nx.cycle_basis(nx.Graph(AllChem.GetAdjacencyMatrix(ko)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max(len(j) for j in cycle_list)
        if cycle_length > 8:
            continue
        assert cycle_length <= 8

        if isUseeToxPred:
            mol = Chem.AddHs(ko)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_string = fp.ToBitString()
            tmpX = np.array(list(fp_string), dtype=float)
            tox_score = eToxPredModel.predict_proba(tmpX.reshape((1, 1024)))[:, 1]
            toxThreshold = config.get('toxThreshold', 0.7)
            if tox_score[0] >= toxThreshold:
                continue

        try:
            score[2] = round(QED.default(ko), 3)
        except Exception:
            score[2] = 0

        with open(dataDir+'./workspace/ligand.smi','w') as f:
            f.write(new_compound[i])
        with open(dataDir+'./output/allLigands.txt','a', newline="\n") as f:
            f.write(new_compound[i]+"\n")

        try:
            cvt_log = open(dataDir+"workspace/cvt_log.txt","w")
            cvt_cmd = ["obabel", dataDir+"workspace/ligand.smi" ,"-O",dataDir+"workspace/ligand.pdbqt" ,"--gen3D","-p"]
            subprocess.run(cvt_cmd, stdin=None, input=None, stdout=cvt_log, stderr=None, shell=False, timeout=300, check=False, universal_newlines=False)
            cvt_log.close()
        except:
            f = open(dataDir+"present/error_output.txt", 'a')
            print("cvt_error: ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
            continue
        
        # === Docking with EGFR (primary target) ===
        try:
            vina_log = open(dataDir+"workspace/log_docking.txt","w")
            docking_cmd =["vina --config "+dataDir+"./input/"+proteinName+"_vina_config.txt --num_modes=1 --receptor="+dataDir+"./input/"+proteinName+".pdbqt --ligand="+dataDir+"./workspace/ligand.pdbqt"]
            subprocess.run(docking_cmd, stdin=None, input=None, stdout=vina_log, stderr=None, shell=True, timeout=600, check=False, universal_newlines=False)
            vina_log.close()
            docking_call_count += 1
            data = pd.read_csv(dataDir+'workspace/log_docking.txt', sep= "\t",header=None)
            m = round(float(data.values[-2][0].split()[1]),2)
        except:
            f = open(dataDir+"./present/error_output.txt", 'a')
            print("vina_error (EGFR): ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
            continue
        assert m < 10**10
        score[0] = m  # EGFR docking score
        
        # === Docking with 3PP0 (secondary target) ===
        try:
            vina_log_3pp0 = open(dataDir+"workspace/log_docking_3pp0.txt","w")
            docking_cmd_3pp0 =["vina --config "+dataDir+"./input/"+proteinName_3PP0+"_vina_config.txt --num_modes=1 --receptor="+dataDir+"./input/"+proteinName_3PP0+".pdbqt --ligand="+dataDir+"./workspace/ligand.pdbqt"]
            subprocess.run(docking_cmd_3pp0, stdin=None, input=None, stdout=vina_log_3pp0, stderr=None, shell=True, timeout=600, check=False, universal_newlines=False)
            vina_log_3pp0.close()
            docking_call_count += 1
            data_3pp0 = pd.read_csv(dataDir+'workspace/log_docking_3pp0.txt', sep= "\t",header=None)
            m_3pp0 = round(float(data_3pp0.values[-2][0].split()[1]),2)
        except:
            f = open(dataDir+"./present/error_output.txt", 'a')
            print("vina_error (3PP0): ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
            continue
        assert m_3pp0 < 10**10
        score[1] = m_3pp0  # 3PP0 docking score

        try:
            score[2]=round(QED.default(ko),3)
        except:
            score[2]=0

        if isUseeToxPred:
            mol = Chem.AddHs(ko)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_string = fp.ToBitString()
            tmpX = np.array(list(fp_string), dtype=float)
            tox_score = eToxPredModel.predict_proba(tmpX.reshape((1, 1024)))[:, 1]
            toxThreshold = config.get('toxThreshold', 0.7)
            if tox_score[0] >= toxThreshold:
                continue

        node_index.append(i)
        valid_compound.append(new_compound[i])
        # Store both docking scores for later processing in MCTS
        # score[0] = EGFR docking score, score[1] = 3PP0 docking score
        scores.append(score)

    docking_stats_file = dataDir + "present/docking_call_count.txt"
    with open(docking_stats_file, 'a') as f:
        f.write(f"{docking_call_count}\n")

    return node_index,scores,valid_compound, docking_call_count