from math import *
import numpy as np

import time

import subprocess
#from load_model import loaded_model
from keras.utils import pad_sequences
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Descriptors

from rdkit.Chem import AllChem

import sascorer
#import pickle
import networkx as nx
from rdkit.Chem import rdmolops
#import os
import pandas as pd
import traceback
from joblib import load

from rdkit import rdBase
import os
import json
import re
def expanded_node(model,state,val):

    all_nodes=[]

    end="\n"

    position=[]
    position.extend(state)

    get_int_old=[]
    for j in range(len(position)):
        get_int_old.append(val.index(position[j]))

    get_int=get_int_old

    x=np.reshape(get_int,(1,len(get_int)))
    x_pad= pad_sequences(x, maxlen=81, dtype='int32',padding='post', truncating='pre', value=0.)

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
        x_pad= pad_sequences(x, maxlen=81, dtype='int32',padding='post', truncating='pre', value=0.)
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
           
            x_pad = pad_sequences(x, maxlen=81, dtype='int32',padding='post', truncating='pre', value=0.)
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
        isUseeToxPred = config['isUseeToxPred']
        saThreshold = config['saThreshold']
        if isUseeToxPred:
            eToxPredModel = load("./ligand_design/etoxpred_best_model.joblib")

    node_index=[]
    valid_compound=[]
    all_smile=[]
    distance=[]
    

    scores=[]
    
    
    for i in range(len(new_compound)):
        new_compound[i] = re.sub('\n','',new_compound[i])
        score = [0.,0.,0.,0.,0.]  
        if len(new_compound[i]) == 0:
            continue
        assert len(new_compound[i]) >0
        with open(dataDir+"./output/allproducts.txt","a") as f:
            f.write(new_compound[i]+"\n")
       
        try:
            
            rdBase.DisableLog('rdApp.*')
            ko = Chem.MolFromSmiles(new_compound[i])
        finally:
            
            rdBase.EnableLog('rdApp.*')
        if ko == None:
            continue
        
        assert ko != None 

        SA_score = sascorer.calculateScore(ko)
        score[3] = SA_score  
        
        score[4] = 0.0       

        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(ko)))
        if len(cycle_list) == 0:
            cycle_length =0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        # TODO: Modified
        if cycle_length > 6:
            continue
        assert cycle_length <= 6

        
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
        try:
            vina_log = open(dataDir+"workspace/log_docking.txt","w")
            docking_cmd =["vina --config "+dataDir+"./input/"+proteinName+"_vina_config.txt --num_modes=1 --receptor="+dataDir+"./input/"+proteinName+".pdbqt --ligand="+dataDir+"./workspace/ligand.pdbqt"]#TODO: direct acess to protein file
            subprocess.run(docking_cmd, stdin=None, input=None, stdout=vina_log, stderr=None, shell=True, timeout=600, check=False, universal_newlines=False)
            vina_log.close()
            data = pd.read_csv(dataDir+'workspace/log_docking.txt', sep= "\t",header=None)
            m = round(float(data.values[-2][0].split()[1]),2)
        except:
            f = open(dataDir+"./present/error_output.txt", 'a')
            print("vina_error: ", time.asctime( time.localtime(time.time()) ),file=f)
            print(traceback.print_exc(),file=f)
            f.close()
            continue
        assert m < 10**10
  
        try:
            score[1]=round(QED.default(ko),3)
            #score[1]=0
        except:
            score[1]=0
     
        
        ## eToxPred
        ## https://github.com/pulimeng/eToxPred/blob/master/etoxpred_predict.py
        if True:
            mol = Chem.AddHs(ko)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fp_string = fp.ToBitString()
            tmpX = np.array(list(fp_string),dtype=float)
            tox_score = eToxPredModel.predict_proba(tmpX.reshape((1,1024)))[:,1]
            toxThreshold = config.get('toxThreshold', 0.7)  # 默认0.7
            if tox_score[0] >= toxThreshold:
                continue
            score[2] = (1- tox_score[0])
          

        node_index.append(i)
        valid_compound.append(new_compound[i])
        score[0]=m 
        scores.append(score)
                
                    






    ##print(scores)
    return node_index,scores,valid_compound