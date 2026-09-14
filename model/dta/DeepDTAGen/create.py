import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict

# 忽略所有警告
#import warnings
# 屏蔽RDKit的弃用警告
#warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit")
from rdkit import Chem

from rdkit import RDLogger

# 屏蔽RDKit所有警告（包括弃用警告）
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils1 import *
import re
from typing import List

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set] + [x not in allowable_set]

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + #Atom symbol
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Number of adjacent atoms
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + # Number of adjacent hydrogens
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + #Implicit valence
                    one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) + #Formal charge
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) + #Hybridization
                    [atom.GetIsAromatic()] + #Aromaticity
                    [atom.IsInRing()] #In ring
                    )

def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_feats = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_feats = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_feats = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_feats = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(bond_feats)

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edge_feats = bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {'edge_feats': edge_feats}))

    g = nx.Graph()
    g.add_edges_from(edges)
    g = g.to_directed()
    edge_index = []
    edge_feats = []
    for e1, e2, feats in g.edges(data=True):
        edge_index.append([e1, e2])
        edge_feats.append(feats['edge_feats'])

    return c_size, features, edge_index, edge_feats

def smile_parse(smiles, tokenizer: Tokenizer):
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    smi = tokenizer.parse(smiles)
    return smi

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []
for dt_name in ['kiba_dta','davis_dta']:
    opts = ['train','val','test']
    for opt in opts:
        df = pd.read_csv('benchmark/' + dt_name + '/' + opt + '.csv')
        compound_iso_smiles += list( df['SMILES'] )
compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}
for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g
dir = 'benchmark'
datasets = ['kiba_dta']# 'kiba_dta']
# convert to PyTorch data format
for dataset in datasets:
    processed_data_file_train = 'benchmark/processed/' + dataset + '_train_1.pt'
    processed_data_file_val = 'benchmark/processed/' + dataset + '_val_1.pt'
    processed_data_file_test = 'benchmark/processed/' + dataset + '_test_1.pt'
    tokenizer_file = f'{dir}/{dataset}/tokenizer.pkl'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        df_train = pd.read_csv('benchmark/' + dataset + '/train.csv')
        df_val = pd.read_csv('benchmark/' + dataset + '/val.csv')
        df_test = pd.read_csv('benchmark/' + dataset + '/test.csv')

        all_smiles = set(df_train['SMILES']).union(set(df_test['SMILES']))
        tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_smiles))

        with open(tokenizer_file, 'wb') as file:
            pickle.dump(tokenizer, file)
        # Process train set
        train_drugs, train_MTS, train_prots, train_Y = list(df_train['SMILES']), list(df_train['SMILES']), list(df_train['Protein']), list(df_train['Y'])
        XT = [seq_cat(t) for t in train_prots]
        train_drugs, train_MTS, train_prots, train_Y = np.asarray(train_drugs), np.asarray(train_MTS), np.asarray(XT), np.asarray(train_Y)
        train_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in train_MTS]

        val_drugs, val_MTS, val_prots, val_Y = list(df_val['SMILES']), list(df_val['SMILES']), list(df_val['Protein']), list(df_val['Y'])
        XT = [seq_cat(t) for t in val_prots]
        val_drugs, val_MTS, val_prots, val_Y = np.asarray(val_drugs), np.asarray(val_MTS), np.asarray(XT), np.asarray(val_Y)
        val_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in val_MTS]


        # Process test set
        test_drugs, test_MTS, test_prots, test_Y = list(df_test['SMILES']), list(df_test['SMILES']), list(df_test['Protein']), list(df_test['Y'])
        XT = [seq_cat(t) for t in test_prots]
        test_drugs, test_MTS, test_prots, test_Y = np.asarray(test_drugs), np.asarray(test_MTS), np.asarray(XT), np.asarray(test_Y)
        test_XD = [torch.LongTensor(tokenizer.parse(smile)) for smile in test_MTS]

        print('preparing ', dataset + '_train.pt in pytorch format!')
        #train_data = TestbedDataset(root='benchmark', dataset=dataset+'_train_1', xd=train_drugs, xdt=train_XD, xt=train_prots, y=train_Y,smile_graph=smile_graph)

        val_data = TestbedDataset(root='benchmark', dataset=dataset+'_val_1', xd=val_drugs, xdt=val_XD, xt=val_prots, y=val_Y,smile_graph=smile_graph)

        print('preparing ', dataset + '_test.pt in pytorch format!')
        test_data = TestbedDataset(root='benchmark', dataset=dataset+'_test_1', xd=test_drugs, xdt=test_XD, xt=test_prots, y=test_Y,smile_graph=smile_graph)
        print(processed_data_file_train, ' and ', processed_data_file_test, ' have been created')
    else:
        print(processed_data_file_train, ' and ', processed_data_file_test, ' are already created')

