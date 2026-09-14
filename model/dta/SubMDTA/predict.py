"""
对 data/kiba/test.csv 做推理，生成 data/kiba/test_p.csv
（在 test.csv 所有列的基础上追加一列 Prediction）。

用法：python predict.py
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from collections import defaultdict

# ---------- 内联辅助函数（与 training.py 完全保持一致） ----------
import networkx as nx
from rdkit import Chem

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return c_size, features, edge_index

word_dict_2 = defaultdict(lambda: len(word_dict_2))
word_dict_3 = defaultdict(lambda: len(word_dict_3))
word_dict_4 = defaultdict(lambda: len(word_dict_4))

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    if ngram == 2:
        words = [word_dict_2[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    elif ngram == 3:
        words = [word_dict_3[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    else:
        words = [word_dict_4[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return words

def pro_ngram_list(prots, ngram, seq_max=1200):
    out = []
    for t in prots:
        words = split_sequence(t, ngram=ngram)
        pro_len = len(words)
        if pro_len > seq_max:
            words = words[:seq_max]
        else:
            words = words + [0] * (seq_max - pro_len)
        out.append(words)
    return np.asarray(out)

# ---------- 主流程 ----------
from utils import TestbedDataset
from torch_geometric.data import DataLoader
from models.ginconv import GINConvNet
from sub import GraphEnhance

DATASET = 'kiba'
FNAME_SUFFIX = 'GINConvNet'
SPLITS = ['train', 'val', 'test']    # 词表和 SMILES 图需要与训练时完全一致，用所有 split 构建
MODEL_PATH = 'model_' + FNAME_SUFFIX + '_' + DATASET + '.model'
SUB_PATH = 'sub_50000.pth'
OUT_CSV = os.path.join('data', DATASET, 'test_p.csv')
BATCH_SIZE = 512

# 设备
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print('device:', device)

# 1) 读所有 split 的 CSV（保持列名兼容 kiba 格式）
fpath = os.path.join('data', DATASET)
orig_test = pd.read_csv(os.path.join(fpath, 'test.csv'))
orig_test_renamed = orig_test.rename(columns={'SMILES': 'compound_iso_smiles',
                                             'Protein': 'target_sequence',
                                             'Y': 'affinity'})

splits = {}
for s in SPLITS:
    df = pd.read_csv(os.path.join(fpath, s + '.csv'))
    df = df.rename(columns={'SMILES': 'compound_iso_smiles',
                            'Protein': 'target_sequence',
                            'Y': 'affinity'})
    splits[s] = df

# 2) 用全部 split 的蛋白质统一构建 n-gram 词表
print('Building n-gram vocabularies from all splits...')
all_proteins = []
for s in SPLITS:
    all_proteins.extend(list(splits[s]['target_sequence']))
for prot in all_proteins:
    _ = split_sequence(prot, 2)
    _ = split_sequence(prot, 3)
    _ = split_sequence(prot, 4)
vocab_sizes = {'ngram_2': len(word_dict_2), 'ngram_3': len(word_dict_3), 'ngram_4': len(word_dict_4)}
print('Vocab sizes:', vocab_sizes)

# 3) SMILES 图缓存
print('Building SMILES graph cache...')
all_smiles = set()
for s in SPLITS:
    all_smiles.update(splits[s]['compound_iso_smiles'])
smile_graph = {sm: smile_to_graph(sm) for sm in all_smiles}

# 4) 准备 test 的 .pt（如已存在则直接加载）
def ensure_pt(split_name):
    pt_path = os.path.join('data', 'processed', DATASET + '_' + split_name + '.pt')
    if not os.path.isfile(pt_path):
        print('Preparing', DATASET + '_' + split_name + '.pt ...')
        df = splits[split_name]
        drugs = np.asarray(list(df['compound_iso_smiles']))
        prots = list(df['target_sequence'])
        Y = np.asarray(list(df['affinity']))
        data_obj = TestbedDataset(root='data',
                                  dataset=DATASET + '_' + split_name,
                                  xd=drugs,
                                  xt_2=pro_ngram_list(prots, 2),
                                  xt_3=pro_ngram_list(prots, 3),
                                  xt_4=pro_ngram_list(prots, 4),
                                  y=Y,
                                  smile_graph=smile_graph)
        print(pt_path, 'created.')
    else:
        print(pt_path, 'exists, loading.')
    return TestbedDataset(root='data', dataset=DATASET + '_' + split_name)

test_data = ensure_pt('test')
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 5) 构建模型并加载权重
print('Loading model from', MODEL_PATH)
sub = GraphEnhance(78, 128, 4, mode='TS', times=2)
sub.load_state_dict(torch.load(SUB_PATH, map_location=device), strict=False)

model = GINConvNet(sub).to(device)
# 替换硬编码 Embedding 尺寸以匹配 kiba 词表
model.protein_encoder_1.embed = nn.Embedding(vocab_sizes['ngram_2'], 128, padding_idx=0).to(device)
model.protein_encoder_2.embed = nn.Embedding(vocab_sizes['ngram_3'], 128, padding_idx=0).to(device)
model.protein_encoder_3.embed = nn.Embedding(vocab_sizes['ngram_4'], 128, padding_idx=0).to(device)

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

# 6) 推理
preds_list = []
print('Running prediction on test set...')
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)
        output = model(data)
        preds_list.append(output.cpu().numpy().flatten())

preds = np.concatenate(preds_list)
assert len(preds) == len(orig_test), \
    f'prediction length {len(preds)} != test.csv rows {len(orig_test)}'

# 7) 写出 test_p.csv
out_df = orig_test.copy()
out_df['Prediction'] = preds
out_df.to_csv(OUT_CSV, index=False)
print('Saved to', OUT_CSV, ', rows =', len(out_df))
