import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.ginconv import GINConvNet
from utils import *
import time
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
from sub import *

# ---------- inline data-prep helpers (adapted from create_data.py) ----------
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

from collections import defaultdict
# n-gram vocabularies are built lazily from the full dataset
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

# ---------- end of inline helpers ----------


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    preds_list, labels_list = [], []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            preds_list.append(output)               # 先留在 GPU，循环结束一次性同步
            labels_list.append(data.y.view(-1, 1))
    total_preds = torch.cat(preds_list, dim=0).cpu().numpy().flatten()
    total_labels = torch.cat(labels_list, dim=0).cpu().numpy().flatten()
    return total_labels, total_preds


# ---------- dataset selection ----------
dataset = 'kiba'
modeling = GINConvNet
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1200

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# ---------- data preparation (inline, supports kiba/{train,val,test}.csv) ----------
print('\nrunning on ', model_st + '_' + dataset)
fpath = os.path.join('data', dataset)

# Load all splits and map columns: SMILES->compound_iso_smiles, Protein->target_sequence, Y->affinity
splits = {}
for split_name in ['train', 'val', 'test']:
    df = pd.read_csv(os.path.join(fpath, split_name + '.csv'))
    df = df.rename(columns={'SMILES': 'compound_iso_smiles',
                            'Protein': 'target_sequence',
                            'Y': 'affinity'})
    splits[split_name] = df

# Build n-gram vocabularies from ALL proteins (so vocab indices are consistent across splits)
print('Building n-gram vocabularies from all splits...')
all_proteins = list(splits['train']['target_sequence']) + \
               list(splits['val']['target_sequence']) + \
               list(splits['test']['target_sequence'])
# Force vocab build by iterating (defaultdict will grow them)
for prot in all_proteins:
    _ = split_sequence(prot, 2)
    _ = split_sequence(prot, 3)
    _ = split_sequence(prot, 4)
vocab_sizes = {'ngram_2': len(word_dict_2), 'ngram_3': len(word_dict_3), 'ngram_4': len(word_dict_4)}
print('Vocab sizes:', vocab_sizes)

# Build SMILES graph cache from all splits
print('Building SMILES graph cache...')
all_smiles = set()
for split_name in ['train', 'val', 'test']:
    all_smiles.update(splits[split_name]['compound_iso_smiles'])
smile_graph = {}
for smile in all_smiles:
    smile_graph[smile] = smile_to_graph(smile)

# Ensure .pt files exist for all three splits
def ensure_pt(split_name):
    pt_path = os.path.join('data', 'processed', dataset + '_' + split_name + '.pt')
    if not os.path.isfile(pt_path):
        print('Preparing', dataset + '_' + split_name + '.pt ...')
        df = splits[split_name]
        drugs = np.asarray(list(df['compound_iso_smiles']))
        prots = list(df['target_sequence'])
        Y = np.asarray(list(df['affinity']))
        pro_2 = pro_ngram_list(prots, 2)
        pro_3 = pro_ngram_list(prots, 3)
        pro_4 = pro_ngram_list(prots, 4)
        data_obj = TestbedDataset(root='data',
                                  dataset=dataset + '_' + split_name,
                                  xd=drugs, xt_2=pro_2, xt_3=pro_3, xt_4=pro_4,
                                  y=Y, smile_graph=smile_graph)
        print(pt_path, 'has been created')
    else:
        print(pt_path, 'already exists, loading ...')
    return TestbedDataset(root='data', dataset=dataset + '_' + split_name)

train_data = ensure_pt('train')
val_data   = ensure_pt('val')
test_data  = ensure_pt('test')

# PyTorch mini-batch loaders
NUM_WORKERS = 4   # 利用多核 CPU 并行加载（你有 32 核，设 4 够了，太多反而 I/O 争抢）
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=TEST_BATCH_SIZE,  shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=TEST_BATCH_SIZE,  shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

# ---------- model construction ----------
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

sub = GraphEnhance(78, 128, 4, mode='TS', times=2)
sub.load_state_dict(torch.load('sub_50000.pth', map_location=device), strict=False)
sub.train()

model = modeling(sub).to(device)

# Override hardcoded Embedding sizes in GINConvNet's protein encoders to match kiba vocab
model.protein_encoder_1.embed = nn.Embedding(vocab_sizes['ngram_2'], 128, padding_idx=0).to(device)
model.protein_encoder_2.embed = nn.Embedding(vocab_sizes['ngram_3'], 128, padding_idx=0).to(device)
model.protein_encoder_3.embed = nn.Embedding(vocab_sizes['ngram_4'], 128, padding_idx=0).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_mse = 1000
best_ci = 0
best_rm2 = 0
best_epoch = -1
model_file_name = 'model_' + model_st + '_' + dataset + '.model'
result_file_name = 'result_' + model_st + '_' + dataset + '.csv'

with open(result_file_name, 'w') as f:
    f.write('epoch,train_mse\n')

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1)

    # 每 50 轮存一次（带 epoch 序号）
    if (epoch + 1) % 50 == 0 or (epoch + 1) == NUM_EPOCHS:
        save_path = model_file_name.replace('.model', f'_ep{epoch+1}.model')
        torch.save(model.state_dict(), save_path)
        print('Model saved:', save_path)

    epoch_time = time.time() - start_time
    print('epoch_time', round(epoch_time, 2), 's')
    start_time = time.time()
