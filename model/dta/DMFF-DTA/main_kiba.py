import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem as Chem
import networkx as nx

import torch
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from build_vocab import WordVocab
from utils import *  # 确保这些函数存在：train, predicting, get_mse, calculate_metrics_and_return

from dataset import DTADataset
from model import *  # 确保GINConvNet, LinkAttention, SpatialGroupEnhance_for_1D已定义
from torch import nn as nn

import argparse

parser = argparse.ArgumentParser(description='DMFF for DTA prediction')
parser.add_argument('-d', '--dataset', type=str, default='davis',
                    help='Dataset name (default: davis, support kiba etc.)')
args = parser.parse_args()

# 将dataset_name改为从命令行参数获取
dataset_name = args.dataset

#############################################################################
# 配置参数
#############################################################################
CUDA = '0'
device = torch.device(f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu')
LR = 1e-3
NUM_EPOCHS = 100
seed = 0
batch_size = 128
#dataset_name = 'davis'

# 数据文件路径配置 - 请修改为你的实际文件路径
TRAIN_CSV = f'benchmark/{dataset_name}/train.csv'
VAL_CSV = f'benchmark/{dataset_name}/val.csv'
TEST_CSV = f'benchmark/{dataset_name}/test.csv'

# 模型保存路径
model_save_path = f'./Model/{dataset_name}_best_model.pt'

# 序列长度配置
tar_len = 1000
seq_len = 540

#############################################################################
# 模型定义（保持不变）
#############################################################################
class DMFF(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26,
                 smile_vocab=45, theta=0.5):
        super(DMFF, self).__init__()
        self.is_bidirectional = True
        # drugs
        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads
        self.MGNN = GINConvNet(num_features_xd = lstm_dim * 2 + 1, n_output=hidden_dim * 2)
    
        # SMILES
        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab + 1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance1= SpatialGroupEnhance_for_1D(groups=20)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)

        # protein
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc = nn.Linear(embedding_dim, lstm_dim)
   
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                  bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.enhance2 = SpatialGroupEnhance_for_1D(groups=200)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)

        # link
        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)

        self.fusion_graph_seq = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

        # Point-wise Feed Forward Network
        self.pwff_1 = nn.Linear(hidden_dim * 3, hidden_dim * 4)
        self.pwff_2 = nn.Linear(hidden_dim * 4, hidden_dim * 3)
   
    def forward(self, data, reset=False):
        batchsize = len(data.sm) if hasattr(data, 'sm') else len(data.y)
        smiles = torch.zeros(batchsize, seq_len).to(device).long()
        protein = torch.zeros(batchsize, tar_len).to(device).long()
        smiles_lengths = []
        protein_lengths = []

        smiles = data.smiles.to(device)
        protein = data.protein.to(device)
        smiles_lengths = data.smiles_lengths
        protein_lengths = data.protein_lengths

        if smiles.shape[0]==69122:
            smiles=smiles[:69120]
        smiles = smiles.view(batchsize, -1)
        protein = protein.view(batchsize, -1)

        # SMILES embedding and processing
        smiles = self.smiles_embed(smiles)  # B * seq len * emb_dim
        smiles = self.smiles_input_fc(smiles)  # B * seq len * lstm_dim
        smiles = self.enhance1(smiles)

        # Protein embedding and processing
        protein = self.protein_embed(protein)  # B * tar_len * emb_dim
        protein = self.protein_input_fc(protein)  # B * tar_len * lstm_dim
        protein = self.enhance2(protein)

        # BiLSTM for drugs and proteins
        smiles, _ = self.smiles_lstm(smiles)  # B * seq len * lstm_dim*2
        smiles = self.ln1(smiles)
        protein, _ = self.protein_lstm(protein)  # B * tar_len * lstm_dim *2
        protein = self.ln2(protein)

        if reset:
            return smiles, protein

        # Generate masks
        smiles_mask = self.generate_masks(smiles, smiles_lengths, self.n_heads)  # B * head* seq len
        protein_mask = self.generate_masks(protein, protein_lengths, self.n_heads)  # B * head * tar_len

        # Attention layers
        smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # B * lstm_dim*2
        protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)

        # Concatenate and attention
        out_cat = torch.cat((smiles, protein), dim=1)  # B * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # B * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        out = torch.cat([smiles_out, protein_out, out_cat], dim=-1)  # B * (rnn*2 *3)

        # Point-wise Feed Forward Network
        pwff = self.pwff_1(out)
        pwff = nn.ReLU()(pwff)
        pwff = self.dropout(pwff)  
        pwff = self.pwff_2(pwff)
        out = pwff + out 

        # Fully connected layers
        out = self.dropout(self.relu(self.out_fc1(out)))  # B * (256*8)
        out = self.dropout(self.relu(self.out_fc2(out)))  # B *  hidden_dim*2

        # Graph fusion
        gout = self.MGNN(data)
        out = torch.cat([gout, out], dim=-1)  # B * (hidden_dim*4)
        out = self.dropout(self.relu(self.fusion_graph_seq(out)))  # B * (hidden_dim*2)

        # Final prediction
        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out
        return out

    def generate_masks(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len: max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.to(device)

#############################################################################
# 辅助函数
#############################################################################
def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, np.array([])
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index

def target_to_graph(target_key, target_sequence, contact_dir, start, end):
    target_edge_index = []
    target_size = len(target_sequence)
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    
    if not os.path.exists(contact_file):
        return target_size, np.array([])
    
    contact_map = np.load(contact_file)
    contact_map = contact_map[start:end, start:end]
    index_row, index_col = np.where(contact_map > 0.8)

    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_edge_index = np.array(target_edge_index)
    return target_size, target_edge_index

def reset_feature(dataset, model):
    """重置数据集特征"""
    torch.cuda.empty_cache()
    batch_size = 128
    with torch.no_grad():
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        start = 0
        for data in tqdm(dataloader, desc="Resetting features"):
            sm, pro = model(data, reset=True)
            tar_len_list = []
            idx_list = []
            
            # 获取批次数据的长度和索引
            for i in range(min(batch_size, len(dataset) - start)):
                sm_id = dataset[start+i].sm
                pro_id = dataset[start+i].target
                pro_id = target_seq[pro_id]
                tar_len_list.append(target_len[pro_id])
                idx_list.append(smiles_idx[sm_id])
            
            # 更新数据集特征
            for i in range(start, min(len(dataset), start + batch_size)):
                batch_idx = i - start
                total_len = tar_len_list[batch_idx] + len(idx_list[batch_idx]) + 1 
                source = torch.zeros(total_len, dtype=torch.long).to(device)
                source[:tar_len_list[batch_idx]] = 0 
                source[tar_len_list[batch_idx]:tar_len_list[batch_idx]+len(idx_list[batch_idx])] = 1
                source[-1] = 2
                
                # 构建特征
                feature = torch.cat([
                    pro[batch_idx, 1:tar_len_list[batch_idx]+1], 
                    sm[batch_idx, idx_list[batch_idx]], 
                    (sm[batch_idx, 0].unsqueeze(0) + pro[batch_idx, 0].unsqueeze(0))/2
                ])
                
                new_feature = torch.cat([feature, source.unsqueeze(-1)], dim=-1) 
                dataset.data[i].x = new_feature
            
            del sm, pro, tar_len_list, idx_list
            
            # 每处理一个batch就清理缓存
            torch.cuda.empty_cache()
            #gc.collect() 
            start += batch_size


def save_result_lxc(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

#############################################################################
# 数据预处理
#############################################################################
# 加载所有数据文件


df_train = pd.read_csv(TRAIN_CSV)
df_val = pd.read_csv(VAL_CSV)
df_test = pd.read_csv(TEST_CSV)

# 合并所有数据用于构建完整的词汇表和图结构
df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

# 提取唯一的SMILES和target
smiles = set(df_all['SMILES'])
targets = set(df_all['Protein_id'])

# 构建target序列字典
target_seq = {}
for i in range(len(df_all)):
    target_seq[df_all.loc[i, 'Protein_id']] = df_all.loc[i, 'target_sequence']

# 构建SMILES图字典
smiles_graph = {}
for sm in tqdm(smiles, desc="Processing SMILES to graph"):
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph

# 构建target的接触图相关字典
target_uniprot_dict = {}
target_process_start = {}
target_process_end = {}

for i in range(len(df_all)):
    target = df_all.loc[i, 'Protein_id']
    if dataset_name == 'kiba':
        uniprot = df_all.loc[i, 'Protein_id']
    else:
        uniprot = df_all.loc[i, 'uniprot']
    target_uniprot_dict[target] = uniprot
    target_process_start[target] = df_all.loc[i, 'target_sequence_start']
    target_process_end[target] = df_all.loc[i, 'target_sequence_end']

# 构建target图字典
contact_dir = './target_contact_map_' + dataset_name + '/'
target_graph = {}
for target in tqdm(target_seq.keys(), desc="Processing target to graph"):
    uniprot = target_uniprot_dict[target]
    contact_file = os.path.join(contact_dir, uniprot + '.npy')
    if not os.path.exists(contact_file):
        print(f"Warning: Contact map file {contact_file} not found!")
        target_graph[target] = np.array([])
        continue

    start = int(target_process_start[target])
    end = int(target_process_end[target])
    _, graph = target_to_graph(uniprot, target_seq[target], contact_dir, start, end)
    target_graph[target] = graph

# 加载词汇表
drug_vocab = WordVocab.load_vocab('./Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('./Vocab/protein_vocab.pkl')

# 预处理SMILES
smiles_idx = {}
smiles_emb = {}
smiles_len = {}
for sm in tqdm(smiles, desc="Processing SMILES embeddings"):
    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)) and drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
            content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
            flag += 2
            continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag += 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)

    # 填充到固定长度
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    # 构建SMILES索引
    tem = []
    for i, c in enumerate(X):
        if 'atom_dict' in locals() and atom_dict.__contains__(c):  # 确保atom_dict已定义
            tem.append(i)
    smiles_idx[sm] = tem

# 预处理Protein
target_emb = {}
target_len = {}
for k in tqdm(target_seq.keys(), desc="Processing protein embeddings"):
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)) and target_vocab.stoi.__contains__(seq[flag:flag + 2]):
            content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
            flag += 2
            continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag += 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)

    # 填充到固定长度
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)

    target_emb[seq] = torch.tensor(X)

#############################################################################
# 构建数据集和数据加载器
#############################################################################
print("Building datasets...")
train_dataset = DTADataset(
    root='./', 
    path=TRAIN_CSV, 
    smiles_emb=smiles_emb, 
    target_emb=target_emb, 
    smiles_idx=smiles_idx, 
    smiles_graph=smiles_graph, 
    target_graph=target_graph, 
    smiles_len=smiles_len, 
    target_len=target_len
)

val_dataset = DTADataset(
    root='./', 
    path=VAL_CSV, 
    smiles_emb=smiles_emb, 
    target_emb=target_emb, 
    smiles_idx=smiles_idx, 
    smiles_graph=smiles_graph, 
    target_graph=target_graph, 
    smiles_len=smiles_len, 
    target_len=target_len
)

test_dataset = DTADataset(
    root='./', 
    path=TEST_CSV, 
    smiles_emb=smiles_emb, 
    target_emb=target_emb, 
    smiles_idx=smiles_idx, 
    smiles_graph=smiles_graph, 
    target_graph=target_graph, 
    smiles_len=smiles_len, 
    target_len=target_len
)   

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#############################################################################
# 模型训练
#############################################################################
print("Building model...")
model = DMFF(
    embedding_dim=256, 
    lstm_dim=128, 
    hidden_dim=256, 
    dropout_rate=0.2,
    alpha=0.2, 
    n_heads=8, 
    bilstm_layers=2, 
    protein_vocab=26,
    smile_vocab=45, 
    theta=0.5
).to(device)

# 优化器和学习率调度器
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 20, eta_min=5e-4, last_epoch=-1
)

# 训练记录
best_mse = float('inf')
best_epoch = -1

# 开始训练
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

    # 第一轮epoch重置特征
    if epoch == 0:
        print('Resetting features...')
        # 对所有数据集重置特征
        reset_feature(train_dataset, model)
        reset_feature(val_dataset, model)
        reset_feature(test_dataset, model)

    # 训练
    train(model, train_loader, optimizer, epoch)

    # 验证
    G, P = predicting(model, val_loader)
    val_mse = get_mse(G, P)

    # 保存最佳模型
    if val_mse < best_mse:
        best_mse = val_mse
        best_epoch = epoch + 1
        torch.save(model.state_dict(), model_save_path)
        print(f'Val MSE improved at epoch {best_epoch}! Best MSE: {best_mse:.4f}')
    else:
        print(f'Val MSE: {val_mse:.4f} (No improvement since epoch {best_epoch}, best MSE: {best_mse:.4f})')

    G, P = predicting(model, test_loader)

    result_dir_lxc = f'res/{dataset_name}/round_0'
    if (os.path.exists(result_dir_lxc) == False):
        os.makedirs(result_dir_lxc)
    result_file_lxc = result_dir_lxc + f"/result_{epoch}.txt"
    save_result_lxc(G, P, result_file_lxc)

    # 更新学习率
    schedule.step()

#############################################################################
# 测试最佳模型
#############################################################################
print("\nEvaluating best model on test set...")
# 加载最佳模型
model.load_state_dict(torch.load(model_save_path))
model.eval()

# 测试集预测
G, P = predicting(model, test_loader)
cindex, rm2, test_mse = calculate_metrics_and_return(G, P, test_loader)

# 打印测试结果
print(f"\nTest Set Results:")
print(f"CI: {cindex:.4f}")
print(f"RM2: {rm2:.4f}")
print(f"MSE: {test_mse:.4f}")

# 保存测试结果
results = {
    'CI': cindex,
    'RM2': rm2,
    'MSE': test_mse,
    'best_epoch': best_epoch,
    'best_val_mse': best_mse
}
pd.DataFrame([results]).to_csv(f'./{dataset_name}_test_results.csv', index=False)
print(f"\nResults saved to {dataset_name}_test_results.csv")


