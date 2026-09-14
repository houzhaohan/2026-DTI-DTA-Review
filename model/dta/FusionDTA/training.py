import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid
from src.models.DAT import DAT3

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--batchsize', type=int, default=256, help='Number of batch_size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--embedding-dim', type=int, default=1280, help='dimension of embedding (default: 512)')
parser.add_argument('--rnn-dim', type=int, default=128, help='hidden unit/s of RNNs (default: 256)')
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden units of FC layers (default: 256)')
parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
parser.add_argument('--dataset', default='kiba', help='dataset: davis or kiba')
parser.add_argument('--training-dataset-path', default='data/kiba/train.csv', help='training dataset path')
parser.add_argument('--save-interval', type=int, default=10, help='save model every N epochs')

args = parser.parse_args()
dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()

batch_size = args.batchsize
epochs = args.epochs
lr = args.lr
weight_decay = args.weight_decay
save_interval = args.save_interval

embedding_dim = args.embedding_dim
rnn_dim = args.rnn_dim
hidden_dim = args.hidden_dim
graph_dim = args.graph_dim

n_heads = args.n_heads
dropout = args.dropout
alpha = args.alpha

is_pretrain = args.pretrain

Alphabet = AminoAcid()

training_dataset_address = args.training_dataset_path

#processing training data
if is_pretrain:
    train_drug, train_protein, train_affinity, pid = getdata_from_csv(training_dataset_address, maxlen=1536)

else:
    train_drug, train_protein, train_affinity = getdata_from_csv(training_dataset_address, maxlen=1024)
    train_protein = [x.encode('utf-8').upper() for x in train_protein]
    train_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in train_protein]
train_affinity = torch.from_numpy(np.array(train_affinity)).float()

dataset_train = DrugTargetDataset(train_drug, train_protein, train_affinity, pid, is_target_pretrain=is_pretrain, self_link=False,dataset=dataset)
dataloader_train = torch.utils.data.DataLoader(dataset_train
                                                , batch_size=batch_size
                                                , shuffle=True
                                                , collate_fn=collate
                                                )

#model
model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)

if use_cuda:
    model.cuda()
    
#optimizer
params = [p for p in model.parameters() if p.requires_grad]
optim = torch.optim.Adam(params, lr=lr)
criterion = nn.MSELoss()

train_epoch_size = len(train_drug)

# 确保保存目录存在
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

print('--- GAT model --- ')
print(f'Dataset: {dataset}, Epochs: {epochs}, Save interval: every {save_interval} epochs')

for epoch in range(epochs):
    
    #train
    model.train()
    b = 0
    total_loss = []
    
    for protein, smiles, affinity in dataloader_train:
        
        if use_cuda:
            protein = [p.cuda() for p in protein]
            smiles = [s.cuda() for s in smiles]
            affinity = affinity.cuda()
        
        _, out = model(protein, smiles)
        loss = criterion(out, affinity)
        
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        loss = loss.cpu().detach()
        
        b = b + batch_size
        total_loss.append(loss)
        print('# [{}/{}] training {:.1%} loss={:.5f}\n'.format(epoch+1
                                                                    , epochs
                                                                    , b/train_epoch_size
                                                                    , loss
                     , end='\r'))
    
    print('total_loss={:.5f}\n'.format(np.mean(total_loss)))
    
    # 每 save_interval 轮保存一次
    if (epoch + 1) % save_interval == 0:
        save_path = os.path.join(save_dir, f'DAT_best_{dataset}_{epoch+1}.pkl')
        model.cpu()
        save_dict = {'model': model.state_dict(), 'optim': optim.state_dict(), 'epoch': epoch+1}
        torch.save(save_dict, save_path)
        print(f'[Epoch {epoch+1}] Saved model to {save_path}')
        if use_cuda:
            model.cuda()
