import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from src.getdata import getdata_from_csv
from src.utils import DrugTargetDataset, collate, AminoAcid
from src.models.DAT import DAT3

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', default=True, help='Disables CUDA training.')
parser.add_argument('--batchsize', type=int, default=128, help='Number of batch_size')
parser.add_argument('--embedding-dim', type=int, default=1280, help='dimension of embedding')
parser.add_argument('--rnn-dim', type=int, default=128, help='hidden unit/s of RNNs')
parser.add_argument('--hidden-dim', type=int, default=256, help='hidden units of FC layers')
parser.add_argument('--graph-dim', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--pretrain', action='store_false', help='protein pretrained or not')
parser.add_argument('--dataset', default='kiba', help='dataset: davis or kiba')
parser.add_argument('--testing-dataset-path', default='data/kiba/test.csv', help='testing dataset path')
parser.add_argument('--model-path', default='saved_models/DAT_best_davis.pkl', help='trained model path')
parser.add_argument('--output-path', default='data/kiba/test_p.csv', help='output CSV path')

args = parser.parse_args()

dataset = args.dataset
use_cuda = args.cuda and torch.cuda.is_available()
batch_size = args.batchsize

embedding_dim = args.embedding_dim
rnn_dim = args.rnn_dim
hidden_dim = args.hidden_dim
graph_dim = args.graph_dim
n_heads = args.n_heads
dropout = args.dropout
alpha = args.alpha
is_pretrain = args.pretrain

Alphabet = AminoAcid()

testing_dataset_address = args.testing_dataset_path

# processing testing data
if is_pretrain:
    test_drug, test_protein, test_affinity, pid = getdata_from_csv(testing_dataset_address, maxlen=1536)
else:
    test_drug, test_protein, test_affinity = getdata_from_csv(testing_dataset_address, maxlen=1024)
    pid = None
    test_protein = [x.encode('utf-8').upper() for x in test_protein]
    test_protein = [torch.from_numpy(Alphabet.encode(x)).long() for x in test_protein]
test_affinity = torch.from_numpy(np.array(test_affinity)).float()

dataset_test = DrugTargetDataset(test_drug, test_protein, test_affinity, pid, is_target_pretrain=is_pretrain, self_link=False, dataset=dataset)
dataloader_test = torch.utils.data.DataLoader(dataset_test
                                                , batch_size=batch_size
                                                , shuffle=False
                                                , collate_fn=collate
                                                )

# model
model = DAT3(embedding_dim, rnn_dim, hidden_dim, graph_dim, dropout, alpha, n_heads, is_pretrain=is_pretrain)

# load trained model (compat for PyTorch < 2.6)
try:
    checkpoint = torch.load(args.model_path, weights_only=False)
except TypeError:
    checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model'], strict=False)

if use_cuda:
    model.cuda()

total_pred = []
model.eval()

with torch.no_grad():
    for protein, smiles, affinity in dataloader_test:
        if use_cuda:
            protein = [p.cuda() for p in protein]
            smiles = [s.cuda() for s in smiles]

        _, out = model(protein, smiles)

        out = out.cpu().detach().numpy().flatten()
        total_pred.extend(out.tolist())

total_pred = np.array(total_pred)

# save predictions to CSV
df = pd.read_csv(testing_dataset_address)
df['Prediction'] = total_pred
df.to_csv(args.output_path, index=False)
print(f'Saved predictions to {args.output_path}')
