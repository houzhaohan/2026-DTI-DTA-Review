import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from data2 import CPIDataset
from models.core import *
from utils import *
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from time import time
from rdkit import RDLogger
import warnings
import copy

warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog('rdApp.*')

# training function at each epoch
def train_dta(model, loss_fn, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()

    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        # print(data.y.view(-1, 1).float().to(device))

        loss = loss_fn(output, data['LABEL'].view(-1, 1).float().cuda())
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dta(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data['LABEL'].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()

def others_dataloader(batch_size, workers, dataset = 'davis', data_path='./data'):
    print('\nrunning on ', dataset)

    path = data_path + '/' + dataset

    train_set = CPIDataset(f'{path}/train.csv', f'{path}/compound', f'{path}/protein/train')
    val_set = CPIDataset(f'{path}/val.csv', f'{path}/compound', f'{path}/protein/val')
    test_set = CPIDataset(f'{path}/test.csv',f'{path}/compound', f'{path}/protein/test')


    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=train_set.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=val_set.collate_fn,
        shuffle=False,
        num_workers=workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=test_set.collate_fn,
        shuffle=False,
        num_workers=workers,
    )

    return train_loader, val_loader, test_loader

def save_result_lxc(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def run(round_index,args: argparse.Namespace):


    data_path = args.root_data_path
    dataset = args.dataset
    batch_size = args.batch_size
    LR = args.learning_rate
    NUM_EPOCHS = args.max_epochs
    seed = args.seed

    model = PMMRNet(args).cuda()


    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    
    train_loader,val_loader,test_loader = others_dataloader(batch_size, args.num_workers, dataset, data_path)


    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30,
                                                               eps=1e-08)
    best_mae = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + dataset + f'_{round_index}.pth'
    result_file_name = 'result_' + dataset + f'_{round_index}.csv'
    model_best = copy.deepcopy(model)
    for epoch in range(NUM_EPOCHS):

        train_dta(model, loss_fn, train_loader, optimizer, epoch + 1)

        G, P = predicting_dta(model, val_loader)
    
        ret = [rmse(G, P), mae(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            # ret = [rmse(G, P), mse(G, P), ci(G, P), rm2(G, P)]

        current_lr = optimizer.param_groups[0]['lr']
        print("current lr:", current_lr)
        with open(result_file_name, 'a') as f:
            f.write(str(epoch) + ',' + ','.join(map(str, ret)) + '\n')
        if ret[1] < best_mae:
            torch.save(model.state_dict(), model_file_name)
            model_best = copy.deepcopy(model)
            best_epoch = epoch + 1
            best_mae = ret[1]
            best_ci = ret[-1]
            print('mae improved at epoch ', best_epoch, '; best_mae,best_ci:', best_mae, best_ci)
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mae,best_ci:', best_mae, best_ci)

        G, P = predicting_dta(model, test_loader)
        print(f'Test:mse:{rmse(G,P)},ci:{ci(G,P)}')
        result_dir = f'res/{dataset}/round_{round_index}/'
        if (os.path.exists(result_dir) == False):
            os.makedirs(result_dir)
        result_file = result_dir + "/result_" + str(epoch + 1) + ".txt"
        save_result_lxc(G, P, result_file)
        
        scheduler.step(best_mae)

    G, P = predicting_dta(model_best, test_loader)
    print(f'Test:mse:{rmse(G, P)},ci:{ci(G, P)}')
    result_dir = f'res/{dataset}/round_{round_index}/'
    if (os.path.exists(result_dir) == False):
        os.makedirs(result_dir)
    result_file = result_dir + "/result_" + str(NUM_EPOCHS + 1) + ".txt"
    save_result_lxc(G, P, result_file)

    scheduler.step(best_mae)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='./benchmark', help='Raw Data Path')
    parser.add_argument('-d','--dataset', type=str, default='davis_dta', help='Datasets')
    parser.add_argument('--objective',
                        type=str,
                        default='regression',
                        help='Objective (classification / regression)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--max_epochs', type=int, default=200, help='Max Trainning Epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of Subprocesses for Data Loading')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate for Trainning')
    parser.add_argument('--decoder_layers', type=int, default=3, help='Number of Layers for Decoder')
    parser.add_argument('--linear_heads', type=int, default=10, help='Number of linear attention heads')
    parser.add_argument('--linear_hidden_dim', type=int, default=32, help='Dimension of linear attention heads')
    parser.add_argument('--decoder_heads', type=int, default=4, help='Number of headers in the decoder')
    parser.add_argument('--encoder_heads', type=int, default=4, help='Number of Transformer heads')
    parser.add_argument('--gnn_layers', type=int, default=3, help='Layers of GNN')
    parser.add_argument('--encoder_layers', type=int, default=1, help='Layers of Transformer')
    parser.add_argument('--decoder_nums', type=int, default=1, help='Layers of Decoder')
    parser.add_argument('--decoder_dim', type=int, default=128, help='Dimension of Decoder')
    parser.add_argument('--compound_gnn_dim', type=int, default=78, help='Hidden Dimension for Attention')
    parser.add_argument('--pf_dim', type=int, default=1024, help='Hidden Dimension for Positional Feed Forward')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout Rate')
    parser.add_argument('--protein_dim', type=int, default=128, help='Dimension for Protein')
    parser.add_argument('--compound_structure_dim', type=int, default=78, help='Dimension for Compound Structure')
    parser.add_argument('--compound_text_dim', type=int, default=128, help='Dimension for Compound Text')
    parser.add_argument('--compound_pretrained_dim', type=int, default=384, help='Dimension of pretrained for '
                                                                                 'compound language model')
    parser.add_argument('--protein_pretrained_dim', type=int, default=480, help='Dimension of pretrained for protein '
                                                                                'language model')




    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    print(params)
    torch.cuda.manual_seed_all(params.seed)
    s=time()
    for i in range(1, 6):
        run(i,params)
    e=time()
    print(f"Total running time: {round(e - s, 2)}s")
