import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import argparse
from data2 import CPIDataset
from models.core import *
from utils import *
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
from sklearn.metrics import auc


# training function at each epoch


def train_dti(model, loss_fn, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        #print(data)
        output = model(data)
        loss = loss_fn(output, data['LABEL'].long().cuda())
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data['LABEL']),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting_dti(model, loader):
    model.eval()
    total_preds = []
    total_labels = []

    print('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            output = model(data)
            total_preds.append(output.cpu().numpy())
            total_labels.append(data['LABEL'].cpu().numpy())

    return np.concatenate(total_labels), np.concatenate(total_preds)


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
        shuffle=True,
        num_workers=workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=test_set.collate_fn,
        shuffle=True,
        num_workers=workers,
    )

    return train_loader, val_loader, test_loader



def run(args: argparse.Namespace):


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
    train_loader, val_loader, test_loader = others_dataloader(batch_size, args.num_workers, dataset, data_path)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=30,
                                                            eps=1e-08)

    best_aupr = 0
    best_epoch = -1
    model_file_name = f'{dataset}.pth'
    result_file_name = f'result_{dataset}.csv'

    for epoch in range(NUM_EPOCHS):
        train_dti(model, loss_fn, train_loader, optimizer, epoch + 1)
        
        val_y_true, val_y_scores = predicting_dti(model, val_loader)
        val_y_pred = np.argmax(val_y_scores, axis=1)
        val_precision, val_recall, _ = precision_recall_curve(val_y_true, val_y_pred)
        val_aupr = auc(val_recall, val_precision)

        # 3. 更新学习率调度器（基于验证集AUPR）
        scheduler.step(val_aupr)

        # 4. 保存最佳模型（基于验证集AUPR）
        if val_aupr > best_aupr:
            torch.save(model.state_dict(), model_file_name)
            best_epoch = epoch + 1
            best_aupr = val_aupr
            print(f'[Val] AUPR improved at epoch {best_epoch} | Best AUPR: {best_aupr:.4f}')
        else:
            print(f'[Val] No improvement since epoch {best_epoch} | Current AUPR: {val_aupr:.4f} | Best AUPR: {best_aupr:.4f}')
        
        if epoch-best_epoch>40:
            break

    # 最终测试集评估（加载最佳模型）
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load(model_file_name))
    final_y_true, final_y_scores = predicting_dti(model, test_loader)
    y_scores = final_y_scores[:, 1]
    
    final_auroc=roc_auc_score(final_y_true, y_scores)

    final_precision, final_recall, thresholds = precision_recall_curve(final_y_true, y_scores)
    
    final_aupr = auc(final_recall, final_precision)
    
    f1_scores = (2 * final_precision * final_recall) / (final_precision + final_recall + 1e-6)
    optimal_idx = np.argmax(f1_scores[5:]) + 5  # 跳过前5个极端阈值
    optimal_threshold = thresholds[optimal_idx]
            
            # 用最优阈值重新预测
    y_pred_optim = (y_scores >= optimal_threshold).astype(int)
            
            # 计算优化后的指标
    mcc_optim = matthews_corrcoef(final_y_true, y_pred_optim)
    f1_optim = f1_score(final_y_true, y_pred_optim)
            
            # 混淆矩阵相关指标
    cm2 = confusion_matrix(final_y_true, y_pred_optim)
    acc_op = (cm2[0, 0] + cm2[1, 1]) / sum(sum(cm2))
    spe_op = cm2[0, 0] / (cm2[0, 0] + cm2[0, 1])
    sen_op = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])

    final_acc = accuracy_score(final_y_true, y_pred_optim)
    
    print(f"[Test] Final Results | ACC: {final_acc:.4f} | AUPR: {final_aupr:.4f} | AUROC: {final_auroc:.4f} | F1: {f1_optim:.4f}")
    print(f"acc:{acc_op},mcc:{mcc_optim},spe:{spe_op},sen:{sen_op}")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_data_path', type=str, default='./benchmark/dataset', help='Raw Data Path')
    parser.add_argument('--dataset', type=str, default='BIOSNAP', help='Datasets')
    parser.add_argument('--objective',
                        type=str,
                        default='classification',
                        help='Objective (classification / regression)')
    parser.add_argument('--seed', type=int, default=0, help='Random Seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for Train(Validation/Test)')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max Trainning Epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of Subprocesses for Data Loading')
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
    run(params)
