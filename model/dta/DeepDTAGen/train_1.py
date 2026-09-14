
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import *
from model import DeepDTAGen
from FetterGrad import FetterGrad

from tqdm import tqdm
import sys, os
import time
import pickle
import random


seed = 4221
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
  generator = torch.Generator('cuda').manual_seed(seed)
else:
  generator = torch.Generator().manual_seed(seed)


"""Train the DeepDTAGen model using the specified data and hyperparameters."""

def train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS):
    model.train()

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as t:
        for i, data in enumerate(t):
            optimizer.zero_grad()
            batch = data.batch.to(device)
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            mse_loss = mse_f(Pridection, data.y.view(-1, 1).float().to(device))

            train_ci =0 # get_cindex(Pridection.cpu().detach().numpy(), data.y.view(-1, 1).float().cpu().detach().numpy())

            loss = kl_loss * 0.001 + mse_loss + lm_loss
            # loss.backward()
            # optimizer.step()

            losses = [loss, mse_loss]
            optimizer.ft_backward(losses)
            optimizer.step()
            t.set_postfix(MSE=mse_loss.item(), Train_cindex=train_ci, KL=kl_loss.item(), LM=lm_loss.item())
        msg = f"Epoch {epoch+1}, total loss={loss.item()}, MSE={mse_loss.item()}, KL_loss={kl_loss.item()}, LM={lm_loss.item()}"
        logging(msg, FLAGS)
    return model

def val(model, device, val_loader, dataset, FLAGS):
    """Validate the DeepDTAGen model on the specified data and report the results."""
    print('Validating on {} samples...'.format(len(val_loader.dataset)))
    model.eval()
    total_true = torch.Tensor()
    total_predict = torch.Tensor()
    total_loss = 0

    if dataset == "kiba_dta":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]

    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):  # 注意：epoch需要外部传入，后续会处理
            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            total_true = torch.cat((total_true, data.y.view(-1, 1).cpu()), 0)
            total_predict = torch.cat((total_predict, Pridection.cpu()), 0)

            #auc_values = []
            #for t in thresholds:
            #    auc = get_aupr(G, P, t)
            #    auc_values.append(auc)
            loss = lm_loss + kl_loss
            total_loss += loss.item() * data.num_graphs

    G = total_true.numpy().flatten()
    P = total_predict.numpy().flatten()
    mse_loss = mse(G, P)
    val_ci = 0 # get_cindex(G, P)
    rm2 =0 # get_rm2(G, P)
    return total_loss, mse_loss, val_ci, rm2, G, P

def test(model, device, test_loader, dataset, FLAGS):
    """Test the DeepDTAGen model on the specified data and report the results."""
    print('Testing on {} samples...'.format(len(test_loader.dataset)))
    model.eval()
    total_true = torch.Tensor()
    total_predict = torch.Tensor()
    total_loss = 0

    if dataset == "kiba_dta":
        thresholds = [10.0, 10.50, 11.0, 11.50, 12.0, 12.50]
    else:
        thresholds = [5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.50]

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):

            Pridection, new_drug, lm_loss, kl_loss = model(data.to(device))

            total_true = torch.cat((total_true, data.y.view(-1, 1).cpu()), 0)
            total_predict = torch.cat((total_predict, Pridection.cpu()), 0)

            loss = lm_loss + kl_loss
            total_loss += loss.item() * data.num_graphs

            #auc_values = []
            #for t in thresholds:
             #   auc = get_aupr(G, P,t)
             #   auc_values.append(auc)
    G = total_true.numpy().flatten()
    P = total_predict.numpy().flatten()
    mse_loss = mse(G, P)
    test_ci =0# get_cindex(G, P)
    rm2 =0# get_rm2(G, P)
            
    return total_loss, mse_loss, test_ci, rm2, G, P

def save_result_lxc(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def experiment(FLAGS, dataset, device, roundid):
    logging('Starting program', FLAGS)

    # Hyperparameters
    BATCH_SIZE = 32
    LR = 0.0002
    NUM_EPOCHS = 200

    # Print hyperparameters
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")

    # Log hyperparameters
    msg = f"Dataset {dataset}, Device {device}, batch size {BATCH_SIZE}, learning rate {LR}, epochs {NUM_EPOCHS}"
    logging(msg, FLAGS)

    # Load tokenizer
    with open(f'benchmark/{dataset}/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Load processed data
    processed_data_file_train = f"benchmark/processed/{dataset}_train_1.pt"

    processed_data_file_val = f"benchmark/processed/{dataset}_val_1.pt"

    processed_data_file_test = f"benchmark/processed/{dataset}_test_1.pt"

    if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
        print("Please run create_data.py to prepare data in PyTorch format!")
    else:
        train_data = TestbedDataset(root="benchmark", dataset=f"{dataset}_train_1")
        val_data = TestbedDataset(root="benchmark", dataset=f"{dataset}_val_1")
        test_data = TestbedDataset(root="benchmark", dataset=f"{dataset}_test_1")

        # Prepare PyTorch mini-batches
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize model, optimizer, and loss function
        model = DeepDTAGen(tokenizer).to(device)
        optimizer = FetterGrad(optim.Adam(model.parameters(), lr=LR))
        mse_f = nn.MSELoss()

        # Train model
        best_mse = float('inf')
        for epoch in range(NUM_EPOCHS):
            model = train(model, device, train_loader, optimizer, mse_f, epoch, train_data, FLAGS)

            # 2. 验证
            val_total_loss, val_mse_loss, val_ci, val_rm2,  _, _ = val(model, device, val_loader, dataset, FLAGS)
            print(f"\nEpoch {epoch+1} [Val] Results:")
            print(f"Val MSE: {val_mse_loss.item():.4f}")
            print(f"Val CI: {val_ci:.4f}")
            print(f"Val RM2: {val_rm2:.4f}")
            #print(f"Val AUCs: {', '.join([f'{auc:.4f}' for auc in val_auc_values])}")

            # 3. 测试
            test_total_loss, test_mse_loss, test_ci, test_rm2, G, P = test(model, device, test_loader, dataset, FLAGS)
            print(f"\nEpoch {epoch+1} [Test] Results:")
            print(f"Test MSE: {test_mse_loss.item():.4f}")
            print(f"Test CI: {test_ci:.4f}")
            print(f"Test RM2: {test_rm2:.4f}")
            ##print(f"Test AUCs: {', '.join([f'{auc:.4f}' for auc in test_auc_values])}")

            result_dir_lxc = f'res/{dataset}/round_{roundid}'
            if (os.path.exists(result_dir_lxc) == False):
                os.makedirs(result_dir_lxc)
            result_file_lxc = result_dir_lxc + f"/result_{epoch}.txt"
            save_result_lxc(G, P, result_file_lxc)

            # 记录最佳模型（基于验证集MSE）
            if val_mse_loss < best_mse:
                best_mse = val_mse_loss
                best_test_results = {
                    'epoch': epoch + 1,
                    'test_mse': test_mse_loss,
                    'test_ci': test_ci,
                    'test_rm2': test_rm2
                    ##'test_aucs': test_auc_values
                }
                # 保存最优模型
                filename = f"saved_models/deepdtagen_model_{dataset}_{roundid}_best.pth"
                torch.save(model.state_dict(), filename)
                print(f"\nBest model saved (Val MSE improved to {val_mse_loss.item():.4f})")

            '''
            if (epoch + 1) % 20 == 0:
                # Test model
                total_loss, mse_loss, test_ci, rm2, auc_values, G, P = test(model, device, test_loader, dataset, FLAGS)
                filename = f"saved_models/deepdtagen_model_{dataset}.pth"
                if mse_loss < best_mse:
                    best_mse = mse_loss
                    torch.save(model.state_dict(), filename)
                    print('model saved')

                print(f"MSE: {mse_loss.item():.4f}")
                print(f"CI: {test_ci:.4f}")
                print(f"RM2: {rm2:.4f}")
                print(f"AUCs: {', '.join([f'{auc:.4f}' for auc in auc_values])}")
            '''
        # Save estimated and true labels
        folder_path = "Affinities/"
        np.savetxt(folder_path + f"estimated_labels_{dataset}_{roundid}.txt", P)
        np.savetxt(folder_path + f"true_labels_{dataset}_{roundid}.txt", G)

        logging('Program finished', FLAGS)

        # 测试保存模型
        print("\nEvaluating best model on test set...")
        # 加载最佳模型
        model.load_state_dict(torch.load(f"saved_models/deepdtagen_model_{dataset}_{roundid}_best.pth"))
        model.eval()

        test_total_loss, test_mse_loss, test_ci, test_rm2, G, P = test(model, device, test_loader, dataset, FLAGS)
        result_dir = f"./results/{dataset}_kiba/round_{roundid}/"
        os.makedirs(result_dir, exist_ok=True)
        result_file = result_dir + "/result1.txt"

        f = open(result_file, "w")
        for i in range(len(G)):
            f.write(str(G[i]) + " " + str(P[i]) + "\n")
        f.close()

        mse_loss = mse(G, P)
        concordance_index = get_cindex(G, P)
        rm2_value = get_rm2(G, P)

        record_file = f"./results/{dataset}_kiba/record_{roundid}"
        f = open(record_file, 'w')
        f.write(f'Test At epoch: MSE {str(mse_loss)} ; R2 {str(rm2_value)}; CI {str(concordance_index)}\n\n')
        f.flush()

if __name__ == "__main__":

    datasets = ['davis_dta', 'kiba_dta']
    #dataset_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dataset = datasets[1]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #device = torch.device("cuda:0")

    FLAGS = lambda: None
    FLAGS.log_dir = 'logs'
    FLAGS.dataset_name = f'dataset_{dataset}_{int(time.time())}'

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    os.makedirs('Affinities', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    for i in range(1, 6):
        experiment(FLAGS, dataset, device,i)

