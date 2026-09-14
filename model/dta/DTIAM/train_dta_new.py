import os
import pickle
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import load_data
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, average_precision_score, roc_curve
from sklearn import metrics
from math import sqrt
from scipy import stats


def save_result(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse

def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp

def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def get_cindex(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()

    y_diff = y[:, None] - y[None, :]
    p_diff = p[:, None] - p[None, :]

    mask = y_diff > 0

    if np.sum(mask) == 0:
        return np.nan

    score = (p_diff > 0).astype(float) + 0.5 * (p_diff == 0).astype(float)

    return np.sum(score[mask]) / np.sum(mask)

def r_squared_error(y_obs,y_pred):

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))

def squared_error_zero(y_obs,y_pred):

    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_rm2(ys_orig,ys_line):

    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2-r02))))

def train_val_test_validation(dataset: str, preset=None, ex_model=[], n_runs: int = 5) -> None:
    """
    Perform training, validation, and testing for DTI task.
    """
    # DTI任务配置
    data_path = "./dataset/" + dataset + "/"
    eval_metric = None

    # 加载特征和数据
    comp_feat = pickle.load(open(data_path + "features/compound_features.pkl", "rb"))
    prot_feat = pickle.load(open(data_path + "features/protein_features.pkl", "rb"))

    # 合并特征
    train_data, val_data, test_data = load_data(data_path, comp_feat, prot_feat)

    test_data_nolab = test_data.drop(columns=["y"])
    G = np.array(test_data["y"])

    for run_idx in range(1, n_runs + 1):

        run_seed = 2024 + run_idx
        run_name = f"run_{run_idx:02d}"
        print(f"\nTraining {dataset} - {run_name}...")

        model_path = f"./models/{dataset}/{run_name}/"
        os.makedirs(model_path, exist_ok=True)

        # 每次预测结果保存路径
        result_dir = f"./results/{dataset}/round{run_idx}/"
        os.makedirs(result_dir, exist_ok=True)
        result_file = result_dir + "/result1.txt"

        predictor = TabularPredictor(
            label="y",
            eval_metric=eval_metric,
            path=model_path
        ).fit(
            train_data=train_data,
            tuning_data=val_data,
            excluded_model_types=ex_model,
            presets=preset
        )

        # 预测为标签 1 的概率值
        P = np.array(predictor.predict(test_data_nolab))

        save_result(G, P, result_file)

        mse_loss = mse(G, P)
        concordance_index = get_cindex(G, P)
        rm2_value = get_rm2(G, P)

        record_file = f"./results/{dataset}/record{run_idx}"
        f = open(record_file, 'w')
        f.write(f'Test At epoch: MSE {str(mse_loss)} ; R2 {str(rm2_value)}; CI {str(concordance_index)}\n\n')
        f.flush()

    f.close()


if __name__ == "__main__":

    datasets = ["kiba_dta_clean"]
    for dataset in datasets:
        train_val_test_validation(dataset, n_runs=1)

