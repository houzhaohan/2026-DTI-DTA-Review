import os
import pickle
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import load_data
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, average_precision_score, roc_curve
from sklearn import metrics


def save_result(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def train_val_test_validation(dataset: str, preset=None, ex_model=[], n_runs: int = 5) -> None:
    """
    Perform training, validation, and testing for DTI task.
    """
    # DTI任务配置
    data_path = "./dataset/" + dataset + "/"
    eval_metric = "roc_auc"

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

        pred_probs = predictor.predict_proba(test_data_nolab)

        # 预测为标签 1 的概率值
        P = np.array(pred_probs.iloc[:, 1])

        save_result(G, P, result_file)

        auroc = roc_auc_score(G, P)
        precision, recall, _ = metrics.precision_recall_curve(G, P)

        auprc = metrics.auc(recall, precision)

        precision, recall, thresholds = precision_recall_curve(G, P)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # 防止除零错误

        thred_optim = thresholds[5:][np.argmax(f1[5:])]

        y_pred_s = [1 if i else 0 for i in (P >= thred_optim)]
        mcc = metrics.matthews_corrcoef(G, y_pred_s)
        F1 = metrics.f1_score(G, y_pred_s)

        cm1 = confusion_matrix(G, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

        record_file = f"./results/{dataset}/record{run_idx}"
        f = open(record_file, 'w')
        f.write(
            f'Test At epoch: AUROC {auroc} ; AUPRC {auprc} ; Accuracy {accuracy} ; Sensitivity {sensitivity} ; Specificity {specificity} ; MCC {mcc} ; F1 {F1}\n')
        f.write("\n")
        f.flush()

    f.close()


if __name__ == "__main__":
    datasets=["BIOSNAP","BindingDB","unseen_drug","unseen_protein","human_random",'human_cold']
    for dataset in datasets:
        train_val_test_validation(dataset, n_runs=1)

