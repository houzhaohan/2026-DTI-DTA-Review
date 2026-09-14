import os
import pickle
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import load_data, rmse, mse, pearson, spearman, ci, roc_auc, pr_auc, mcc_f1


def train_val_test_validation(dataset: str, preset=None, ex_model=[]) -> None:
    """
    Perform training, validation, and testing for DTI task.
    """
    # DTI任务配置
    dataset_path = "./dataset/" + dataset + "/"
    eval_metric = "roc_auc"
    res_metrics = ["AUROC", "AUPR", "MCC", "F1", "ACC", "SEN", "SPE"]

    # 加载特征和数据
    comp_feat = pickle.load(open(dataset_path + "features/compound_features.pkl", "rb"))
    prot_feat = pickle.load(open(dataset_path + "features/protein_features.pkl", "rb"))
    data_path = dataset_path 

    # 合并特征
    train_data, val_data, test_data = load_data(data_path, comp_feat, prot_feat)


    print(f"Training the model on {dataset} dataset...")
    
    # 训练模型（使用验证集调参）
    predictor = TabularPredictor(
        label="y", 
        eval_metric=eval_metric
    ).fit(
        train_data=train_data,
        tuning_data=val_data,  # 显式指定验证集
        excluded_model_types=ex_model,
        presets=preset
    )

    # 在测试集上评估
    test_data_nolab = test_data.drop(columns=["y"])
    res_all = pd.DataFrame(columns=res_metrics)

    pred_probs = predictor.predict_proba(test_data_nolab)
    auroc = roc_auc(np.array(test_data["y"]), np.array(pred_probs.iloc[:, 1]))
    aupr = pr_auc(np.array(test_data["y"]), np.array(pred_probs.iloc[:, 1]))
    mcc, f1, accuracy, sensitivity, specificity = mcc_f1(
        np.array(test_data["y"]), np.array(pred_probs.iloc[:, 1])
    )
    print(f"AUROC: {auroc}, AUPR: {aupr}")
    print(f"MCC: {mcc}, F1: {f1}")
    print(f"ACC: {accuracy}, SEN: {sensitivity}, SPE: {specificity}")
    res_all.loc[0] = [auroc, aupr, mcc, f1, accuracy, sensitivity, specificity]

    # 保存结果
    os.makedirs("./results/", exist_ok=True)
    res_all.to_csv(f"./results/{dataset}_test_results.csv", index=None, sep="\t")
    print("Test results saved.")


if __name__ == "__main__":
    datasets=["BIOSNAP","BindingDB","unseen_drug","unseen_protein","human_random",'human_cold']
    for dataset in datasets:
        train_val_test_validation(dataset)

