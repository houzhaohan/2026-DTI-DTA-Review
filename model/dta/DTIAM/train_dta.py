import os
import pickle
import sys

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from utils import load_data, rmse, mse, pearson, spearman, ci, roc_auc, pr_auc, mcc_f1
from sklearn.metrics import r2_score


def train_val_test_validation(dataset: str, preset=None, ex_model=[]) -> None:
    """
    Perform training, validation, and testing for DTI task.
    """
    # DTI任务配置
    dataset_path = "./dataset/dta/" + dataset + "/"
    eval_metric = None
    res_metrics = ["RMSE", "MSE", "Pearson", "Spearman", "CI"]

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

    pred_scores = predictor.predict(test_data_nolab)

    G, P = np.array(test_data["y"]), np.array(pred_scores)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    R2=r2_score(G, P)

    print(
        f"RMSE: {ret[0]}, MSE: {ret[1]}, Pearson: {ret[2]}, Spearman: {ret[3]}, CI: {ret[4]}, R2: {R2}"
    )
    res_all.loc[0] = ret

    # 保存结果
    os.makedirs("./results/dta/", exist_ok=True)
    res_all.to_csv(f"./results/dta/{dataset}_test_results.csv", index=None, sep="\t")
    print("Test results saved.")


if __name__ == "__main__":
    
    datasets=["davis_dta","kiba_dta"]
    for dataset in datasets:
        train_val_test_validation(dataset)
