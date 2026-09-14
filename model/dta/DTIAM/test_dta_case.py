import os
import pickle

import numpy as np
from autogluon.tabular import TabularPredictor
from utils import load_data_test
from math import sqrt
from scipy import stats


def save_result(P, result_file):
    """
    保存预测结果：
    第一列：真实值
    第二列：预测值
    """
    with open(result_file, "w") as f:
        for i in range(len(P)):
            f.write(str(P[i]) + "\n")



def predict_with_saved_model(dataset: str, n_runs: int = 5) -> None:
    """
    不训练模型，直接加载已经保存的模型进行预测。

    每个 run 会：
    1. 加载 ./models/{dataset}/run_xx/ 下的模型
    2. 对 test_data 进行预测
    3. 保存预测结果 result1.txt
    4. 保存 MSE、RM2、CI 指标
    """

    data_path = "./dataset/" + dataset + "/"

    # 加载特征和数据
    comp_feat = pickle.load(open(data_path + "features/compound_features.pkl", "rb"))
    prot_feat = pickle.load(open(data_path + "features/protein_features.pkl", "rb"))

    # 这里只是为了得到 test_data
    test_data = load_data_test(data_path, comp_feat, prot_feat)

    test_data_nolab = test_data.drop(columns=["y"])

    for run_idx in range(1, n_runs + 1):
        run_name = f"run_{run_idx:02d}"

        print(f"\nLoading model for {dataset} - {run_name}...")

        model_path = f"./models/kiba_dta_clean/{run_name}/"

        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            continue

        # 加载已经训练好的模型
        predictor = TabularPredictor.load(model_path)

        # DTA 是回归任务，用 predict，不用 predict_proba
        P = np.array(predictor.predict(test_data_nolab))

        # 保存预测结果
        result_dir = f"./results/{dataset}/round{run_idx}/"
        os.makedirs(result_dir, exist_ok=True)

        result_file = os.path.join(result_dir, "result1.txt")
        save_result(P, result_file)



if __name__ == "__main__":

    datasets = ["drugbank_case"]

    for dataset in datasets:
        predict_with_saved_model(dataset, n_runs=1)