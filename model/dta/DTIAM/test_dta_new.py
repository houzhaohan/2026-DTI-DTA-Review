import os
import pickle

import numpy as np
from autogluon.tabular import TabularPredictor
from utils import load_data
from math import sqrt
from scipy import stats


def save_result(G, P, result_file):
    """
    保存预测结果：
    第一列：真实值
    第二列：预测值
    """
    with open(result_file, "w") as f:
        for i in range(len(G)):
            f.write(str(G[i]) + " " + str(P[i]) + "\n")


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


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


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    y_obs_mean = np.mean(y_obs)

    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 - r02))))


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
    train_data, val_data, test_data = load_data(data_path, comp_feat, prot_feat)

    test_data_nolab = test_data.drop(columns=["y"])
    G = np.array(test_data["y"])

    for run_idx in range(1, n_runs + 1):
        run_name = f"run_{run_idx:02d}"

        print(f"\nLoading model for {dataset} - {run_name}...")

        model_path = f"./models/{dataset}/{run_name}/"

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
        save_result(G, P, result_file)

        # 计算指标
        mse_loss = mse(G, P)
        concordance_index = get_cindex(G, P)
        rm2_value = get_rm2(G, P)

        # 保存指标
        record_file = f"./results/{dataset}/record{run_idx}"

        with open(record_file, "w") as f:
            f.write(
                f"Test: MSE {str(mse_loss)} ; "
                f"RM2 {str(rm2_value)} ; "
                f"CI {str(concordance_index)}\n\n"
            )


if __name__ == "__main__":

    datasets = ["kiba_dta"]

    for dataset in datasets:
        predict_with_saved_model(dataset, n_runs=1)