from typing import Tuple, Dict
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


def load_data(data_path: str, comp_feat: Dict, prot_feat: Dict) -> Tuple:
    """Load training and testing data."""
    print("Loading data ...")
    train = pd.read_csv(data_path + "train.csv")[["SMILES_index","Protein_index","Y"]]
    val = pd.read_csv(data_path + "val.csv")[["SMILES_index","Protein_index","Y"]]
    test = pd.read_csv(data_path + "test.csv")[["SMILES_index","Protein_index","Y"]]
    train.columns = ["cid", "pid", "label"]
    val.columns = ["cid", "pid", "label"]
    test.columns = ["cid", "pid", "label"]
    print(train)
    #print(comp_feat)
    return pack(train, comp_feat, prot_feat), pack(val, comp_feat, prot_feat), pack(test, comp_feat, prot_feat)


def load_data_test(data_path: str, comp_feat: Dict, prot_feat: Dict) -> Tuple:
    """Load training and testing data."""
    print("Loading data ...")
    test = pd.read_csv(data_path + "test.csv")[["SMILES_index","Protein_index","Y"]]
    test.columns = ["cid", "pid", "label"]

    return pack(test, comp_feat, prot_feat)


def pack(data: pd.DataFrame, comp_feat: Dict, prot_feat: Dict) -> pd.DataFrame:
    """Pack compound and protein features into a dataframe."""
    vecs = []
    for i in range(len(data)):
        cid, pid = data.loc[i, ['cid','pid']]
        cid=int(cid)
        pid=int(pid)
        vecs.append(list(comp_feat[cid]) + list(prot_feat[pid]))
    vecs_df = pd.DataFrame(vecs)
    vecs_df["y"] = data["label"]
    return vecs_df


def roc_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the ROC AUC score."""
    fpr, tpr, _ = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc


def pr_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Compute the Precision-Recall AUC score."""
    precision, recall, _ = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)

    return pr_auc

def mcc_f1(G, P):
    precision, recall, thresholds = metrics.precision_recall_curve(G, P)
    
    f1 = (2 * precision * recall) / (precision + recall + 1e-6)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]
                        
    y_pred_s = [1 if i else 0 for i in (P >= thred_optim)]
    mcc = metrics.matthews_corrcoef(G, y_pred_s)
    F1=metrics.f1_score(G,y_pred_s)
                                
    cm1 = metrics.confusion_matrix(G, y_pred_s)
    accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
    specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return mcc, F1, accuracy, sensitivity, specificity



def rmse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Root Mean Squared Error."""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Mean Squared Error."""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def pearson(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Pearson correlation coefficient."""
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Spearman correlation coefficient."""
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y: np.ndarray, f: np.ndarray) -> float:
    """Compute the Concordance Index."""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

