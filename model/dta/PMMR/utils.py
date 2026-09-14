import math
import os
import numpy as np
from math import sqrt
from scipy import stats

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


def ci(y, p):
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

    mult = sum((y_pred-y_pred_mean) * (y_obs-y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))
    y_pred_sq = sum((y_pred-y_pred_mean) * (y_pred-y_pred_mean))

    return mult / float(y_obs_sq*y_pred_sq)

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs-(k*y_pred)) * (y_obs-(k*y_pred)))
    down= sum((y_obs-y_obs_mean) * (y_obs-y_obs_mean))

    return 1 - (upp/float(down))

def rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2-r02))))


def mae(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("error")

    total_error = np.sum(np.abs(predictions - targets))
    mae = total_error / len(predictions)
    return mae
