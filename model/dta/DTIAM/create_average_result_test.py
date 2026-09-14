import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, roc_auc_score, average_precision_score, roc_curve

def read_result(result_file):

    f = open(result_file, "r")
    text = f.read()
    f.close()

    pred = []
    real = []

    for line in text.splitlines():
        values = line.strip().split()
        real.append(float(values[0]))
        pred.append(float(values[1]))

    return np.array(pred), np.array(real)

def save_result(G, P, result_file):

    f = open(result_file, "w")

    for i in range(len(G)):
        f.write(str(G[i]) + " " + str(P[i]) + "\n")
    f.close()

def get_average_result(workdir, round_times, iteration_number):

    result_file = workdir + "/round1/result" + str(iteration_number) + ".txt"

    pred, real = read_result(result_file)

    for i in range(2, round_times + 1):

        result_file = workdir + "/round" + str(i) + "/result" + str(iteration_number) + ".txt"

        s_pred, _ = read_result(result_file)
        pred = pred + s_pred

    pred = pred/round_times

    return pred, real

def create_result(workdir, round_times, max_iteration):

    best_aupr = 0
    best_epoch = 0

    res = workdir + "/final_record"
    f = open(res, 'w')

    for epoch in range(1, max_iteration + 1):

        P, G = get_average_result(workdir, round_times, epoch)

        result_dir = workdir + "/final_result/test"
        if (os.path.exists(result_dir) == False):
            os.makedirs(result_dir)
        result_file = result_dir + "/result" + str(epoch) + ".txt"
        save_result(G, P, result_file)

        auroc = roc_auc_score(G, P)

        precision, recall, thresholds = metrics.precision_recall_curve(G, P)

        auprc = metrics.auc(recall, precision)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # 防止除零错误

        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        print(thred_optim)
        y_pred_s = [1 if i else 0 for i in (P >= thred_optim)]
        mcc = metrics.matthews_corrcoef(G, y_pred_s)
        F1 = metrics.f1_score(G, y_pred_s)

        cm1 = confusion_matrix(G, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        specificity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        sensitivity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        print('***')
        print('Test At epoch: ', epoch, 'AUROC', auroc, '; AUPRC', auprc, '; Accuracy', accuracy,
              '; Sensitivity', sensitivity, '; Specificity', specificity, '; MCC', mcc, '; F1', F1)
        print('***')

        f.write(f'Test At epoch: {epoch}, AUROC {auroc} ; AUPRC {auprc} ; Accuracy {accuracy} ; Sensitivity {sensitivity} ; Specificity {specificity} ; MCC {mcc} ; F1 {F1}\n')
        f.write("\n")
        f.flush()

    f.write("Best epoch " + str(best_epoch))

    f.close()


workdir = sys.argv[1]
round_times = 5
max_iteration = 1

create_result(workdir, round_times, max_iteration)













