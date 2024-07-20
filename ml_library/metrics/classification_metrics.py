import numpy as np

def calc_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / y_true.shape[0]

def calc_precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0

def calc_recall(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

def calc_f1(y_true, y_pred):
    precision = calc_precision(y_true, y_pred)
    recall = calc_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

def calc_roc_auc(y_true, y_pred):
    y_pred = np.round(y_pred, 10)
    score_sorted = sorted(zip(y_true, y_pred), key=lambda x: x[1])
    ranked = 0
    for i in range(len(score_sorted) - 1):
        cur_true = score_sorted[i][0]
        if cur_true == 1:
            continue
        for j in range(i + 1, len(score_sorted)):
            if score_sorted[j][0] == 1 and score_sorted[j][1] == score_sorted[i][1]:
                ranked += 0.5
            elif score_sorted[j][0] == 1:
                ranked += 1
    return ranked / np.sum(y_true == 1) / np.sum(y_true == 0)
