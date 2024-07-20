import numpy as np
from ml_library.metrics.regression_metrics import calc_mse

def calc_loss(y_true, y_pred, weights, reg=None, l1_coef=0, l2_coef=0):
    loss = calc_mse(y_true, y_pred)
    if reg == "l1":
        loss += l1_coef * np.sum(np.abs(weights))
    if reg == "l2":
        loss += l2_coef * np.sum(np.square(weights))
    if reg == "elasticnet":
        loss += l1_coef * np.sum(np.abs(weights)) + l2_coef * np.sum(np.square(weights))
    return loss
