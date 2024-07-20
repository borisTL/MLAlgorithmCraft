import numpy as np

def apply_regularization(grad, weights, reg, l1_coef, l2_coef):
    if reg == "l1":
        grad += l1_coef * np.sign(weights)
    if reg == "l2":
        grad += l2_coef * 2 * weights
    if reg == "elasticnet":
        grad += l1_coef * np.sign(weights) + l2_coef * 2 * weights
    return grad

def calc_grad_linear(data, y_true, y_pred, weights, reg=None, l1_coef=0, l2_coef=0):
    n_rows = y_true.shape[0]
    delta = np.transpose(y_pred - y_true)
    grad = 2 / n_rows * np.dot(delta, data)
    grad = apply_regularization(grad, weights, reg, l1_coef, l2_coef)
    return grad

def calc_grad_logistic(data, y_true, y_pred, weights, reg=None, l1_coef=0, l2_coef=0):
    grad = np.dot((y_pred - y_true).T, data) / data.shape[0]
    grad = apply_regularization(grad, weights, reg, l1_coef, l2_coef)
    return grad
