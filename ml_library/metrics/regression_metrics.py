import numpy as np

def calc_mae(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))

def calc_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

def calc_rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def calc_r2(y_true, y_pred):
    return 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

def calc_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    