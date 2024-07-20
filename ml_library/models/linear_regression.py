import numpy as np
import pandas as pd
from numpy import random
from ml_library.metrics.regression_metrics import calc_mae, calc_mse, calc_rmse, calc_r2, calc_mape
from ml_library.utils.gradient_calculations import calc_grad_linear
from ml_library.utils.loss_functions import calc_loss

class LineRegression:
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.calc_metric = {
            "mae": calc_mae,
            "mse": calc_mse,
            "rmse": calc_rmse,
            "mape": calc_mape,
            "r2": calc_r2,
            None: lambda x, y: None,
        }
        self.best_score = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.random_state = random_state
        self.sample_size = sgd_sample

    def calc_learning_rate(self, iteration):
        if isinstance(self.learning_rate, (float, int)):
            return self.learning_rate
        return self.learning_rate(iteration)

    def get_batch(self, data, labels):
        if isinstance(self.sample_size, int):
            sample_rows_idx = random.sample(range(data.shape[0]), self.sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        if isinstance(self.sample_size, float):
            sample_size = int(data.shape[0] * self.sample_size)
            sample_rows_idx = random.sample(range(data.shape[0]), sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        return data, labels

    def fit(self, train_data, labels, verbose=False):
        random.seed(self.random_state)
        data = train_data.copy()
        n_rows = data.shape[0]
        n_features = data.shape[1]
        data["intercept"] = [1] * n_rows
        self.weights = np.ones(n_features + 1)

        for iteration in range(self.n_iter):
            data_batch, y_batch = self.get_batch(data, labels)
            predictions = np.dot(data_batch, self.weights)
            loss = calc_loss(labels, np.dot(data, self.weights), self.weights, self.reg, self.l1_coef, self.l2_coef)
            grad = calc_grad_linear(data_batch, y_batch, predictions, self.weights, self.reg, self.l1_coef, self.l2_coef)
            self.weights -= self.calc_learning_rate(iteration + 1) * grad
            updated_predictions = np.dot(data, self.weights)
            metric_value = self.calc_metric[self.metric](labels, updated_predictions)
            self.best_score = metric_value
            if not verbose:
                continue
            if iteration % verbose == 0:
                print(f"{iteration} | loss: {loss} | {self.metric}: {metric_value}")

    def predict(self, predict_data):
        data = predict_data.copy()
        data["intercept"] = [1] * predict_data.shape[0]
        predictions = np.dot(data, self.weights)
        return predictions

    def get_best_score(self):
        return self.best_score

    def get_coef(self):
        return self.weights[:-1]

    def __repr__(self):
        return f"LineRegression class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
