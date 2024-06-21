import numpy as np
import pandas as pd
from numpy import random

class LineRegression :
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.calc_metric = {
            "mae": self.calc_mae,
            "mse": self.calc_mse,
            "rmse": self.calc_rmse,
            "mape": self.calc_mape,
            "r2": self.calc_r2,
            None: lambda x, y: None,
        }
        self.best_score = None
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        self.random_state = random_state
        self.sample_size = sgd_sample

    def calc_mae(self, y_true, y_pred):
        return np.mean(np.abs(y_pred - y_true))

    def calc_mse(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def calc_rmse(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    def calc_r2(self, y_true, y_pred):
        return 1 - (np.sum(np.square(y_true - y_pred)) / np.sum(np.square(y_true - np.mean(y_true))))

    def calc_mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def calc_loss(self, y_true, y_pred):
        loss = self.calc_mse(y_true, y_pred)
        if self.reg == "l1":
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == "l2":
            loss += self.l2_coef * np.sum(np.square(self.weights))
        if self.reg == "elasticnet":
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(np.square(self.weights))
        return loss

    def calc_grad(self, data, y_true, predictions):
        n_rows = y_true.shape[0]
        delta = np.transpose(predictions - y_true)
        grad = 2 / n_rows * np.dot(delta, data)
        if self.reg == "l1":
            grad += self.l1_coef * np.sign(self.weights)
        if self.reg == "l2":
            grad += self.l2_coef * 2 * self.weights
        if self.reg == "elasticnet":
            grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        return grad

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
            loss = self.calc_loss(labels, np.dot(data, self.weights))
            grad = self.calc_grad(data_batch, y_batch, predictions)
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
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
