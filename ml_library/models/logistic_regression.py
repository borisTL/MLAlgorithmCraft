import random
import numpy as np
from ml_library.metrics.classification_metrics import calc_accuracy, calc_precision, calc_recall, calc_f1, calc_roc_auc
from  ml_library.utils.gradient_calculations import calc_grad_logistic
from ml_library.utils.loss_functions import calc_loss

class LogRegression:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
        self.calc_metric = {
            "accuracy": calc_accuracy,
            "precision": calc_precision,
            "recall": calc_recall,
            "f1": calc_f1,
            "roc_auc": calc_roc_auc,
            None: lambda x, y: None,
        }
        
    def fit(self, data_train, y_train, verbose=False):
        random.seed(self.random_state)
        data = data_train.copy()
        data["intercept"] = np.ones(data.shape[0])
        self.weights = np.ones(data.shape[1])
        for iter_num in range(self.n_iter):
            x_batch, y_batch = self.get_batch(data, y_train)
            predictions = 1 / (1 + np.exp(-np.dot(x_batch, self.weights)))
            loss = calc_loss(y_train, 1 / (1 + np.exp(-np.dot(data, self.weights))), self.weights, self.reg, self.l1_coef, self.l2_coef)
            grad = calc_grad_logistic(x_batch, y_batch, predictions, self.weights, self.reg, self.l1_coef, self.l2_coef)
            self.weights -= self.calc_learning_rate(iter_num + 1) * grad
            if verbose and (iter_num % verbose == 0) and not self.metric:
                print(f"{iter_num} | loss: {loss}")
                continue
            if self.metric == "roc_auc":
                metric = self.calc_metric["roc_auc"](y_train, self.predict_proba(data))
            else:
                metric = self.calc_metric[self.metric](y_train, self.predict(data))
            self.best_score = metric
            if verbose and (iter_num % verbose == 0):
                print(f"{iter_num} | loss: {loss} | {self.metric}: {metric}")
                
    def predict_proba(self, x_test):
        data = x_test.copy()
        data["intercept"] = np.ones(data.shape[0])
        predictions = 1 / (1 + np.exp(-np.dot(data, self.weights)))
        return predictions

    def predict(self, x_test):
        predictions = self.predict_proba(x_test)
        return np.where(predictions > 0.5, 1, 0)

    def calc_learning_rate(self, iteration):
        return self.learning_rate if isinstance(self.learning_rate, (int, float)) else self.learning_rate(iteration)

    def get_batch(self, data, labels):
        if isinstance(self.sgd_sample, int):
            sample_rows_idx = random.sample(range(data.shape[0]), self.sgd_sample)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        if isinstance(self.sgd_sample, float):
            sample_size = int(data.shape[0] * self.sgd_sample)
            sample_rows_idx = random.sample(range(data.shape[0]), sample_size)
            return data.iloc[sample_rows_idx], labels.iloc[sample_rows_idx]
        return data, labels

    def get_coef(self):
        return self.weights[:-1]

    def get_best_score(self):
        return self.best_score

    def __repr__(self):
        return f"LogRegression class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
