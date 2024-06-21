import random
import numpy as np

class LogRegression:
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None, reg=None, l1_coef=None, l2_coef=None, sgd_sample=None, random_state=42):
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
        
    def fit(self, data_train, y_train, verbose=False):
        random.seed(self.random_state)
        data = data_train.copy()
        data["intercept"] = np.ones(data.shape[0])
        self.weights = np.ones(data.shape[1])
        for iter_num in range(self.n_iter):
            x_batch, y_batch = self.get_batch(data, y_train)
            predictions = 1 / (1 + np.exp(-np.dot(x_batch, self.weights)))
            loss = self.calc_loss(y_train, 1 / (1 + np.exp(-np.dot(data, self.weights))))
            grad = self.calc_grad(x_batch, y_batch, predictions)
            self.weights -= self.calc_learning_rate(iter_num + 1) * grad
            if verbose and (iter_num % verbose == 0) and not self.metric:
                print(f"{iter_num} | loss: {loss}")
                continue
            if self.metric == "roc_auc":
                metric = self.calc_metric(y_train, self.predict_proba(data))
            else:
                metric = self.calc_metric(y_train, self.predict(data))
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

    def calc_metric(self, y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        false_negatives = np.sum((y_pred == 0) & (y_true == 1))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        if self.metric == "accuracy":
            return np.sum(y_true == y_pred) / y_true.shape[0]
        if self.metric == "precision":
            return precision
        if self.metric == "recall":
            return recall
        if self.metric == "f1":
            return 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        if self.metric == "roc_auc":
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

    def calc_loss(self, y_true, y_pred):
        eps = 1e-15
        loss = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))
        if self.reg == "l1":
            loss += self.l1_coef * np.sum(np.abs(self.weights))
        if self.reg == "l2":
            loss += self.l2_coef * np.sum(np.square(self.weights))
        if self.reg == "elasticnet":
            loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(np.square(self.weights))
        return loss

    def calc_grad(self, data, y_true, y_pred):
        grad = np.dot((y_pred - y_true).T, data) / data.shape[0]
        if self.reg == "l1":
            grad += self.l1_coef * np.sign(self.weights)
        if self.reg == "l2":
            grad += self.l2_coef * 2 * self.weights
        if self.reg == "elasticnet":
            grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights
        return grad

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
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"
"
