import numpy as np
import pandas as pd
import random

class LogisticRegression:
    
    def __init__(self, n_iter=10, learning_rate=0.01, reg=None, l1_coef=0, l2_coef=0, metric=None, sgd_sample=None, random_state=42):
        self._iter = n_iter
        self._learning_rate = learning_rate
        self.weights = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.metric = metric
        self.best_score = None
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def fit(self, X, y, verbose=False):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        X.insert(0, "bias", pd.Series(1, index=X.index))
        self.weights = np.ones(X.shape[1])
        
        if verbose:
            start_loss = self.compute_loss(self.predict_proba(X), y)
            metric_value = self.get_metric_score(self.predict(X), y) if self.metric else None
            output = f"start | loss: {start_loss:.2f}"
            if metric_value is not None:
                output += f" | {self.metric}: {metric_value:.2f}"
            print(output)

        for i in range(1, self._iter + 1):
            current_learning_rate = self._learning_rate(i) if callable(self._learning_rate) else self._learning_rate
            
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, float) and 0 < self.sgd_sample < 1:
                    sample_size = int(self.sgd_sample * X.shape[0])
                else:
                    sample_size = self.sgd_sample
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
            else:
                X_sample = X
                y_sample = y
            
            y_predicted = self.predict_proba(X_sample)
            gradient = self.compute_gradient(X_sample, y_sample, y_predicted)
            self.weights -= current_learning_rate * gradient

            loss = self.compute_loss(self.predict_proba(X), y)
            if verbose and i % verbose == 0:
                metric_value = self.get_metric_score(self.predict(X), y) if self.metric else None
                output = f"{i} | loss: {loss:.2f} | learning_rate: {current_learning_rate:.5f}"
                if metric_value is not None:
                    output += f" | {self.metric}: {metric_value:.2f}"
                print(output)

            current_metric_value = self.get_metric_score(self.predict(X), y) if self.metric else None
            if self.metric and current_metric_value is not None:
                if self.best_score is None or \
                   (self.metric in ['loss'] and current_metric_value < self.best_score) or \
                   (self.metric not in ['loss'] and current_metric_value > self.best_score):
                    self.best_score = current_metric_value

    def get_coef(self):
        return np.array(self.weights)
        
    def predict_proba(self, X):
        X_with_bias = pd.DataFrame(X)
        if "bias" not in X_with_bias.columns:
            X_with_bias.insert(0, "bias", pd.Series(1, index=X_with_bias.index))
        linear_combination = np.dot(X_with_bias, self.weights)
        probabilities = 1 / (1 + np.exp(-linear_combination))
        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def compute_loss(self, y_predicted, y):
        eps = 1e-15
        y_predicted = np.clip(y_predicted, eps, 1 - eps)
        log_loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
        
        if self.reg == 'l1':
            penalty = self.l1_coef * np.sum(np.abs(self.weights[1:]))
        elif self.reg == 'l2':
            penalty = self.l2_coef * np.sum(self.weights[1:] ** 2)
        elif self.reg == 'elasticnet':
            l1_penalty = self.l1_coef * np.sum(np.abs(self.weights[1:]))
            l2_penalty = self.l2_coef * np.sum(self.weights[1:] ** 2)
            penalty = l1_penalty + l2_penalty
        else:
            penalty = 0
        
        return log_loss + penalty
    
    def compute_gradient(self, X, y, y_predicted):
        gradient = np.dot(X.T, y_predicted - y) / len(y)
        
        if self.reg == 'l1':
            reg_term = self.l1_coef * np.sign(self.weights)
        elif self.reg == 'l2':
            reg_term = self.l2_coef * 2 * self.weights
        elif self.reg == 'elasticnet':
            l1_term = self.l1_coef * np.sign(self.weights)
            l2_term = self.l2_coef * 2 * self.weights
            reg_term = l1_term + l2_term
        else:
            reg_term = np.zeros_like(self.weights)
        
        reg_term[0] = 0  # Do not regularize the bias term
        
        return gradient + reg_term

    def compute_confusion_matrix(self, y_true, y_pred):
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        return np.array([[TN, FP], [FN, TP]])
    
    def calculate_roc_auc(self, y_pred, y):
        probabilities = y_pred
        rounded_probabilities = np.round(probabilities, decimals=10)
        thresholds = np.unique(rounded_probabilities)
        tprs = []
        fprs = []

        for threshold in thresholds:
            y_pred = (rounded_probabilities >= threshold).astype(int)
            TP = np.sum((y_pred == 1) & (y == 1))
            FP = np.sum((y_pred == 1) & (y == 0))
            TN = np.sum((y_pred == 0) & (y == 0))
            FN = np.sum((y_pred == 0) & (y == 1))

            TPR = TP / (TP + FN) if TP + FN != 0 else 0
            FPR = FP / (FP + TN) if FP + TN != 0 else 0
            tprs.append(TPR)
            fprs.append(FPR)

        sorted_indices = np.argsort(fprs)
        sorted_fprs = np.array(fprs)[sorted_indices]
        sorted_tprs = np.array(tprs)[sorted_indices]

        roc_auc = np.trapz(sorted_tprs, sorted_fprs)
        return roc_auc
    
    def get_metric_score(self, y_predicted, y):
        confusion = self.compute_confusion_matrix(y, y_predicted)
        TN, FP, FN, TP = confusion.ravel()

        if self.metric == 'accuracy':
            return (TP + TN) / (TP + FP + FN + TN)
        elif self.metric == 'precision':
            return TP / (TP + FP) if TP + FP != 0 else 0
        elif self.metric == 'recall':
            return TP / (TP + FN) if TP + FN != 0 else 0
        elif self.metric == 'f1':
            precision = TP / (TP + FP) if TP + FP != 0 else 0
            recall = TP / (TP + FN) if TP + FN != 0 else 0
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif self.metric == 'roc_auc':
            return self.calculate_roc_auc(y_predicted, y)
        else:
            raise ValueError("Unknown metric specified")
    
    def get_best_score(self):
        return self.best_score

    def __str__(self):
        return f"MyLogReg class: n_iter={self._iter}, learning_rate={self._learning_rate}, reg={self.reg}, l1_coef={self.l1_coef}, l2_coef={self.l2_coef}"
