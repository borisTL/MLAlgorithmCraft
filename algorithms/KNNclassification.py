import pandas as pd
import numpy as np

class KNNClf: # for binary classification
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.train_size = X.shape

    def distance_calculation(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'chebyshev':
            return np.max(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            dot_product = np.dot(x1, x2)
            norm_x1 = np.linalg.norm(x1)
            norm_x2 = np.linalg.norm(x2)
            return 1 - dot_product / (norm_x1 * norm_x2)
        else:
            raise ValueError("Unsupported metric")

    def calculate_weight(self, dist, rank):
        if self.weight == 'uniform':
            return 1
        elif self.weight == 'rank':
            return 1 / (rank + 1)
        elif self.weight == 'distance':
            return 1 / (dist + 1e-5)  # Small value added to avoid division by zero
        else:
            raise ValueError("Unsupported weight type")

    def predict(self, X_test):
        predictions = []
        for idx, x_test in X_test.iterrows():
            distances = []
            for i, x_train in self.X.iterrows():
                dist = self.distance_calculation(x_test.values, x_train.values)
                distances.append((dist, self.y[i], i))  # Include index of the training data
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            class_weights = {0: 0, 1: 0}
            for rank, (dist, label, _) in enumerate(k_nearest):
                weight = self.calculate_weight(dist, rank)
                class_weights[label] += weight
            
            if class_weights[0] == class_weights[1]:
                most_common = 1
            else:
                most_common = max(class_weights, key=class_weights.get)
            
            predictions.append(most_common)
        return np.array(predictions)

    def predict_proba(self, X_test):
        probabilities = []
        for idx, x_test in X_test.iterrows():
            distances = []
            for i, x_train in self.X.iterrows():
                dist = self.distance_calculation(x_test.values, x_train.values)
                distances.append((dist, self.y[i], i))  # Include index of the training data
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            class_weights = {0: 0, 1: 0}
            for rank, (dist, label, _) in enumerate(k_nearest):
                weight = self.calculate_weight(dist, rank)
                class_weights[label] += weight

            total_weight = class_weights[0] + class_weights[1]
            prob_class_1 = class_weights[1] / total_weight if total_weight != 0 else 0.5
            probabilities.append(prob_class_1)
        return np.array(probabilities)

    def __str__(self):
        return f"MyKNNClf(k={self.k}, metric='{self.metric}', weight='{self.weight}')"