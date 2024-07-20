import numpy as np
import pandas as pd
from ml_library.metrics.knn_regression import get_distance_function
from ml_library.utils.weights import get_weight_function

class KNNClf:
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.distance_function = get_distance_function(metric)
        self.weight_function = get_weight_function(weight)

    def fit(self, X, y):
        self.X = X.copy()
        self.y = y.copy()
        self.train_size = X.shape

    def predict(self, X_test):
        predictions = []
        for idx, x_test in X_test.iterrows():
            distances = []
            for i, x_train in self.X.iterrows():
                dist = self.distance_function(x_test.values, x_train.values)
                distances.append((dist, self.y[i], i))  # Include index of the training data
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            class_weights = {0: 0, 1: 0}
            for rank, (dist, label, _) in enumerate(k_nearest):
                weight = self.weight_function(dist, rank)
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
                dist = self.distance_function(x_test.values, x_train.values)
                distances.append((dist, self.y[i], i))  # Include index of the training data
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            class_weights = {0: 0, 1: 0}
            for rank, (dist, label, _) in enumerate(k_nearest):
                weight = self.weight_function(dist, rank)
                class_weights[label] += weight

            total_weight = class_weights[0] + class_weights[1]
            prob_class_1 = class_weights[1] / total_weight if total_weight != 0 else 0.5
            probabilities.append(prob_class_1)
        return np.array(probabilities)

    def __str__(self):
        return f"KNNClf(k={self.k}, metric='{self.metric}', weight='{self.weight}')"
