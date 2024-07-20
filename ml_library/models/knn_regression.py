import numpy as np
import pandas as pd
from ml_library.metrics.knn_regression import get_distance_function
from ml_library.utils.weights import get_weight_function

class KNNRegresion:
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
            
            weighted_sum = 0
            total_weight = 0
            for rank, (dist, value, _) in enumerate(k_nearest):
                weight = self.weight_function(dist, rank)
                weighted_sum += weight * value
                total_weight += weight
            
            prediction = weighted_sum / total_weight if total_weight != 0 else np.mean(self.y)
            predictions.append(prediction)
        return np.array(predictions)

    def __str__(self):
        return f"KNNRegresion class: k={self.k}, metric='{self.metric}', weight='{self.weight}'"
