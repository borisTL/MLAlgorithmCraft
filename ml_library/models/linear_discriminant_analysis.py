# ml_library/models/linear_discriminant_analysis.py
import numpy as np

class LinearDiscriminantAnalysis:
    def __init__(self):
        self.means = None
        self.priors = None
        self.w = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.means = {}
        self.priors = {}
        overall_mean = np.mean(X, axis=0)

        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

            S_W += np.dot((X_c - self.means[cls]).T, (X_c - self.means[cls]))

            mean_diff = (self.means[cls] - overall_mean).reshape(n_features, 1)
            S_B += X_c.shape[0] * np.dot(mean_diff, mean_diff.T)

        self.w = np.linalg.inv(S_W).dot(self.means[self.classes[1]] - self.means[self.classes[0]])

    def transform(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        projections = self.transform(X)
        threshold = np.dot(self.means[self.classes[0]] + self.means[self.classes[1]], self.w) / 2
        return np.where(projections > threshold, self.classes[1], self.classes[0])
