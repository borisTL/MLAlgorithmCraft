# ml_library/models/regularized_discriminant_analysis.py
import numpy as np

class RegularizedDiscriminantAnalysis:
    def __init__(self, alpha=0.5, gamma=0.5):
        self.alpha = alpha
        self.gamma = gamma
        self.means = None
        self.covariances = None
        self.priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]

        self.means = {}
        self.covariances = {}
        self.priors = {}

        for cls in self.classes:
            X_c = X[y == cls]
            self.means[cls] = np.mean(X_c, axis=0)
            self.covariances[cls] = np.cov(X_c, rowvar=False)
            self.priors[cls] = X_c.shape[0] / X.shape[0]

        # Compute pooled covariance
        pooled_cov = sum(self.priors[cls] * self.covariances[cls] for cls in self.classes)

        # Regularize covariance matrices
        for cls in self.classes:
            self.covariances[cls] = (
                self.alpha * self.covariances[cls]
                + (1 - self.alpha) * pooled_cov
                + self.gamma * np.eye(n_features)
            )

    def predict(self, X):
        y_pred = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                likelihood = self._compute_likelihood(x, cls)
                posterior = prior + likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

    def _compute_likelihood(self, x, cls):
        mean = self.means[cls]
        cov = self.covariances[cls]
        cov_inv = np.linalg.inv(cov)
        diff = x - mean
        return -0.5 * (np.log(np.linalg.det(cov)) + diff.T @ cov_inv @ diff)

    def predict_proba(self, X):
        proba = []
        for x in X:
            posteriors = []
            for cls in self.classes:
                prior = np.log(self.priors[cls])
                likelihood = self._compute_likelihood(x, cls)
                posterior = prior + likelihood
                posteriors.append(np.exp(posterior))
            total = np.sum(posteriors)
            proba.append([p / total for p in posteriors])
        return np.array(proba)
