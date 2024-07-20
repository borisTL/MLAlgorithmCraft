import numpy as np
import pandas as pd

class PCA:
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def __str__(self):
        return f"MyPCA class: n_components={self.n_components}"

    def fit(self, X):
        
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

      
        cov_matrix = np.cov(X_centered.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
       
        X_centered = X - self.mean

        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
