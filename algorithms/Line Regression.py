import numpy as np
import pandas as pd 
import random

class LineRegression:                                                                                                                                                                                                                                                        
                                                                                                                                              
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=None):
#___________________________________________________________________________________________________________________________________________________________________________________________

        self.n_iter = n_iter
                     
        self.learning_rate = learning_rate 
        

        self.weights = None
                           
        self.metric = metric if metric else 'mse'   
        
        self.reg = reg 
                                  
        self.l1_coef = l1_coef 
        
        self.l2_coef = l2_coef
                     
        self.sgd_sample = sgd_sample
                  
        self.random_state = random_state
          
        self.best_score_ = np.inf
        
        self.components = None
        
#___________________________________________________________________________________________________________________________________________________________________________________________
    def fit(self, X, y, verbose=False):
        
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        X.insert(0, "bias", 1)
        
        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        
        if self.components == 'pca': 
            X = self.transform_data(X) 
        
        random.seed(self.random_state)
        
        for i in range(self.n_iter):
            
            if self.sgd_sample is not None:
                if isinstance(self.sgd_sample, int):
                    sample_size = self.sgd_sample
                else:
                    sample_size = int(X.shape[0] * self.sgd_sample)
                sample_rows_idx = random.sample(range(X.shape[0]), sample_size)  
                X_sample = X.iloc[sample_rows_idx]
                y_sample = y.iloc[sample_rows_idx]
            else:
                X_sample = X
                y_sample = y
            
            y_predicted_sample = self.predict(X_sample)
            gradient = self.compute_gradient(X_sample, y_sample, y_predicted_sample)
            self.weights -= self.learning_rate * gradient
            
            if verbose and (i+1) % verbose == 0:
                y_predicted = self.predict(X)
                metric = self.compute_metric(y, y_predicted)
                print(f'{i+1} | {self.metric}: {metric:.2f}')
                if metric < self.best_score_:
                    self.best_score_ = metric
    

#___________________________________________________________________________________________________________________________________________________________________________________________

    def predict(self, X):
        
        X = pd.DataFrame(X)                                
        if "bias" not in X.columns:                                       
            X.insert(0, "bias", 1)                          
        predictions = np.dot(X, self.weights)              
        return predictions                                   

#___________________________________________________________________________________________________________________________________________________________________________________________
    def compute_metric(self, y_true, y_predicted):
                                     
        if self.metric == 'mse':
            
            metric = np.mean((y_true - y_predicted) ** 2)

        elif self.metric == 'mae':
            
            metric = np.mean(np.abs(y_true - y_predicted))                        
        elif self.metric == 'rmse':
            
            metric = np.sqrt(np.mean((y_true - y_predicted) ** 2))                 
        elif self.metric == 'mape':
            
            metric = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100         
        elif self.metric == 'r2':
            
            ss_res = np.sum((y_true - y_predicted) ** 2)                           
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)                       
            metric = 1 - (ss_res / ss_tot)                                         
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')               
        return metric
#___________________________________________________________________________________________________________________________________________________________________________________________
    def compute_gradient(self, X, y_true, y_predicted):
                                                                
        gradient = -2 * np.dot(X.T, (y_true - y_predicted)) / X.shape[0]                            
        if self.reg == 'l1':
               
            gradient += self.l1_coef * np.sign(self.weights)                                        
        elif self.reg == 'l2':
            
            gradient += 2 * self.l2_coef * self.weights                                             
        elif self.reg == 'elasticnet':
           
            gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights      
        elif self.reg is not None:

            raise ValueError(f'Unsupported regularization: {self.reg}')                             
        return gradient
#___________________________________________________________________________________________________________________________________________________________________________________________

    def get_coef(self): 
                                                          
        return np.array(self.weights)
                                           
#___________________________________________________________________________________________________________________________________________________________________________________________
    def get_best_score(self):
        
    
        if self.best_score_ != np.inf:
            return self.best_score_
        else:
            raise ValueError("The model hasn't been fitted yet.")
#___________________________________________________________________________________________________________________________________________________________________________________________
        

    def forward_selection(self, X, y, verbose=False):
       
       
       
        features = X.columns.tolist()
        selected_features = []
        best_metric = np.inf

        while True:
            metric_improved = False
            best_feature = None

            for feature in features:
                current_features = selected_features + [feature]
                X_subset = X[current_features]
                self.fit(X_subset, y, verbose=False)
                y_pred = self.predict(X_subset)
                metric = self.compute_metric(y, y_pred)

                if metric < best_metric:
                    best_metric = metric
                    best_feature = feature
                    metric_improved = True

            if metric_improved:
                selected_features.append(best_feature)
                features.remove(best_feature)
                if verbose:
                    print(f"Added feature: {best_feature}, {self.metric}: {best_metric:.2f}")
            else:
                break

        return selected_features
    
#___________________________________________________________________________________________________________________________________________________________________________________________
    def backward_elimination(self, X, y, verbose=False):
        
        features = X.columns.tolist()
        best_metric = np.inf
        selected_features = features[:]

        while True:
            metric_improved = False
            worst_feature = None

            for feature in selected_features:
                current_features = selected_features.copy()
                current_features.remove(feature)
                X_subset = X[current_features]
                self.fit(X_subset, y, verbose=False)
                y_pred = self.predict(X_subset)
                metric = self.compute_metric(y, y_pred)

                if metric < best_metric:
                    best_metric = metric
                    worst_feature = feature
                    metric_improved = True

            if metric_improved:
                selected_features.remove(worst_feature)
                if verbose:
                    print(f"Removed feature: {worst_feature}, {self.metric}: {best_metric:.2f}")
            else:
                break

        return selected_features
#_____________________________________________________________________________________________________________________________________________________________________________________________
    def stepwise_selection(self, X, y, verbose=False):
        
        features = X.columns.tolist()
        selected_features = []
        best_metric = np.inf

        while True:
            metric_improved = False
            best_feature = None

            for feature in features:
                current_features = selected_features + [feature]
                X_subset = X[current_features]
                self.fit(X_subset, y, verbose=False)
                y_pred = self.predict(X_subset)
                metric = self.compute_metric(y, y_pred)

                if metric < best_metric:
                    best_metric = metric
                    best_feature = feature
                    metric_improved = True

            if metric_improved:
                selected_features.append(best_feature)
                features.remove(best_feature)
                if verbose:
                    print(f"Added feature: {best_feature}, {self.metric}: {best_metric:.2f}")
            else:
                break

            # Обратный шаг
            for feature in selected_features:
                current_features = selected_features.copy()
                current_features.remove(feature)
                X_subset = X[current_features]
                self.fit(X_subset, y, verbose=False)
                y_pred = self.predict(X_subset)
                metric = self.compute_metric(y, y_pred)

                if metric < best_metric:
                    best_metric = metric
                    best_feature = feature
                    metric_improved = True

            if metric_improved:
                selected_features.remove(best_feature)
                if verbose:
                    print(f"Removed feature: {best_feature}, {self.metric}: {best_metric:.2f}")
            else:
                break

        return selected_features
#___________________________________________________________________________________________________________________________________________________________________________________________    
    def compute_principal_components(self, X):
        
        cov_matrix = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        idx = np.argsort(eigen_values)[::-1]
        self.components = eigen_vectors[:, idx]
#___________________________________________________________________________________________________________________________________________________________________________________________
    def transform_data(self, X):
        
        if self.components is None:
            raise ValueError("The principal components of the data have not been computed. Please first call the compute_principal_components method.")
        return np.dot(X, self.components)
#___________________________________________________________________________________________________________________________________________________________________________________________

    def __str__(self):
        
        return f'MyLineReg instance: n_iter={self.n_iter},learning_rate={self.learning_rate},metric={self.metric},reg={self.reg},l1_coef={self.l1_coef},l2_coef={self.l2_coef}'
#___________________________________________________________________________________________________________________________________________________________________________________________
    def __repr__(self):
        
        return f'MyLineReg instance: n_iter={self.n_iter},learning_rate={self.learning_rate},metric={self.metric},reg={self.reg},l1_coef={self.l1_coef},l2_coef={self.l2_coef}'
