import numpy as np
import pandas as pd 
import random

class LineRegression:                                                                                                                                                                                                                                                        
                                                                                                                                              
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=None):
#___________________________________________________________________________________________________________________________________________________________________________________________

        self.n_iter = n_iter
        """
           The n_iter attribute in the LineReg class represents the number of iterations or epochs during the training phase of the regression model.
           In machine learning, particularly in iterative optimization algorithms like gradient descent, an iteration refers to one complete cycle where the model
           sees the entire training dataset. Each iteration involves updating the model parameters (weights) based on the calculated gradients to minimize the loss function.
        """                      
        self.learning_rate = learning_rate 
        """
           Learning rate is a hyperparameter that determines the size of the step taken to update the model weights during training using gradient descent
           or other optimization algorithms. The learning rate controls how much we adjust the model weights in response to the estimated error at each
           training step. If the learning rate is too large, the model may overshoot the minimum of the loss function; if it is too small, the training process will be very slow.
           Selecting the optimal learning rate is a key aspect of training neural networks and other machine learning models.
        """         

        self.weights = None
        """
           The weights attribute in the LineReg class represents the model coefficients or parameters that are learned during the training phase.
           The weights are adjusted during training to minimize the loss function and make accurate predictions on the training data.
        """                         
        self.metric = metric if metric else 'mse'   
        """
           The metric attribute in the LineReg class specifies the evaluation metric used to assess the performance of the regression model.
           Common regression metrics include mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE),
           and coefficient of determination (R^2). The metric attribute allows the user to choose the desired evaluation metric for the model.
        """
        self.reg = reg 
        """
           The reg attribute in the LineReg class specifies the type of regularization to apply during training to prevent overfitting.
           Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function that discourages large weights.
           Common types of regularization include L1 regularization (Lasso), L2 regularization (Ridge), and elastic net regularization.
           The reg attribute allows the user to choose the type of regularization or set it to None for no regularization.
        """                           
        self.l1_coef = l1_coef 
        """
           The l1_coef attribute in the LineReg class represents the coefficient for L1 regularization (Lasso) when the reg attribute is set to 'l1'.
           L1 regularization adds a penalty term to the loss function that is proportional to the absolute values of the model weights.
           The l1_coef attribute allows the user to set the strength of the L1 regularization penalty.
        """
        self.l2_coef = l2_coef
        """
           The l2_coef attribute in the LineReg class represents the coefficient for L2 regularization (Ridge) when the reg attribute is set to 'l2'.
           L2 regularization adds a penalty term to the loss function that is proportional to the square of the model weights.
           The l2_coef attribute allows the user to set the strength of the L2 regularization penalty.
        """                    
        self.sgd_sample = sgd_sample
        """
           The sgd_sample attribute in the LineReg class specifies the size of the sample used for stochastic gradient descent (SGD) training.
           Stochastic gradient descent is an optimization algorithm that updates the model weights based on a subset of the training data (mini-batch) rather than the full dataset.
           The sgd_sample attribute allows the user to specify the sample size as an integer or a float representing the percentage of the training data to use in each iteration.
        """            
        self.random_state = random_state
        """
           The random_state attribute in the LineReg class is used to set the random seed for reproducibility.
           Random seeds are used to initialize the random number generator in machine learning algorithms to ensure that the results are reproducible across runs.
           Setting the random_state attribute to a specific value will produce the same results each time the model is trained.
        """         
        self.best_score_ = np.inf
        """"
              The best_score_ attribute in the LineReg class stores the best evaluation metric score achieved during training.
              The best_score_ attribute is used to track the best model performance and can be accessed after training to evaluate the model's quality.
        """
        self.components = None
        """
           The components attribute in the LineReg class represents the principal components of the data when using PCA for feature transformation.
           Principal component analysis (PCA) is a dimensionality reduction technique that identifies the principal components or directions of maximum variance in the data.
           The components attribute stores the principal components of the data after fitting the PCA model.
        """
#___________________________________________________________________________________________________________________________________________________________________________________________
    def fit(self, X, y, verbose=False):
        """"
              The fit method in the LineReg class is used to train the linear regression model on the input features X and target values y.
              The fit method updates the model weights based on the training data to minimize the loss function and make accurate predictions.
              The verbose parameter controls whether to print the evaluation metric during training for monitoring the model's progress.
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)
        """
            Convert the input features X and target values y to pandas DataFrame and Series, respectively.
        """
        X.insert(0, "bias", 1)
        """
            Add a bias column to the input features X to account for the intercept term in the linear regression model.
        """
        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        """
            Initialize the model weights to ones if they are not already set.
        """
        if self.components == 'pca': 
            X = self.transform_data(X) 
        """
            Transform the input features X using the principal components if the components attribute is set to 'pca'.
        """
        random.seed(self.random_state)
        """
            This line of code initializes the random number generator with a fixed initial value (self.random_state), ensuring reproducibility of random operations.
        """
        for i in range(self.n_iter):
            """
    Stochastic Gradient Descent (SGD) Sampling:
    If the sgd_sample parameter is specified, it determines the sample size for SGD.
    If sgd_sample is an integer, the sample size is set to that integer.
    Otherwise (if sgd_sample is a fraction), the sample size is calculated as a percentage of the total number of rows in the data X.
    Random row indices (sample_rows_idx) are generated from the range of 0 to the number of rows in the data.
    Samples X_sample and y_sample are created based on the generated indices.
    Prediction on Sample and Gradient Computation:

    Prediction (y_predicted_sample) is made on the current sample (X_sample) using the predict method.
    The gradient (gradient) of the loss function on the current sample is computed using the compute_gradient method.
    Updating Model Weights:

    The model weights are updated on each iteration using the gradient descent method.
    New weights are computed by subtracting the adjusted gradient, multiplied by the learning rate (learning_rate), from the current model weights (self.weights).
    Training Progress Output (Optional):

    If the verbose mode is enabled and the current iteration divides evenly by the verbose value, the following actions are performed:
    Predictions are made on the full data (X) using the predict method.
    The quality metric (metric) is computed on the full data using the compute_metric method.
    If the current quality metric is lower than the best saved metric (self.best_score_), the best metric is updated.
    Thus, the training loop runs for the specified number of iterations (n_iter), updating the model on each iteration according to the chosen optimization method and data sampling for SGD. If needed, the training progress is printed, and the best model quality metric is updated.
        """
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
        """
            The predict method in the LineReg class is used to make predictions on the input features X using the trained linear regression model.
            The predict method computes the dot product of the input features and the model weights to generate predictions.
        """
        X = pd.DataFrame(X)                                
        if "bias" not in X.columns:                                       
            X.insert(0, "bias", 1)                          
        predictions = np.dot(X, self.weights)              
        return predictions                                   

#___________________________________________________________________________________________________________________________________________________________________________________________
    def compute_metric(self, y_true, y_predicted):
        """
                The compute_metric method in the LineReg class is used to calculate the evaluation metric between the true target values (y_true) and the predicted values (y_predicted).
                The compute_metric method supports various regression metrics, including mean squared error (MSE), mean absolute error (MAE), root mean squared error (RMSE),
                mean absolute percentage error (MAPE), and coefficient of determination (R^2). The method returns the computed metric value.
        """                                 
        if self.metric == 'mse':
            """
            The mean squared error (MSE) is a common regression metric that measures the average squared difference between the true target values and the predicted values.
            """
            metric = np.mean((y_true - y_predicted) ** 2)

        elif self.metric == 'mae':
            """
            The mean absolute error (MAE) is a regression metric that measures the average absolute difference between the true target values and the predicted values.
            """
            metric = np.mean(np.abs(y_true - y_predicted))                        
        elif self.metric == 'rmse':
            """
            The root mean squared error (RMSE) is a regression metric that measures the square root of the average squared difference between the true target values and the predicted values.
            """
            metric = np.sqrt(np.mean((y_true - y_predicted) ** 2))                 
        elif self.metric == 'mape':
            """
            The mean absolute percentage error (MAPE) is a regression metric that measures the average percentage difference between the true target values and the predicted values.
            """
            metric = np.mean(np.abs((y_true - y_predicted) / y_true)) * 100         
        elif self.metric == 'r2':
            """
            The coefficient of determination (R^2) is a regression metric that measures the proportion of the variance in the target values that is predictable from the input features.
            """
            ss_res = np.sum((y_true - y_predicted) ** 2)                           
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)                       
            metric = 1 - (ss_res / ss_tot)                                         
        else:
            raise ValueError(f'Unsupported metric: {self.metric}')               
        return metric
#___________________________________________________________________________________________________________________________________________________________________________________________
    def compute_gradient(self, X, y_true, y_predicted):
        """
            The compute_gradient method in the LineReg class is used to calculate the gradient of the loss function with respect to the model weights.
            The gradient represents the direction and magnitude of the steepest ascent of the loss function and is used to update the model weights during training.
        """                                                            
        gradient = -2 * np.dot(X.T, (y_true - y_predicted)) / X.shape[0]                            
        if self.reg == 'l1':
            """
            L1 regularization (Lasso) adds a penalty term to the loss function that is proportional to the absolute values of the model weights.
            The L1 regularization term encourages sparsity in the model by driving some weights to zero.
            """    
            gradient += self.l1_coef * np.sign(self.weights)                                        
        elif self.reg == 'l2':
            """
            L2 regularization (Ridge) adds a penalty term to the loss function that is proportional to the square of the model weights.
            The L2 regularization term prevents overfitting by discouraging large weights.
            """
            gradient += 2 * self.l2_coef * self.weights                                             
        elif self.reg == 'elasticnet':
            """
            Elastic net regularization combines L1 and L2 regularization by adding both penalty terms to the loss function.
            The elastic net regularization term is a linear combination of the L1 and L2 penalty terms with separate coefficients.
            """
            gradient += self.l1_coef * np.sign(self.weights) + 2 * self.l2_coef * self.weights      
        elif self.reg is not None:

            raise ValueError(f'Unsupported regularization: {self.reg}')                             
        return gradient
#___________________________________________________________________________________________________________________________________________________________________________________________

    def get_coef(self): 
        """
            The get_coef method in the LineReg class is used to retrieve the model coefficients (weights) after training.
            The method returns the learned model coefficients as a NumPy array.
        """                                                    
        return np.array(self.weights)
                                           
#___________________________________________________________________________________________________________________________________________________________________________________________
    def get_best_score(self):
        """
            The get_best_score method in the LineReg class is used to retrieve the best evaluation metric score achieved during training.
            The method returns the best model performance score as a floating-point number.
        """
    
        if self.best_score_ != np.inf:
            return self.best_score_
        else:
            raise ValueError("The model hasn't been fitted yet.")
#___________________________________________________________________________________________________________________________________________________________________________________________
        

    def forward_selection(self, X, y, verbose=False):
        """
            The forward_selection method in the LineReg class is used to perform forward feature selection on the input features X and target values y.
            Forward feature selection is a greedy search algorithm that iteratively adds features to the model based on their individual performance.
            The method returns a list of selected features that improve the model's evaluation metric.
        """
       
       
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
        """
            The backward_elimination method in the LineReg class is used to perform backward feature elimination on the input features X and target values y.  
            Backward feature elimination is a greedy search algorithm that iteratively removes features from the model based on their individual performance.
            The method returns a list of selected features that improve the model's evaluation metric.
        """
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
        """
            The stepwise_selection method in the LineReg class is used to perform stepwise feature selection on the input features X and target values y.
            Stepwise feature selection is a combination of forward and backward selection methods that iteratively adds and removes features from the model.
            The method returns a list of selected features that improve the model's evaluation metric.
        """
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
        """
            The compute_principal_components method in the LineReg class is used to compute the principal components of the input features X using principal component analysis (PCA).
            Principal component analysis is a dimensionality reduction technique that identifies the principal components or directions of maximum variance in the data.
            The method computes the principal components of the data and stores them in the components attribute for feature transformation.
        """
        cov_matrix = np.cov(X.T)
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
        idx = np.argsort(eigen_values)[::-1]
        self.components = eigen_vectors[:, idx]
#___________________________________________________________________________________________________________________________________________________________________________________________
    def transform_data(self, X):
        """
            The transform_data method in the LineReg class is used to transform the input features X using the principal components computed by PCA.
            The method returns the transformed features after projecting them onto the principal components.

        """
        if self.components is None:
            raise ValueError("The principal components of the data have not been computed. Please first call the compute_principal_components method.")
        return np.dot(X, self.components)
#___________________________________________________________________________________________________________________________________________________________________________________________

    def __str__(self):
        """
            The __str__ method in the LineReg class returns a string representation of the model instance with key attributes and parameters.
            The method provides a concise summary of the LineReg object for display and debugging purposes.
        """
        return f'MyLineReg instance: n_iter={self.n_iter},learning_rate={self.learning_rate},metric={self.metric},reg={self.reg},l1_coef={self.l1_coef},l2_coef={self.l2_coef}'
#___________________________________________________________________________________________________________________________________________________________________________________________
    def __repr__(self):
        """
            The __repr__ method in the LineReg class returns a string representation of the model instance with key attributes and parameters.
            The method provides a detailed representation of the LineReg object for debugging and logging purposes.
        """
        return f'MyLineReg instance: n_iter={self.n_iter},learning_rate={self.learning_rate},metric={self.metric},reg={self.reg},l1_coef={self.l1_coef},l2_coef={self.l2_coef}'