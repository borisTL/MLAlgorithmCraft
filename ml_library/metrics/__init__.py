# ml_library/metrics/__init__.py
from .regression_metrics import calc_mae, calc_mse, calc_rmse, calc_r2, calc_mape
from .classification_metrics import calc_accuracy, calc_precision, calc_recall, calc_f1, calc_roc_auc
from .knn_regression import euclidean_distance, manhattan_distance, chebyshev_distance, cosine_distance, get_distance_function
