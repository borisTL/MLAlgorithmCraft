import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))

def cosine_distance(x1, x2):
    dot_product = np.dot(x1, x2)
    norm_x1 = np.linalg.norm(x1)
    norm_x2 = np.linalg.norm(x2)
    return 1 - dot_product / (norm_x1 * norm_x2)

def get_distance_function(metric):
    if metric == 'euclidean':
        return euclidean_distance
    elif metric == 'manhattan':
        return manhattan_distance
    elif metric == 'chebyshev':
        return chebyshev_distance
    elif metric == 'cosine':
        return cosine_distance
    else:
        raise ValueError("Unsupported metric")
