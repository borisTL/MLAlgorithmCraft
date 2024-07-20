def uniform_weight(dist, rank):
    return 1

def rank_weight(dist, rank):
    return 1 / (rank + 1)

def distance_weight(dist, rank):
    return 1 / (dist + 1e-5)  # Small value added to avoid division by zero

def get_weight_function(weight):
    if weight == 'uniform':
        return uniform_weight
    elif weight == 'rank':
        return rank_weight
    elif weight == 'distance':
        return distance_weight
    else:
        raise ValueError("Unsupported weight type")
