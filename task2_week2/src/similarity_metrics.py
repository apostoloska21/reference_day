import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2))


# def weighted_euclidean_distance(a, b, weights):
#     a = np.asarray(a, dtype=float)
#     b = np.asarray(b, dtype=float)
#     weights = np.asarray(weights, dtype=float)
#     return np.sqrt(np.sum(weights * (a - b) ** 2))


def cosine_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(a == 0) or np.all(b == 0):
        return 1.0
    try:
        return cosine(a, b)
    except:
        return 1.0


def correlation_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return 1.0
    try:
        corr, _ = pearsonr(a, b)
        if np.isnan(corr):
            return 1.0
        return 1 - abs(corr)
    except:
        return 1.0