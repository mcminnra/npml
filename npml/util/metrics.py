import numpy as np


def log_loss(features, target, weights):
    scores = np.dot(features, weights)
    ll = -np.sum(target*scores - np.log(1 + np.exp(scores)))

    return ll


def mean_squared_error(features, target, weights):
    """Computes Mean Squared Error

    Args:
        features: numpy array of predictor values
        target: numpy array of response values
        weights: weights to estimate target

    Returns:
        Mean Squared Error

    """
    N = len(features)
    loss = target - np.dot(features, weights)
    mse = (1 / (2 * N)) * np.sum(loss ** 2)

    return mse

