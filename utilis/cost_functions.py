import numpy as np


def subtract_err(outputs, targets):
    res = outputs - targets
    return res


def mse(outputs, targets):
    res = np.power(outputs - targets, 2)
    return res


def cross_entropy_cost(outputs, targets):
    epsilon = 1e-11
    targets = np.clip(targets, epsilon, 1 - epsilon)
    return -np.mean(outputs * np.log(targets) + (1 - outputs) * np.log(1 - targets))
