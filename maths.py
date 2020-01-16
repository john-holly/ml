import numpy as np


def heaviside(x, threshold=.5):
    return 1 if x >= threshold else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(output):
    return output * (1.0 - output)
