import numpy as np
from .Loss import Loss


class Crossentropy(Loss):
    @staticmethod
    def forward(pred, truth):
        return - np.dot(truth.T, np.log(pred))

    @staticmethod
    def backward(pred, truth):
        return pred - truth
        