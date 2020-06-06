import numpy as np


class Loss:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(a):
        raise NotImplementedError


class Crossentropy(Loss):
    @staticmethod
    def forward(pred, truth):
        return -np.dot(truth.T, np.log(pred))

    @staticmethod
    def backward(pred, truth):
        return pred - truth
