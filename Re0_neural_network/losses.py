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
        return - np.mean(np.einsum('ijk,ijk->i', truth, np.log(pred + 1e-6)))

    @staticmethod
    def backward(pred, truth):
        return pred - truth

    @staticmethod
    def derivative(pred, truth):
        return -truth / (1e-2 + pred)
