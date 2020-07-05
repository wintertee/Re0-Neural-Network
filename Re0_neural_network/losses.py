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
        return - np.mean(np.einsum('ijk,ijk->i', truth, np.log(pred + 1e-6)))  # Prevent division by 0

    @staticmethod
    def backward(pred, truth):
        """
        this returns the derivative of "cross entropy round softmax"
        """
        return pred - truth

    @staticmethod
    def derivative(pred, truth):
        return -truth / (1e-2 + pred)  # Prevent division by 0
