import numpy as np
from .Activation import Activation


class softmax(Activation):
    @staticmethod
    def forward(z):
        """
        softmax only
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    @staticmethod
    def backward(a, dL_da):
        """
        implemented in crossentropy
        """
        return dL_da
