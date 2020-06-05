import numpy as np
from .Activation import Activation


class relu(Activation):
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def backword(a, dL_da):
        derivative = a > 0
        derivative = derivative.astype(np.int32)
        return np.dot(dL_da, derivative.reshape(a.shape))
