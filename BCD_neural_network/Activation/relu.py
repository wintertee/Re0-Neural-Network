import numpy as np
from .Activation import Activation


class relu(Activation):
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def backward(a):
        da_dz = a > 0
        da_dz = da_dz.astype(np.int32)
        return da_dz
