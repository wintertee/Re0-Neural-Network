import numpy as np


class Activation:
    """
    Base class for activation functions
    """
    @staticmethod
    def forward(z):
        """
        parameters:
            z:  input
        return:
            a = activation(z)
        """
        raise NotImplementedError

    @staticmethod
    def backward(a):
        """
        parameters:
            a:  output of the activation function
        return:
            derivative of the activation function
        """
        raise NotImplementedError


class relu(Activation):
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def backward(a):
        da_dz = a > 0
        da_dz = da_dz.astype(np.int32)
        return da_dz


class softmax(Activation):
    @staticmethod
    def forward(z):
        """
        softmax only
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    @staticmethod
    def backward(a):
        """
        implemented in Loss/Crossentropy.py, here returns identity matrix.
        """
        return np.ones(a.shape)
