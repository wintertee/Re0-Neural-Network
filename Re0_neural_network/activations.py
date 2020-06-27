import numpy as np


class Activation:
    """
    Base class for activation functions
    """
    @classmethod
    def forward(cls, z):
        """
        parameters:
            z:  input
        return:
            a = activation(z)
        """
        raise NotImplementedError

    @classmethod
    def backward(cls, a):
        """
        parameters:
            a:  output of the activation function
        return:
            derivative of the activation function
        """
        raise NotImplementedError

    @classmethod
    def derivative(cls, a):
        return cls.backward(a)


class relu(Activation):
    @classmethod
    def forward(cls, z):
        return np.maximum(0, z)

    @classmethod
    def backward(cls, a):
        da_dz = a > 0
        da_dz = da_dz.astype(np.int32)
        return da_dz


class softmax(Activation):
    @classmethod
    def forward(cls, z):
        """
        softmax only
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=(1, 2))[:, np.newaxis][:, np.newaxis]

    @classmethod
    def backward(cls, a):
        """
        implemented in Loss/Crossentropy.py, here returns identity matrix.
        use derivative if need
        """
        return np.ones(a.shape)

    @classmethod
    def derivative(cls, a):
        batch_size = a.shape[0]
        num_features = a.shape[1]
        x = np.zeros((batch_size, num_features))
        for i in range(num_features):
            x[:, i] = (a[:, i] * (1 - a[:, i])).reshape(batch_size)
        return x.reshape(*x.shape, 1)
