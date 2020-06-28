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
        da_dz_M = np.zeros((a.shape[0], a.shape[1], a.shape[1]))
        for i in range(a.shape[0]):
            da_dz_M[i] = np.diag(da_dz[i].squeeze())
        return da_dz_M

    @classmethod
    def d2(cls, a):
        return cls.backward(a)


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
        return np.broadcast_to(np.eye(a.shape[1]), (a.shape[0], a.shape[1], a.shape[1]))

    @classmethod
    def derivative(cls, a):
        batch_size = a.shape[0]
        num_features = a.shape[1]
        d = np.zeros((batch_size, num_features, num_features))
        for i in range(num_features):
            for j in range(i, num_features):
                d[:, i, j] = a[:, i, 0] * (1 - a[:, i, 0]) if i == j else - a[:, i, 0] * a[:, i, 0]
        for i in range(batch_size):
            d[i] += d[i].T - np.diag(d[i].diagonal())
        return d

    @classmethod
    def d2(cls, a):
        return 1 - 2 * a