import numpy as np


class Layer:
    def __init__(self, in_channels, out_channels, activation=None, initializer=None):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dL_da):
        raise NotImplementedError

    def update(self, lr):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_channels, out_channels, activation=None, initializer=None):

        # initialize propoties
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # initialize parameters
        self.P = {}
        self.G = {}
        if initializer:
            self.P['w'] = initializer((out_channels, in_channels))
        else:
            self.P['w'] = np.ones((out_channels, in_channels))
        self.P['b'] = np.zeros((out_channels, 1))

        # self.clear()

    # def clear(self):
    #     self.dL_dw = np.zeros((self.out_channels, self.in_channels))
    #     self.dL_db = np.zeros((self.out_channels, 1))
    #     self.dL_dx = np.zeros((self.in_channels, 1))
    #     self.x = np.zeros((self.in_channels, 1))
    #     self.z = np.zeros((self.out_channels, 1))
    #     self.a = np.zeros((self.out_channels, 1))

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.P['w'], self.x) + self.P['b']
        if self.activation is None:
            self.a = self.z
        else:
            self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, dL_da):
        if self.activation is None:
            da_dz = np.ones(self.out_channels)
        else:
            da_dz = self.activation.backward(self.a)

        dL_dz = dL_da * da_dz  # NOTE * not np.dot here!
        self.G['w'] = np.dot(dL_dz, self.x.T)  # dL_dw
        self.dL_dx = np.dot(self.P['w'].T, dL_dz)
        self.G['b'] = dL_dz  # dL_db

        return self.dL_dx  # NOTE x is the `a` in the last layer
