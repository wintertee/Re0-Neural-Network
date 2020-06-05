import numpy as np
from .Layer import Layer


class Dense(Layer):
    def __init__(self, in_channels, out_channels, activation=None, initializer=None):

        # initialize propoties
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # initialize parameters
        if initializer:
            self.w = initializer((out_channels, in_channels))
        else:
            self.w = np.ones((out_channels, in_channels))
        self.b = np.zeros((out_channels, 1))

        self.clear()

    def clear(self):
        self.dL_dw = np.zeros((self.out_channels, self.in_channels))
        self.dL_db = np.zeros((self.out_channels, 1))
        self.dL_dx = np.zeros((self.in_channels, 1))
        self.x = np.zeros((self.in_channels, 1))
        self.z = np.zeros((self.out_channels, 1))
        self.a = np.zeros((self.out_channels, 1))

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.w, self.x) + self.b
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
        self.dL_dw = np.dot(dL_dz, self.x.T)
        self.dL_dx = np.dot(self.w.T, dL_dz)
        self.dL_db = dL_dz

        return self.dL_dx  # x is the `a` in the last layer

    def update(self, lr):
        self.w -= lr * self.dL_dw
        self.b -= lr * self.dL_db
