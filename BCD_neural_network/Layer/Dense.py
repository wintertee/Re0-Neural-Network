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
            self.w = np.ones(out_channels, in_channels)
        self.b = np.zeros(out_channels)

        self.clear()

    def clear(self):
        self.dL_dw = np.zeros(self.out_channels, self.in_channels)
        self.dL_db = np.zeros(self.out_channels)
        self.dL_dx = np.zeros(self.in_channels)
        self.x = np.zeros(self.out_channels)
        self.z = np.zeros(self.out_channels)
        self.a = np.zeros(self.out_channels)

    def forward(self, x):
        self.x = x
        self.z = np.dot(self.w, x) + self.b
        if self.activation is None:
            self.a = self.z
        else:
            self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, dL_da):
        if self.activation is None:
            da_dz = np.ones(self.out_channels)
        else:
            da_dz = self.activation.derivative(self.a, dL_da)

        da_dw = np.dot(da_dz, self.x)
        self.dL_dw = np.dot(dL_da, da_dw)

        da_dx = np.dot(da_dz, self.w)
        self.dL_dx = np.dot(dL_da, da_dx)

        da_db = da_dz
        self.dL_db = np.dot(dL_da, da_db)
