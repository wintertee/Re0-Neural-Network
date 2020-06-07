import numpy as np


class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_channels, out_channels, activation=None, initializer=None):

        # initialize properties
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # initialize parameters
        self.P = {}  # parameters
        self.G = {}  # gradients
        if initializer:
            self.P['w'] = initializer((out_channels, in_channels))
        else:
            self.P['w'] = np.ones((out_channels, in_channels))
        self.P['b'] = np.zeros((out_channels, 1))

    def forward(self, x):
        # 先把 x 分解成 batch_size 个矩阵 (in ,1)
        # 分别计算a
        # 最后通过 np.stack 把 a 堆叠成 (batch_size, out, 1)
        self.x = x
        self.batch_size = x.shape[0]
        a = []
        for i in range(self.batch_size):
            x_single = x[i]
            self.z = np.dot(self.P['w'], x_single) + self.P['b']
            if self.activation is None:
                a.append(self.z)
            else:
                a.append(self.activation.forward(self.z))
        self.a = np.stack(a)
        return self.a

    def backward(self, dL_da):
        if self.activation is None:
            da_dz = np.ones((self.batch_size, self.out_channels, 1))
        else:
            da_dz = self.activation.backward(self.a)

        dL_dz = []
        dL_dx = []
        G_w = []

        for i in range(self.batch_size):
            dL_dz.append(dL_da[i] * da_dz[i])  # NOTE * not np.dot here!
            G_w.append(np.dot(dL_dz[i], self.x[i].T))  # dL_dw
            dL_dx.append(np.dot(self.P['w'].T, dL_dz[i]))

        G_b = dL_dz
        self.G['w'] = np.mean(G_w, axis=0)
        self.G['b'] = np.mean(G_b, axis=0)

        return np.stack(dL_dx)  # NOTE x is the `a` in the last layer


class Flatten(Layer):
    def __init__(self):
        self.P = {}
        self.G = {}
        self.P['w'] = np.array([])
        self.P['b'] = np.array([])
        self.G['w'] = np.array([])
        self.G['b'] = np.array([])

    def forward(self, x):
        self.original_shape = x.shape
        size = x.size // x.shape[0]
        return x.reshape(x.shape[0], size, 1)

    def backward(self, dL_da):
        return dL_da.reshape(self.original_shape)
