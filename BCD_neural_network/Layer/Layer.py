class Layer:
    def __init__(self, in_channels, out_channels, activation=None, initializer=None):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, dL_da):
        raise NotImplementedError
