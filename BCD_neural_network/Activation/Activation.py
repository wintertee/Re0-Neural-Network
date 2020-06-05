class Activation:
    @staticmethod
    def forward(z):
        raise NotImplementedError

    @staticmethod
    def backward(a, dL_da):
        raise NotImplementedError
