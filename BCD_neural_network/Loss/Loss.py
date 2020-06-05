class Loss:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def derivative(a):
        raise NotImplementedError