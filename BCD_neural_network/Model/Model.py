class Model:
    def __init__(self):
        self.layers = []

    def append(self, layer):
        self.layers.append(layer)

    def setLoss(self, loss):
        self.loss = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.pred = x

    def backward(self, dL_da):
        for layer in reversed(self.layers):
            layer.backward()

    def train(self, x, y, lr):
        self.forward(x)
        loss = self.loss(self.pred, y)
        dL_da = self.backward(self.pred, y)
        # unfinished
