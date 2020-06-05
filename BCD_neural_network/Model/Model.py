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
            dL_da = layer.backward(dL_da)

    def train(self, x, y, lr):

        # forward
        self.forward(x)
        loss = self.loss.forward(self.pred, y)

        # backward
        dL_da = self.loss.backward(self.pred, y)
        self.backward(dL_da)

        # update parameters
        for layer in self.layers:
            layer.update(lr)
            layer.clear()

        return loss

    def val(self, x, y):
        self.forward(x)
        loss = self.loss.forward(self.pred, y)
        return loss

    def fit(self, train_x_data, train_y_data, lr):
        # TODO ramdom choose x from x_data
        # TODO add val()
        losses = []
        for i in range(train_x_data.shape[0]):
            x = train_x_data[i]
            x = x.reshape(x.shape[0], 1)
            y = train_y_data[i]
            y = y.reshape(y.shape[0], 1)
            loss = self.train(x, y, lr).reshape(1)
            losses.append(loss)
        return losses
