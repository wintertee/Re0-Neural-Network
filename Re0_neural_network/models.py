import numpy as np


class Sequential:
    def __init__(self):
        self.layers = []

    def append(self, layer):
        self.layers.append(layer)

    def build(self):
        """
        add all parameters to model
        """
        self.P = []
        self.G = []
        for i, layer in enumerate(self.layers):
            self.P.append(layer.P)
            self.G.append(layer.G)
            print("layer " + str(i) + " parameters: " + str(layer.P['w'].size + layer.P['b'].size))

    def config(self, optimizer=None, loss=None, metrics=None, lr=None, batch_size=None, **kwargs):
        if batch_size is not None:
            self.batch_size = batch_size

        if loss is not None:
            self.loss = loss

        if metrics is not None:
            self.metrics = metrics

        if lr is not None:
            self.lr = lr

        if optimizer is not None:
            try:
                self.optimizer = optimizer(self.P, self.G, self.lr, self, **kwargs)
            except NameError:
                print("ERROR: build and set lr before set optimizer")

    def forward(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)

        self.pred = x
        loss = self.loss.forward(self.pred, y)
        return loss

    def backward(self, y, min_index=0):
        dL_da = self.loss.backward(self.pred, y)
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < min_index:
                break
            dL_da = layer.backward(dL_da)

    def train(self, x, y):
        loss = self.optimizer.step(x, y)
        return loss

    def val(self, x, y):
        loss = self.forward(x, y)
        return loss

    def fit(self, train_x_data, train_y_data, lr, shuffle=True):
        """
        parameters:
            train_x_data : shape(N,x,1)
        """
        losses = []

        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(train_x_data)
            np.random.set_state(state)
            np.random.shuffle(train_y_data)

        for i in range(train_x_data.shape[0] // self.batch_size):  # drop last batch if not full
            x = train_x_data[i * self.batch_size:(i + 1) * self.batch_size]
            y = train_y_data[i * self.batch_size:(i + 1) * self.batch_size]

            loss = self.train(x, y).mean()
            losses.append(loss)
        return losses
