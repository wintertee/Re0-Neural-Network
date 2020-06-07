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
        print("===========================================")
        print("              Model summary                ")
        print("===========================================")
        self.P = []
        self.G = []
        parameters = []
        for i, layer in enumerate(self.layers):
            self.P.append(layer.P)
            self.G.append(layer.G)
            parameters.append(layer.P['w'].size + layer.P['b'].size)
            print("layer {} {:<10} parameters: {:<10}".format(i + 1, layer.__class__.__name__, parameters[i]))
        print("===========================================")
        print("total parameters: {}".format(np.sum(parameters)))
        print("===========================================")

    def config(self, optimizer=None, loss=None, metric=None, lr=None, batch_size=None, **kwargs):
        if batch_size is not None:
            self.batch_size = batch_size

        if loss is not None:
            self.loss = loss

        if metric is not None:
            self.metric = metric

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
        metric = self.metric(self.pred, y)
        return (loss, metric)

    def backward(self, y, min_index=0):
        dL_da = self.loss.backward(self.pred, y)
        for i, layer in reversed(list(enumerate(self.layers))):
            if i < min_index:
                break
            dL_da = layer.backward(dL_da)

    def train(self, x, y):
        loss, metric = self.optimizer.step(x, y)
        return (loss, metric)

    def val(self, x, y):
        loss = self.forward(x, y)
        return loss

    def fit(self, train_x_data, train_y_data, lr, shuffle=True):
        """
        parameters:
            train_x_data : shape(N,x,1)
        """
        losses = []
        metrics = []

        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(train_x_data)
            np.random.set_state(state)
            np.random.shuffle(train_y_data)

        for i in range(train_x_data.shape[0] // self.batch_size):  # drop last batch if not full
            x = train_x_data[i * self.batch_size:(i + 1) * self.batch_size]
            y = train_y_data[i * self.batch_size:(i + 1) * self.batch_size]

            loss, metric = self.train(x, y)
            losses.append(loss)
            metrics.append(metric)
        return (losses, metrics)
