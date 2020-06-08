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
            print("Batch size: " + str(batch_size))

        if loss is not None:
            self.loss = loss
            print("Loss: " + loss.__name__)

        if metric is not None:
            self.metric = metric
            print("metric: " + metric.__name__)

        if lr is not None:
            self.lr = lr
            print("learning rate: " + str(lr))

        if optimizer is not None:
            try:
                self.optimizer = optimizer(self.P, self.G, self.lr, self, **kwargs)
                print("optimizer: " + optimizer.__name__)
            except NameError:
                print("ERROR: build and set lr before set optimizer")

        print("===========================================")

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
        loss, metric = self.forward(x, y)
        return (loss, metric)

    def fit(self, x_data, y_data, lr, val_split=0.2, shuffle=True):
        """
        parameters:
            train_x_data : shape(N,x,1)
            shuffle: shuffle train data between each epoch
        """

        train_losses = []
        train_metrics = []
        val_losses = []
        val_metrics = []

        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(x_data)
            np.random.set_state(state)
            np.random.shuffle(y_data)

        # split val data
        val_size = int(val_split * x_data.shape[0])
        val_x = x_data[:val_size]
        val_y = y_data[:val_size]
        train_x = x_data[val_size:]
        train_y = y_data[val_size:]

        for i in range(train_x.shape[0] // self.batch_size):  # drop last batch if not full
            # 批量梯度累加？？？

            x = train_x[i * self.batch_size:(i + 1) * self.batch_size]
            y = train_y[i * self.batch_size:(i + 1) * self.batch_size]
            train_loss, train_metric = self.train(x, y)

            train_losses.append(train_loss)
            train_metrics.append(train_metric)

            j = int(i * val_split)
            x = val_x[j * self.batch_size:(j + 1) * self.batch_size]
            y = val_y[j * self.batch_size:(j + 1) * self.batch_size]
            val_loss, val_metric = self.val(x, y)

            val_losses.append(val_loss)
            val_metrics.append(val_metric)

        return (np.mean(train_losses), np.mean(train_metrics), np.mean(val_losses), np.mean(val_metrics))
