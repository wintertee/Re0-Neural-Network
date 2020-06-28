import numpy as np
import time


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

    def config(self, optimizer=None, loss=None, metric=None, lr=None, **kwargs):

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

    def forward(self, x, y, update_a=True):
        for layer in self.layers:
            x = layer.forward(x, update_a=update_a)

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

    def train(self, x, y, **kwargs):
        loss, metric = self.optimizer.step(x, y, **kwargs)
        return (loss, metric)

    def val(self, x, y):
        loss, metric = self.forward(x, y, update_a=False)
        return (loss, metric)

    def run_epoch(self, epoch, x_train, y_train, x_val, y_val, batch_size, verbose=0):

        all_iter = x_train.shape[0] // batch_size

        train_losses = np.zeros(all_iter)
        train_metrics = np.zeros(all_iter)
        val_losses = np.zeros(all_iter)
        val_metrics = np.zeros(all_iter)

        for i in range(all_iter):  # drop last batch if not full

            x = x_train[i * batch_size:(i + 1) * batch_size]
            y = y_train[i * batch_size:(i + 1) * batch_size]
            train_loss, train_metric = self.train(x, y, first=(i == 0 and epoch == 0))
            # train_loss, train_metric = self.train(x, y, first=True)

            train_losses[i] = train_loss
            train_metrics[i] = train_metric

            j = int(i * x_val.size / (x_val.size + x_train.size))
            x = x_val[j * batch_size:(j + 1) * batch_size]
            y = y_val[j * batch_size:(j + 1) * batch_size]
            val_loss, val_metric = self.val(x, y)

            val_losses[i] = val_loss
            val_metrics[i] = val_metric

            if verbose == 2:
                print("\r{}/{}".format(i, all_iter), end="")

        return (train_losses.mean(), train_metrics.mean(), val_losses.mean(), val_metrics.mean())

    def split(self, x_data, y_data, val_split, shuffle):
        if shuffle:
            state = np.random.get_state()
            np.random.shuffle(x_data)
            np.random.set_state(state)
            np.random.shuffle(y_data)
        val_size = int(val_split * x_data.shape[0])
        x_val = x_data[:val_size]
        y_val = y_data[:val_size]
        x_train = x_data[val_size:]
        y_train = y_data[val_size:]
        return (x_train, y_train, x_val, y_val)

    def fit(self, x_train, y_train, x_val=None, y_val=None, epochs=None, batch_size=None, val_split=None, shuffle=True, verbose=0, freq=1):
        """
        parameters:
            train_x_data : shape(N,x,1)
            shuffle: shuffle train data between each epoch
            verbose: 0 nothing, 1 show loss and accuracy, 2 show progress bar
        """

        train_losses = np.zeros(epochs)
        train_metrics = np.zeros(epochs)
        val_losses = np.zeros(epochs)
        val_metrics = np.zeros(epochs)

        if val_split:
            x_train, y_train, x_val, y_val = self.split(x_train, y_train, val_split, shuffle)

        for i in range(epochs):

            begin_time = time.time()

            train_loss, train_metric, val_loss, val_metric = self.run_epoch(i, x_train, y_train, x_val, y_val, batch_size, verbose=verbose)

            train_losses[i] = train_loss
            train_metrics[i] = train_metric
            val_losses[i] = val_loss
            val_metrics[i] = val_metric

            if verbose >= 1 and i % freq == 0:
                print("epoch: {} train_loss: {:.3f} train_accuracy: {:.2%} val_loss: {:.3f} val_accuracy: {:.2%} time_per_epoch: {:.1f}s".format(
                    i, train_loss, train_metric, val_loss, val_metric,
                    time.time() - begin_time))

        return (train_losses, train_metrics, val_losses, val_metrics)
