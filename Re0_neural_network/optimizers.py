import numpy as np


class Optimizer:
    def __init__(self, P, G, lr, model):
        """
        parameters:
            lr: learning rate
            P:  list of dictionaries of parameters, containing key 'w' and 'b'
            G:  list of dictionaries of gradients, containing key 'dw' and 'db'
        """
        assert len(P) == len(G)
        self.P = P
        self.G = G
        self.lr = lr
        self.model = model
        self.layer_numbers = len(P)  # real number is layer_numbers + 1 !
        self.count = 0

    def step(self):
        """
        forward and backward.
        need to update P and return (loss, metric) in child classes
        """
        # self.count += 1
        # if self.count % 100 == 0:
        #     self.lr *= 2


class SGD(Optimizer):
    def step(self, x, y, **kwargs):
        super().step()
        self.loss, self.metric = self.model.forward(x, y, True)
        self.model.backward(y)
        for layer in range(self.layer_numbers):
            self.P[layer]['w'] -= self.lr * self.G[layer]['w']
            self.P[layer]['b'] -= self.lr * self.G[layer]['b']
        return (self.loss, self.metric)


class BCD(Optimizer):

    def _update_w_b(self, layer, last_layer=None, x=None):
        if x is None:
            x = last_layer.a
        layer_output = layer.forward(x, update_a=False)

        db = (-2 * layer.a + 2 * layer_output) * layer.activation.derivative(layer_output)
        dw = np.einsum('ijk,ilk->jl', db, x)
        layer.P['w'] -= self.lr * dw
        layer.P['b'] -= self.lr * db.sum(axis=0)

    def _update_a(self, layer, next_layer, last_layer=None, x=None):
        next_layer_output = next_layer.forward(layer.a, update_a=False)
        if x is None:
            x = last_layer.a
        layer_output = layer.forward(x, update_a=False)

        da = np.einsum('ijk,jl->ilk',(-2 * next_layer.a + 2 * next_layer_output) * next_layer.activation.backward(next_layer_output),next_layer.P['w'])
        da += 2 * layer.a - 2 * layer_output

        layer.a -= self.lr * da

    def step(self, x, y, first):
        super().step()
        self.loss, self.metric = self.model.forward(x, y, update_a=first)

        layer_1 = self.model.layers[-1]
        layer_2 = self.model.layers[-2]

        if (self.model.pred != layer_1.a).any():
            raise ValueError

        # da = self.model.loss.derivative(self.model.pred, y)
        # da = 2 * layer_1.a - 2 * y
        # layer_1.a -= da
        layer_1.a = y
        self._update_w_b(layer_1, layer_2)

        for layer_i in range(self.layer_numbers - 2, -1, -1):

            layer = self.model.layers[layer_i]
            next_layer = self.model.layers[layer_i + 1]
            

            if layer_i == 0:
                self._update_a(layer, next_layer, x=x)
                self._update_w_b(layer, x=x)
            else:
                last_layer = self.model.layers[layer_i - 1]
                self._update_a(layer, next_layer, last_layer=last_layer)
                self._update_w_b(layer, last_layer=last_layer)

            

        return (self.loss, self.metric)





class PRBCD(Optimizer):
    """
    "pseudo" randomized block coordinate descent
    """
    def step(self, x, y):
        super().step(x, y)
        layer = np.random.randint(self.layer_numbers)
        key = 'w' if np.random.randint(1) == 0 else 'b'
        self.P[layer][key] -= self.lr * self.G[layer][key]

        return (self.loss, self.metric)


class RCD(Optimizer):
    def __init__(self, P, G, lr, model, n=1):
        """
        parameters:
            n:  number of parameters to update at each iteration
        """
        super().__init__(P, G, lr, model)
        self.n = n

        # create all combinations of ['w', layer, i, j]
        layers = np.arange(self.layer_numbers).astype(int)
        for layer in layers:
            i = np.arange(self.P[layer]['w'].shape[0]).astype(int)
            j = np.arange(self.P[layer]['w'].shape[1]).astype(int)
            i, j = np.meshgrid(i, j)
            i = i.reshape(i.size, 1)
            j = j.reshape(j.size, 1)
            layer_index_list = np.ones(i.size).reshape(i.size, 1).astype(int) * layer
            layer_index_list = np.hstack((layer_index_list, i, j))
            if layer == 0:
                layer_list = layer_index_list
            else:
                layer_list = np.vstack((layer_list, layer_index_list))

        layer_list = layer_list.astype(int)

        w = np.zeros(layer_list.shape[0]).reshape(layer_list.shape[0], 1).astype(int)  # 0 for 'w'

        w = np.hstack((w, layer_list))

        # create all combinations of ['b', layer, i, j]

        layers = np.arange(self.layer_numbers).astype(int)
        layer_list = []
        for layer in layers:
            i = np.arange(self.P[layer]['b'].shape[0]).astype(int)
            j = np.arange(self.P[layer]['b'].shape[1]).astype(int)
            i, j = np.meshgrid(i, j)
            i = i.reshape(i.size, 1)
            j = j.reshape(j.size, 1)
            layer_index_list = np.ones(i.size).reshape(i.size, 1).astype(int) * layer
            layer_index_list = np.hstack((layer_index_list, i, j))
            if layer == 0:
                layer_list = layer_index_list
            else:
                layer_list = np.vstack((layer_list, layer_index_list))

        layer_list = layer_list.astype(int)

        b = np.ones(layer_list.shape[0]).reshape(layer_list.shape[0], 1).astype(int)  # 1 for 'b'

        b = np.hstack((b, layer_list))

        # all combinations of ['b', layer, i, j] and ['w', layer, i, j]
        self.all_list = np.vstack((w, b)).astype(int)

    def step(self, x, y):

        np.random.shuffle(self.all_list)

        for k in range(0, self.all_list.shape[0] + 1, self.n):
            min_layer = min(self.all_list[k + kk][1] for kk in range(self.n) if k + kk < self.all_list.shape[0])
            # only backward to the last layer whose parameters need to be updated
            super().step(x, y, min_index=min_layer)
            for kk in range(self.n):
                index = k + kk
                if index == self.all_list.shape[0]:
                    break
                if self.all_list[index][0] == 0:
                    self.P[self.all_list[index][1]]['w'][self.all_list[index][2]][self.all_list[index][3]] -= self.lr *\
                    self.G[self.all_list[index][1]]['w'][self.all_list[index][2]][self.all_list[index][3]]  # noqa: E122
                if self.all_list[index][0] == 1:
                    self.P[self.all_list[index][1]]['b'][self.all_list[index][2]][self.all_list[index][3]] -= self.lr *\
                    self.G[self.all_list[index][1]]['b'][self.all_list[index][2]][self.all_list[index][3]]  # noqa: E122

        return (self.loss, self.metric)


class PBCSCD(Optimizer):
    """
    "pseudo" Block-Cyclic Stochastic Coordinate
    """
    def step(self, x, y):
        k = self.model.batch_size // (self.layer_numbers + 1)  # number samples used to calculate gradient
        L_layer = [x for x in range(self.layer_numbers)]  # liste of number of layers

        for i in range(self.layer_numbers):
            layer_i = L_layer[np.random.randint(len(L_layer))]  # randomly choose a layer
            x_i = x[i * k:(i + 1) * k]  # samples used to calculate gradient
            y_i = y[i * k:(i + 1) * k]

            super().step(x_i, y_i)

            self.P[layer_i]['w'] -= self.lr * self.G[layer_i]['w']  # update parameters of the randomly chosen layer
            self.P[layer_i]['b'] -= self.lr * self.G[layer_i]['b']

            L_layer.remove(
                layer_i
            )  # the layer whos parameters that have been updated will be removed from the list so that all parameters will be updated in one epoch
        return (self.loss, self.metric)
