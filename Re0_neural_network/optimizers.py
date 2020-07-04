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
        self.count += 1


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
        db = np.einsum('ijk,ijj->ijk', (-2 * layer.a + 2 * layer_output), layer.activation.derivative(layer_output))
        dw = np.einsum('ijk,ilk->jl', db, x, optimize=False)
        layer.P['w'] -= self.lr * dw / db.shape[0]
        layer.P['b'] -= self.lr * db.mean(axis=0)

    def _update_a(self, layer, next_layer, last_layer=None, x=None):
        next_layer_output = next_layer.forward(layer.a, update_a=False)
        if x is None:
            x = last_layer.a
        layer_output = layer.forward(x, update_a=False)

        dh_dsigma = -2 * next_layer.a + 2 * next_layer_output  # (out, 1)
        dsigma_dz = next_layer.activation.derivative(next_layer_output)  # (out, out) in S
        dz_da = next_layer.P['w']  # (out, in)
        da = np.matmul(dsigma_dz, dh_dsigma)
        da = np.matmul(dz_da.T, da)
        da += 2 * layer.a - 2 * layer_output

        layer.a -= self.lr * da

    def _update_last_a(self, last_layer, llast_layer, y):
        last_layer.a = y

    def step(self, x, y, first):
        super().step()

        self.loss, self.metric = self.model.forward(x, y, update_a=True)

        layer_1 = self.model.layers[-1]
        layer_2 = self.model.layers[-2]

        self._update_last_a(layer_1, layer_2, y)
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


class BCD_V2(BCD):
    def __init__(self, P, G, lr, model):
        super().__init__(P, G, lr, model)
        self.mu = 1

    def _nu_upper_bound(self, x, y):
        return 1 - (x + y / (2 * self.mu)).max(axis=1)

    def _nu_lower_bound(self, p, x, y, theta):
        return (1 - x.sum(axis=1) - y.sum(axis=1) / (2 * self.mu) * np.exp(theta)) / p

    def _cal_z_star(self, x, y, nu):
        temp = (x + nu) / 2
        return temp + np.sqrt(temp**2 + y / (2 * self.mu))

    def _update_last_a(self, last_layer, llast_layer, y):

        xx = last_layer.forward(llast_layer.a, update_a=False)
        p = self.loss + self.mu * np.linalg.norm(xx - last_layer.a, axis=1)**2
        theta = self.mu * np.linalg.norm(y - last_layer.a, axis=1)**2

        nu_max = self._nu_upper_bound(xx, y)
        nu_min = self._nu_lower_bound(p, xx, y, theta)

        for i in range(xx.shape[0]):  # batch size dim

            while nu_max[i] - nu_min[i] > 1e-2:
                nu = (nu_max[i] + nu_min[i]) / 2
                z_star = self._cal_z_star(xx[i], y[i], nu)
                if z_star.sum() > 1:
                    nu_max[i] = nu
                else:
                    nu_min[i] = nu

            nu = (nu_max[i] + nu_min[i]) / 2

            z_star = self._cal_z_star(xx[i], y[i], nu)

            last_layer.a[i] = z_star

    def step(self, x, y, first):
        self.mu = self.count // 500 + 1
        return super().step(x, y, first)


class BCD_V3(BCD):
    def _update_w_b(self, layer, last_layer=None, x=None):
        if x is None:
            x = last_layer.a
        W = np.zeros(layer.P['w'].shape)  # update W row-wise
        for i in range(x.shape[0]):
            A = x[i]
            first_term = np.linalg.inv(A.dot(A.T) + np.eye(A.shape[0]) * 1e-3).dot(A)
            for j in range(W.shape[0]):
                w_j = first_term * (layer.a[i][j] - layer.P['b'][j])
                W[j] += w_j.flatten()
        layer.P['w'] = W / x.shape[0]

        layer.P['b'] = (layer.a - np.einsum('ij,kjl->kil', layer.P['w'], x)).mean(axis=0)
