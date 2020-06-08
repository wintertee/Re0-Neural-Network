import numpy as np


class Loss:
    @staticmethod
    def forward(x):
        raise NotImplementedError

    @staticmethod
    def backward(a):
        raise NotImplementedError


class Crossentropy(Loss):
    @staticmethod
    def forward(pred, truth):
        a = []
        for i in range(pred.shape[0]):
            a.append(-np.dot(truth[i].T, np.log(pred[i])))
        return np.mean(a)

    @staticmethod
    def backward(pred, truth):
        return pred - truth
