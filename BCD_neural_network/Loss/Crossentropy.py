import numpy as np
from .Loss import Loss


class loss(Loss):
    @staticmethod
    def forward(pred, truth):
        return - np.dot(truth, np.log(pred))

    @staticmethod
    def derivative(pred, truth):
        return pred - truth