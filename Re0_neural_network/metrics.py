import numpy as np


def categorical_accuracy(pred, y):

    pred_index = np.argmax(pred, axis=1)
    y_index = np.argmax(y, axis=1)
    bingo = pred_index == y_index
    return bingo.astype(int).mean()
