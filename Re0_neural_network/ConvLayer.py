import numpy as np
from functools import reduce
import math

class ConvLayer(object):
    def __init__(self,input_image,output_channels,kernel_size,padding,activation=None):

        # initialize propoties
        self.input_image = input_image
        self.input_channels = input_image[-1]
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.batchsize = input_image[0]

        self.eta = np.zeros((input_image[0], (input_image[1] - kernel_size + 1), (input_image[1] - kernel_size + 1), self.output_channels))

        # initialize parameters
        self.P = {}
        self.G = {}

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, input_image) / self.output_channels)
        self.P['w'] = np.random.standard_normal((kernel_size, kernel_size, self.input_channels, self.output_channels)) / weights_scale
        self.P['b'] = np.random.standard_normal(self.output_channels) / weights_scale

        self.G['w'] = np.zeros(self.P['w'].shape)
        self.G['b'] = np.zeros(self.P['b'].shape)
        self.output_image = self.eta.shape

    def forward(self,x):
        col_weights = self.P['w'].reshape([-1, self.output_channels])
        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = im2col(img_i, self.kernel_size)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.P['b'], self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out




    def backward(self,dL_da,alpha=0.00001, weight_decay=0.0004):
        # 避免过拟合，可考虑删除
        self.P['w'] *= (1 - weight_decay)
        self.P['b'] *= (1 - weight_decay)
        self.P['w'] -= alpha * self.G['w']
        self.P['b'] -= alpha * self.P['b']

        col_delta = np.reshape(dL_da, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.G['w'] += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.G['w'].shape)
        self.G['b'] += np.sum(col_delta, axis=(0, 1))

def im2col(image, kernel_size):
    image_col = []
    for i in range(0, image.shape[1] - kernel_size + 1, 1):
        for j in range(0, image.shape[2] - kernel_size + 1, 1):
            col = image[:, i:i + kernel_size, j:j + kernel_size, :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col

if __name__ == "__main__":

    img = np.ones((1, 4, 4, 1))   # batchsize, width, height, channel
    print("img:", img)

    img *= 2
    print("img:",img)

    conv = ConvLayer(img.shape, 2, 3, 1)

    next = conv.forward(img)
    print("next:",next)

    next1 = next.copy() + 1
    print("next1:",next1)

    print("w:",conv.G['w'])
    print("b:", conv.G['b'])

    conv.backward(next1-next)

    print("w:",conv.G['w'])
    print("b:", conv.G['b'])