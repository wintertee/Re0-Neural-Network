import numpy as np
from functools import reduce
import math


class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, in_features, out_features, activation=None, initializer=None):

        # initialize properties
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # initialize parameters
        self.P = {}  # parameters
        self.G = {}  # gradients
        if initializer:
            self.P['w'] = initializer((out_features, in_features))
        else:
            self.P['w'] = np.ones((out_features, in_features))
        self.P['b'] = np.zeros((out_features, 1))

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.x = x
        z = np.einsum('jk,ikl->ijl', self.P['w'], self.x, optimize=True) + self.P['b']
        if self.activation is None:
            self.a = z
        else:
            self.a = self.activation.forward(z)
        return self.a

    def backward(self, dL_da):
        if self.activation is None:
            da_dz = np.ones((self.batch_size, self.out_features, 1))
        else:
            da_dz = self.activation.backward(self.a)

        dL_dz = dL_da * da_dz
        self.G['w'] = np.einsum('ijk,ilk->ijl', dL_dz, self.x, optimize=True)
        dL_dx = np.einsum('kj,ikl->ijl', self.P['w'], dL_dz, optimize=True)
        self.G['b'] = dL_dz

        self.G['w'] = np.mean(self.G['w'], axis=0)
        self.G['b'] = np.mean(self.G['b'], axis=0)

        return dL_dx  # NOTE x is the `a` in the last layer


class Flatten(Layer):
    def __init__(self):
        self.P = {}
        self.G = {}
        self.P['w'] = np.array([])
        self.P['b'] = np.array([])
        self.G['w'] = np.array([])
        self.G['b'] = np.array([])

    def forward(self, x):
        self.original_shape = x.shape
        size = x.size // x.shape[0]
        return x.reshape(x.shape[0], size, 1)

    def backward(self, dL_da):
        return dL_da.reshape(self.original_shape)


class Conv2d(Layer):
    def __init__(self, input_image_shape, output_channels, kernel_size, activation=None):

        # initialize properties
        self.input_image_shape = input_image_shape
        self.input_channels = input_image_shape[-1]
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.activation = activation

        # initialize parameters
        self.P = {}
        self.G = {}

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, input_image_shape) / self.output_channels)
        self.P['w'] = np.random.standard_normal(
            (kernel_size, kernel_size, self.input_channels, self.output_channels)) / weights_scale
        self.P['b'] = np.random.standard_normal(self.output_channels) / weights_scale

        self.G['w'] = np.zeros(self.P['w'].shape)
        self.G['b'] = np.zeros(self.P['b'].shape)

    def im2col(self, image, kernel_size):
        image_col = []
        for i in range(0, image.shape[1] - kernel_size + 1, 1):
            for j in range(0, image.shape[2] - kernel_size + 1, 1):
                col = image[:, i:i + kernel_size, j:j + kernel_size, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col

    def forward(self, x):

        # initialization
        if not hasattr(self, 'batchsize'):
            self.batchsize = x.shape[0]
        if not hasattr(self, 'eta'):
            self.eta = np.zeros((self.batchsize, self.input_image_shape[1] - self.kernel_size + 1,
                                 self.input_image_shape[1] - self.kernel_size + 1, self.output_channels))
        if not hasattr(self, 'output_size'):
            self.output_size = self.eta.shape

        col_weights = self.P['w'].reshape([-1, self.output_channels])
        self.col_image = []
        self.a = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self.im2col(img_i, self.kernel_size)
            self.a[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.P['b'], self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        self.z = self.a
        if self.activation is None:
            return self.a
        else:
            self.a = self.activation.forward(self.a)
            return self.a

    def backward(self, dL_da):

        if self.activation is None:
            col_delta = np.reshape(dL_da, [self.batchsize, -1, dL_da.shape[3]])
        else:
            col_delta = self.activation.backward(self.a)
            col_delta = dL_da
            col_delta = np.reshape(col_delta, [self.batchsize, -1, dL_da.shape[3]])

        for i in range(self.batchsize):
            self.G['w'] += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.G['w'].shape)
        self.G['b'] += np.sum(col_delta, axis=(0, 1))

        pad_eta = np.pad(self.eta, ((0, 0), (self.kernel_size - 1, self.kernel_size - 1),
                                    (self.kernel_size - 1, self.kernel_size - 1), (0, 0)),
                         'constant',
                         constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.P['w']))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array(
            [self.im2col(pad_eta[i][np.newaxis, :], self.kernel_size) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(
            next_eta, (self.batchsize, self.input_image_shape[1], self.input_image_shape[2], self.input_image_shape[3]))
        return next_eta


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=-1):  # 默认步长为池化核尺寸

        self.kernel_size = kernel_size
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride

        self.P = {}
        self.G = {}
        self.P['w'] = np.array([])
        self.P['b'] = np.array([])
        self.G['w'] = np.array([])
        self.G['b'] = np.array([])
        self.delta_matrix = []

    def forward(self, input_image):

        # initialization
        if not hasattr(self, 'batchsize'):
            self.batchsize = input_image.shape[0]
        if not hasattr(self, 'input_size'):
            self.input_size = input_image.shape[1]
        if not hasattr(self, 'channel_number'):
            self.channel_number = input_image.shape[3]
        if not hasattr(self, 'output_size'):
            self.output_size = (self.input_size - self.kernel_size) // self.stride + 1
        if not hasattr(self, 'output_matrix'):
            self.output_matrix = np.zeros((self.batchsize, self.output_size, self.output_size, self.channel_number))

        self.edge = input_image.shape[1] % self.kernel_size
        if self.edge != 0:
            input_image = np.pad(input_image,
                                 ((0, 0), (0, self.kernel_size - self.edge), (0, self.kernel_size - self.edge), (0, 0)),
                                 'constant',
                                 constant_values=-1)

        self.input_image = input_image

        # for backward
        self.delta_matrix = np.zeros((self.batchsize, self.input_size, self.input_size, self.channel_number))
        self.max_index_matrix = np.zeros(
            (self.batchsize, self.input_size, self.input_size, self.channel_number, 2)).astype(int)

        for i in range(self.output_size):
            for j in range(self.output_size):
                patch_image = self.get_patch(self.input_image[:, :, :, :], i, j, self.kernel_size, self.kernel_size,
                                             self.stride)
                self.output_matrix[:, i, j, :] = patch_image.max()  # forward
                max_index = self.argmax(patch_image)  # backward
                self.max_index_matrix[:, i, j, :] = max_index

        if (self.output_matrix != self.forward2(input_image)).any():
            raise ValueError

        return self.output_matrix

    def backward(self, dL_da):
        self.delta_matrix = np.zeros((self.batchsize, self.input_size, self.input_size, self.channel_number))
        for i in range(self.output_size):
            for j in range(self.output_size):
                k, l = self.max_index_matrix[:, i, j, :, 0], self.max_index_matrix[:, i, j, :, 1]  # noqa: E741
                for p in range(self.batchsize):
                    for q in range(self.channel_number):
                        self.delta_matrix[p, i * self.stride + k[p][q], j * self.stride + l[p][q], q] += dL_da[p, i, j,
                                                                                                               q]

        if (self.delta_matrix != self.backward2(dL_da)).any():
            raise ValueError
        return self.delta_matrix

    def argmax(self, x):
        """
        x: (N, W, H, C)
        return: (N, W, H, C)
        """
        # (N, W, H, C)
        x = x.transpose(1, 2, 0, 3)  # (W, H, N, C)
        shape = x.shape
        x = x.reshape(shape[0] * shape[1], shape[2], shape[3])  # (W * H, N, C)
        index = np.unravel_index(np.argmax(x, axis=0), (shape[0], shape[1]))  # (W_index, N, C), (H_index, N, C)
        index = np.stack(index, axis=-1)  # (N, C, 2)
        return index

    def get_patch(self, input_array, i, j, filter_width, filter_height, stride):
        start_i = i * stride
        start_j = j * stride

        input_array_conv = input_array[:, start_i:start_i + filter_height, start_j:start_j + filter_width, :]
        return input_array_conv

    def forward2(self, input_image):
        self.input_image = input_image
        for k in range(self.batchsize):
            for d in range(self.channel_number):
                for i in range(self.output_size):
                    for j in range(self.output_size):
                        self.output_matrix[k, i, j, d] = self.get_patch2(input_image[k, :, :, d], i, j, self.kernel_size,
                                                                        self.kernel_size, self.stride).max()
        return self.output_matrix

    def backward2(self, dL_da):
        input_image = self.input_image
        # N = 1 / (1 + (self.input_size - self.kernel_size) // self.stride)**2
        self.delta_matrix = np.zeros((self.batchsize, self.input_size, self.input_size, self.channel_number))
        for m in range(self.batchsize):
            for d in range(self.channel_number):
                for i in range(self.output_size):
                    for j in range(self.output_size):
                        patch_image = self.get_patch2(input_image[m, :, :, d], i, j, self.kernel_size, self.kernel_size,
                                                      self.stride)
                        k, l = self.get_max_index2(patch_image)  # noqa: E741
                        self.delta_matrix[m, i * self.stride + k, j * self.stride + l, d] += dL_da[m, i, j, d]
        return self.delta_matrix

    def get_max_index2(self, array):
        max_i = 0
        max_j = 0
        max_value = 0
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] > max_value:
                    max_value = array[i, j]
                    max_i, max_j = i, j
        return max_i, max_j

    def get_patch2(self, input_array, i, j, filter_width, filter_height, stride):
        start_i = i * stride
        start_j = j * stride
        if input_array.ndim == 2:
            input_array_conv = input_array[start_i:start_i + filter_height, start_j:start_j + filter_width]
            return input_array_conv
        elif input_array.ndim == 3:
            input_array_conv = input_array[:, start_i:start_i + filter_height, start_j:start_j + filter_width]
            return input_array_conv
