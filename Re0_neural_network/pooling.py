import numpy as np
from functools import reduce
import math

def get_patch(input_array, i, j, filter_width,
              filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        input_array_conv = input_array[
                start_i : start_i + filter_height,
                start_j : start_j + filter_width]
        return input_array_conv
    elif input_array.ndim == 3:
        input_array_conv = input_array[:,
            start_i : start_i + filter_height,
            start_j : start_j + filter_width]
        return input_array_conv

def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i,j] > max_value:
                max_value = array[i,j]
                max_i, max_j = i, j
    return max_i, max_j

class MaxPoolingLayer(object):
    def __init__(self, input_size, channel_number, kernel_size, stride = -1 ): # 默认步长为池化核尺寸
        self.input_size = input_size
        self.channel_number = channel_number
        self.kernel_size = kernel_size
        if stride == -1:
            self.stride = kernel_size
        else:
            self.stride = stride
        self.output_size = (input_size - kernel_size) // self.stride + 1
        self.output_matrix = np.zeros((self.channel_number, self.output_size, self.output_size))
        self.delta_matrix = []


    def forward(self, input_image):
        for d in range(self.channel_number):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    self.output_matrix[d, i, j] = get_patch(input_image[d], i, j,
                                                            self.kernel_size,
                                                            self.kernel_size,
                                                            self.stride).max()
        return self.output_matrix

    def backward(self, input_image):
        dL_da = self.output_matrix.copy()+1-self.output_matrix
        N = 1/(1+(self.input_size-self.kernel_size)//self.stride)**2
        print(N)
        self.delta_matrix = np.zeros(input_image.shape)

        for d in range(self.channel_number):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    patch_image = get_patch(
                        input_image[d], i, j,
                        self.kernel_size,
                        self.kernel_size,
                        self.stride)
                    k, l = get_max_index(patch_image)
                    self.delta_matrix[d,
                                     i * self.stride + k,
                                     j * self.stride + l] += dL_da[d, i, j] * N

if __name__ == "__main__":
    image = np.array(
        [[
            [1, 3, 4, 2, 5, 2],
            [4, 2, 5, 6, 8, 3],
            [5, 3, 7, 8, 9, 3],
            [8, 4, 3, 6, 0, 2],
            [7, 10, 3, 6, 3, 5],
            [4, 7, 2, 4, 7, 8]
        ]]
    )

    pool = MaxPoolingLayer(6,1,2)
    next = pool.forward(image)
    print(next)
    pool.backward(image)
    print(pool.delta_matrix)
