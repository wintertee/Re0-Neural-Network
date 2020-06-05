from BCD_neural_network.Model import Model
from BCD_neural_network.Activation import relu
from BCD_neural_network.Activation import softmax
from BCD_neural_network.Layer import Dense
from BCD_neural_network.Loss import Crossentropy
from BCD_neural_network.Initializer import He

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
train_images = train_images.reshape(60000, 784).astype(float)
train_images /= 255

targets = train_labels.reshape(-1)
train_labels = np.eye(10)[targets]

model = Model.Model()
model.append(Dense.Dense(784, 64, relu.relu, He.He))
model.append(Dense.Dense(64, 10, softmax.softmax, He.He))
model.setLoss(Crossentropy.Crossentropy)

losses = []
for epoch in range(20):
    loss = model.fit(train_images, train_labels, 0.001)
    loss = np.array(loss).mean()
    losses.append(loss)
    print("epoch: " + str(epoch) + " loss: " + str(loss))

epochs = range(1, len(losses) + 1)

plt.plot(epochs, losses, label='loss')
plt.show()
