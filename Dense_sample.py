from Re0_neural_network import activations, initializers, layers, losses, models, optimizers, metrics

import numpy as np
import matplotlib.pyplot as plt

# load dataset
mnist = np.load('datasets/mnist.npz')
train_images = mnist['train_images']
train_labels = mnist['train_labels']
test_images = mnist['test_images']
test_labels = mnist['test_labels']

# add 1 dim to the last axis of the image
train_images = train_images.reshape(*(train_images.shape), 1).astype(float)
train_images = train_images.reshape(60000, 784, 1).astype(float)
test_images = test_images.reshape(*(test_images.shape), 1).astype(float)
test_images = test_images.reshape(10000, 784, 1).astype(float)

# normalisation
train_images /= 255
test_images /= 255

# one-hot encode
targets = train_labels.reshape(-1)
train_labels = np.eye(10)[targets].reshape(60000, 10, 1)

targets = test_labels.reshape(-1)
test_labels = np.eye(10)[targets].reshape(10000, 10, 1)

# model
model = models.Sequential()
# model.append(layers.Flatten())
model.append(layers.Dense(784, 16, activations.relu, initializers.He))
model.append(layers.Dense(16, 10, activations.softmax, initializers.He))
model.build()

# model.config(optimizer=optimizers.SGD, loss=losses.Crossentropy, lr=0.01, metric=metrics.categorical_accuracy)
# model.config(optimizer=optimizers.PRBCD, loss=losses.Crossentropy, lr=0.001)
# model.config(optimizer=optimizers.RCD, loss=losses.Crossentropy, lr=0.001, n=10000)
model.config(optimizer=optimizers.BCD, loss=losses.Crossentropy, lr=0.1, metric=metrics.categorical_accuracy)

epoch = 100
batch_size = 60000

train_losses, train_metrics, val_losses, val_metrics = model.fit(train_images, train_labels, test_images, test_labels, epoch, batch_size, verbose=1, freq=1)

# visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(epoch), train_losses, label='train')
plt.plot(range(epoch), val_losses, label='val')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(epoch), train_metrics, label='train')
plt.plot(range(epoch), val_metrics, label='val')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()
