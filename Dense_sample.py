from Re0_neural_network import activations, initializers, layers, losses, models, optimizers, metrics

import numpy as np
import matplotlib.pyplot as plt

# load dataset
mnist = np.load('datasets/mnist.npz')
train_images = mnist['train_images']
train_labels = mnist['train_labels']
test_images = mnist['test_images']
test_labels = mnist['test_labels']

# flatten image
train_images = train_images.reshape(*(train_images.shape), 1).astype(float)
# normalisation
train_images /= 255

# one-hot encode
targets = train_labels.reshape(-1)
train_labels = np.eye(10)[targets].reshape(60000, 10, 1)

model = models.Sequential()
model.append(layers.Flatten())
model.append(layers.Dense(784, 64, activations.relu, initializers.He))
model.append(layers.Dense(64, 10, activations.softmax, initializers.He))
model.build()

model.config(optimizer=optimizers.SGD,
             loss=losses.Crossentropy,
             lr=0.001,
             batch_size=20,
             metric=metrics.categorical_accuracy)
# model.config(optimizer=optimizers.PRBCD, loss=losses.Crossentropy, lr=0.001)
# model.config(optimizer=optimizers.RCD, loss=losses.Crossentropy, lr=0.001, n=10000)

losses = []
metrics = []
for epoch in range(1, 6):
    loss, metric = model.fit(train_images, train_labels, 0.001)
    loss = np.mean(loss)
    metric = np.mean(metric)
    losses.append(loss)
    metrics.append(metric)
    print("epoch: {} loss: {:.3f} accuracy: {:.2%}".format(epoch, loss, metric))

epochs = range(1, len(losses) + 1)

# visualize
plt.figure(figsize=(4, 8))

plt.subplot(2, 1, 1)
plt.plot(epochs, losses, label='loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(epochs, metrics, label='accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()
