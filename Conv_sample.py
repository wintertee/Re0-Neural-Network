from Re0_neural_network import activations, initializers, layers, losses, models, optimizers, metrics

import numpy as np
import matplotlib.pyplot as plt
import time

# load dataset
mnist = np.load('datasets/mnist.npz')
train_images = mnist['train_images']
train_labels = mnist['train_labels']
test_images = mnist['test_images']
test_labels = mnist['test_labels']

# add 1 dim to the last axis of the image
train_images = train_images.reshape(*(train_images.shape), 1).astype(float)
test_images = test_images.reshape(*(test_images.shape), 1).astype(float)

# normalisation
train_images /= 255
test_images /= 255

# one-hot encode
targets = train_labels.reshape(-1)
train_labels = np.eye(10)[targets].reshape(60000, 10, 1)

targets = test_labels.reshape(-1)
test_labels = np.eye(10)[targets].reshape(10000, 10, 1)

batch_size = 20

# model
model = models.Sequential()
model.append(layers.Conv2d(train_images.shape, 2, 3, 1, batch_size, activations.relu()))
model.append(layers.MaxPool2d(26, 2, 2, 2, batch_size))
model.append(layers.Flatten())
model.append(layers.Dense(338, 64, activations.relu, initializers.He))
model.append(layers.Dense(64, 10, activations.softmax, initializers.He))
model.build()

model.config(optimizer=optimizers.SGD,
             loss=losses.Crossentropy,
             lr=0.001,
             batch_size=20,
             metric=metrics.categorical_accuracy)
# model.config(optimizer=optimizers.PRBCD, loss=losses.Crossentropy, lr=0.001)
# model.config(optimizer=optimizers.RCD, loss=losses.Crossentropy, lr=0.001, n=10000)

epochs = range(1, 6)

train_losses = []
train_metrics = []
val_losses = []
val_metrics = []

for epoch in epochs:
    begin_time = time.time()
    train_loss, train_metric, val_loss, val_metric = model.fit(train_images, train_labels, verbose=1)

    train_losses.append(train_loss)
    train_metrics.append(train_metric)
    val_losses.append(val_loss)
    val_metrics.append(val_metric)
    print(
        "epoch: {} train_loss: {:.3f} train_accuracy: {:.2%} val_loss: {:.3f} val_accuracy: {:.2%} time_per_epoch: {:.1f}s"
        .format(epoch, train_loss, train_metric, val_loss, val_metric,
                time.time() - begin_time))

# visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'o', label='train')
plt.plot(epochs, val_losses, label='val')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_metrics, 'o', label='train')
plt.plot(epochs, val_metrics, label='val')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()

# evaluate on test set
print("evaluate on test set")
loss, metric = model.val(test_images, test_labels)
print("loss: {:.3f} accuracy:  {:.2%}".format(loss, metric))
