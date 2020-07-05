# Re0-Neural-Network

A neural network designed from scratch, using only `numpy` and `matplotlib` library.

The code is availble at [GitHub](https://github.com/wintertee/Re0-Neural-Network)

## Usage

Our code demos are `Dense_sample.py` and `Conv_sample.py`.

In `Dense_sample.py`, you can choose different optimizers by one line among the code as below:

```python3
model.config(optimizer=optimizers.SGD, loss=losses.Crossentropy, lr=0.01, metric=metrics.categorical_accuracy)
model.config(optimizer=optimizers.BCD, loss=losses.Crossentropy, lr=0.1, metric=metrics.categorical_accuracy)
model.config(optimizer=optimizers.BCD_V2, loss=losses.Crossentropy, lr=0.1, metric=metrics.categorical_accuracy)
```

In `Conv_sample.py`, only `optimizer.SGD` is supported, because the other BCD optimizers don't support `layers.maxpool2d`.

## General View

Our coding structure design is like this:

```
+------------------------------------------------------------------+
|                             models                               |
+----------------+-----------------------+---------+---------+-+---+
                 ^                       ^         ^         ^ |
                 |                       |         |         | v
+----------------+-----------------+ +---+---+ +---+--+ +----+-+---+
|              layers              | |metrics| |losses| |optimizers|
+--------------+--------+----------+ +-------+ +------+ +----------+
|linear(Dense) | Conv2d | MaxPool2D|
+--------+-----+--------++---------+
         ^               ^
         |               |
   +-----+-----+  +------+-----+
   |activations|  |initializers|
   +-----------+  +------------+

```
