# Re0-Neural-Network

A neural network designed from scratch.

## TODO

- [ ] flatten layer
- [ ] conv2d layer
- [ ] metrics
- [x] batch size

## structure

Our coding structure is like this:

```
+------------------------------------------------------------------+
|                             models                               |
+------------------------------------------------------------------+
                 ^                       ^         ^         ^
                 |                       |         |         |
+----------------+-----------------+ +---+---+ +---+--+ +----+-----+
|              layers              | |metrics| |losses| |optimizers|
+--------------+--------+----------+ +-------+ +------+ +----------+
|linear(Dense) | Conv2d | MaxPool2D|
+--------------+-------------------+
         ^               ^
         |               |
   +-----+-----+  +------+-----+
   |activations|  |initializers|
   +-----------+  +------------+
```
