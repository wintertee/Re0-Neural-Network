# Re0-Neural-Network

A neural network designed from scratch.

## TODO

- [x] flatten layer
- [x] conv2d layer
- [x] metrics
- [x] batch size
- [ ] validation/evaluation

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
