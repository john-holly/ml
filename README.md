# ML

This repo houses hand-rolled perceptron and back propagation algorithms in Python.

## Requirements

`pip install -r requirements.txt`

## Example Usage

Assume the following format for data (list of tuples - where tuple[0] are the inputs and tuple[1] is the expected label)

```python
XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]
```

### Perceptron

```python
from perceptron import Perceptron

OR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]

neuron = Perceptron(2)
neuron.train(OR, 3000, 0.1) # Where or is a list of tuple/list where inputs are at index 0 and output is at index 1
print("%s -> %d" % ([0, 0], neuron.activate([0, 0], binarize=True)))
```

Output

```
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 1
```

### Back Propagation Neural Network

```python
from bpnn import BPNN

XOR = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# The network shape is 2 inputs, 2 hidden nodes and 2 
# possible labels (0 or 1, we use argmax to turn this into a 
# number rather than an array of probabilities for both)
ann = BPNN([2, 2, 2], 2)
ann.train(XOR, 5000, 0.4)
print("%s -> %d" % ([0, 0], ann.forward([0.0, 0.0], argmax=True)))
```

Output

```
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```