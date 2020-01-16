import numpy as np
from perceptron import Perceptron


class BPNN(object):
    def __init__(self, shape):
        """ Construct a backpropagation network
        :param shape: Where shape[0] is equal to the number of inputs
        and shape[-1] is equal to the number of possible labels"""
        self.layers = []
        for l in range(0, len(shape) - 1):
            layer = []
            for _ in range(shape[l + 1]):
                layer.append(Perceptron(shape[l]))
            self.layers.append(layer)

    def forward(self, inputs, argmax=False):
        """ Forward propagation
        :param inputs:
        :param argmax: Return the index of label with highest
        probability instead of an array of all possible labels
        :return: output """
        x = inputs
        for layer in self.layers:
            layer_outs = []
            for neuron in layer:
                layer_outs.append(neuron.activate(x))
            x = layer_outs
        return x if not argmax else np.argmax(x)

    def backward(self, expected):
        """ Backward propagation
        :param expected: Expected output
        :return:
        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            errors = list()
            if i != len(self.layers) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.layers[i + 1]:
                        error += (neuron.weights[j] * neuron.delta)
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron.output)
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = errors[j] * neuron.derivative(neuron.output)

    def update_weights(self, inputs, learning_rate):
        """ Update network weights
        :param inputs:
        :param learning_rate:
        :return:
        """
        for i in range(len(self.layers)):
            if i != 0:
                inputs = [neuron.output for neuron in self.layers[i - 1]]
            for neuron in self.layers[i]:
                for j in range(len(inputs)):
                    neuron.weights[j] += learning_rate * neuron.delta * inputs[j]
                neuron.weights[-1] += learning_rate * neuron.delta

    def train_one(self, inputs, expected, learning_rate):
        """ Train a single sample
        :param inputs:
        :param expected: Expected output
        :param learning_rate:
        :return: Sum error
        """
        outputs = self.forward(inputs)
        expected_outs = [0 for i in range(len(self.layers[-1]))]
        expected_outs[expected] = 1
        self.backward(expected_outs)
        self.update_weights(inputs, learning_rate)
        return sum([(expected_outs[i] - outputs[i]) ** 2 for i in range(len(expected_outs))])

    def train(self, dataset, epochs, learning_rate):
        """ Train a whole dataset
        :param dataset:
        :param epochs: Number of iterations to train for
        :param learning_rate:
        :return:
        """
        for i in range(epochs):
            sum_error = 0
            for inputs, expected in dataset:
                sum_error += self.train_one(inputs, expected, learning_rate)
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, learning_rate, sum_error), end="\r")


if __name__ == "__main__":
    from bitwise_op_data import XOR, XNOR
    np.random.seed(31337)

    print("XOR")
    xor_ann = BPNN([2, 2, 2])
    xor_ann.train(XOR, 5000, 0.4)
    print("%s -> %d" % ([0, 0], xor_ann.forward([0.0, 0.0], argmax=True)))
    print("%s -> %d" % ([0, 1], xor_ann.forward([0.0, 1.0], argmax=True)))
    print("%s -> %d" % ([1, 0], xor_ann.forward([1.0, 0.0], argmax=True)))
    print("%s -> %d" % ([1, 1], xor_ann.forward([1.0, 1.0], argmax=True)))

    print("XNOR")
    xnor_ann = BPNN([2, 2, 2])
    xnor_ann.train(XNOR, 5000, 0.4)
    print("%s -> %d" % ([0, 0], xnor_ann.forward([0.0, 0.0], argmax=True)))
    print("%s -> %d" % ([0, 1], xnor_ann.forward([0.0, 1.0], argmax=True)))
    print("%s -> %d" % ([1, 0], xnor_ann.forward([1.0, 0.0], argmax=True)))
    print("%s -> %d" % ([1, 1], xnor_ann.forward([1.0, 1.0], argmax=True)))