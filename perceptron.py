import numpy as np
import maths


class Perceptron(object):
    def __init__(self, num_inputs, biased=True, activation=maths.sigmoid, derivative=maths.dsigmoid):
        self.num_inputs = num_inputs
        self.biased = biased
        self.activation = activation
        self.derivative = derivative
        self.weights = np.random.rand(num_inputs if not biased else num_inputs + 1)
        self.output = 0
        self.delta = 0

    def transfer(self, inputs):
        return self.weights[:self.num_inputs] @ inputs + self.weights[-1]

    def activate(self, inputs, binarize=False):
        self.output = self.activation(self.transfer(inputs))
        self.delta = self.derivative(self.output)
        return self.output if not binarize else maths.heaviside(self.output)

    def train_one(self, inputs, expected, learning_rate):
        y = self.activate(inputs)
        err = expected - y
        for i in range(self.num_inputs):
            self.weights[i] += learning_rate * err * inputs[i]
        if self.biased:
            self.weights[-1] += learning_rate * err
        return err

    def train(self, dataset, epoch, learning_rate):
        for i in range(epoch):
            for inputs, expected in dataset:
                error = self.train_one(inputs, expected, learning_rate)
                print("epoch=%d, lrate=%f, error=%f" % (epoch, learning_rate, error), end="\r")
                if i > epoch:
                    break


if __name__ == "__main__":
    from bitwise_op_data import OR, AND, NOR, NAND
    np.random.seed(31337)

    print("OR")
    or_neuron = Perceptron(2)
    or_neuron.train(OR, 3000, 0.1)
    print("%s -> %d" % ([0, 0], or_neuron.activate([0, 0], binarize=True)))
    print("%s -> %d" % ([0, 1], or_neuron.activate([0, 1], binarize=True)))
    print("%s -> %d" % ([1, 0], or_neuron.activate([1, 0], binarize=True)))
    print("%s -> %d" % ([1, 1], or_neuron.activate([1, 1], binarize=True)))

    print("NOR")
    nor_neuron = Perceptron(2)
    nor_neuron.train(NOR, 3000, 0.1)
    print("%s -> %d" % ([0, 0], nor_neuron.activate([0, 0], binarize=True)))
    print("%s -> %d" % ([0, 1], nor_neuron.activate([0, 1], binarize=True)))
    print("%s -> %d" % ([1, 0], nor_neuron.activate([1, 0], binarize=True)))
    print("%s -> %d" % ([1, 1], nor_neuron.activate([1, 1], binarize=True)))

    print("AND")
    and_neuron = Perceptron(2)
    and_neuron.train(AND, 3000, 0.1)
    print("%s -> %d" % ([0, 0], and_neuron.activate([0, 0], binarize=True)))
    print("%s -> %d" % ([0, 1], and_neuron.activate([0, 1], binarize=True)))
    print("%s -> %d" % ([1, 0], and_neuron.activate([1, 0], binarize=True)))
    print("%s -> %d" % ([1, 1], and_neuron.activate([1, 1], binarize=True)))

    print("NAND")
    nand_neuron = Perceptron(2)
    nand_neuron.train(NAND, 3000, 0.1)
    print("%s -> %d" % ([0, 0], nand_neuron.activate([0, 0], binarize=True)))
    print("%s -> %d" % ([0, 1], nand_neuron.activate([0, 1], binarize=True)))
    print("%s -> %d" % ([1, 0], nand_neuron.activate([1, 0], binarize=True)))
    print("%s -> %d" % ([1, 1], nand_neuron.activate([1, 1], binarize=True)))
