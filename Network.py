import numpy as np


def activation_function(val):
    return max(0, val)


class Network:
    def __init__(self, layers):
        self.inputs = np.zeros((layers[0], 1))
        self.hidden_layers = [np.zeros((layers[i], 1)) for i in range(1, len(layers) - 1)]
        self.outputs = np.zeros((layers[-1], 1))
        self.weights = [np.random.rand(layers[i + 1], layers[i]) for i in range(len(layers) - 1)]
        self.biases = [np.random.rand(layers[i], 1) for i in range(1, len(layers) - 1)]

    def print_network(self):
        print(self.inputs)
        print()
        for layer in self.hidden_layers:
            print(layer)
            print()
        print(self.outputs)
        print()
        for weight in self.weights:
            print(weight)
            print()
        for bias in self.biases:
            print(bias)
            print()

    def set_inputs(self, inputs):
        self.inputs = np.copy(inputs)

    def get_outputs(self):
        return self.outputs

    def forward_propagation(self):
        for i in range(len(self.hidden_layers) + 1):
            A = self.hidden_layers[i] if i < len(self.hidden_layers) else self.outputs
            B = self.hidden_layers[i - 1] if i > 0 else self.inputs
            A[:] = self.weights[i].dot(B) + (self.biases[i] if i < len(self.hidden_layers) else 0)
            for j in range(len(A)):
                A[j] = activation_function(A[j])
