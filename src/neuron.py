import math, random, numpy as np

class Neuron:
    def __init__(self, activation):
        self.activation = activation
        self.weights = [random.random() for _ in range(30)]
        self.bias = random.random()

    def linear_transformation(self, inputs):
        return sum([i * w for i, w in zip(inputs, self.weights)]) + self.bias

    def forward(self, inputs):
        return self.activation(self.linear_transformation(inputs))

    def __str__(self):
        return f'Neuron({self.activation.__name__}, {self.weights}, {self.bias})'

    def __repr__(self):
        return str(self)

def output(x):
    return x

# hidden layer neuron gelu, output layer neuron softmax
# GELU activation function
def gelu(x):
    try:
        return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))
    except OverflowError:
        return 0    

# softmax activation function
def softmax(x):
    try:
        return np.exp(x) / np.sum(np.exp(x))
    except Exception:
        return np.zeros(len(x))

# derivative of the GELU activation function
def derivative(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2))) + 0.5 * x * math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)
