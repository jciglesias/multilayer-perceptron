import math

# GELU activation function
def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))

# softmax activation function
def softmax(x):
    return math.exp(x) / sum([math.exp(i) for i in x])

# hidden layer neuron
def hidden_neuron(inputs, weights):
    return gelu(sum([i * w for i, w in zip(inputs, weights)]))

# output layer neuron
def output_neuron(inputs, weights):
    return softmax(sum([i * w for i, w in zip(inputs, weights)]))

# derivative of the GELU activation function
def derivative(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2))) + 0.5 * x * math.exp(-x**2 / 2) / math.sqrt(2 * math.pi)