from neuron import hidden_neuron, output_neuron, derivative
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Initialize weights
        self.hidden_weights1 = [[random.random() for _ in range(input_size)] for _ in range(hidden_size1)]
        self.hidden_weights2 = [[random.random() for _ in range(hidden_size1)] for _ in range(hidden_size2)]
        self.output_weights = [[random.random() for _ in range(hidden_size2)] for _ in range(output_size)]


        self.learning_rate = learning_rate

    def forward(self, inputs):
        # Calculate first hidden layer outputs
        hidden_outputs1 = [hidden_neuron(inputs, weights) for weights in self.hidden_weights1]

        # Calculate second hidden layer outputs
        hidden_outputs2 = [hidden_neuron(hidden_outputs1, weights) for weights in self.hidden_weights2]

        # Calculate output layer outputs
        return [output_neuron(hidden_outputs2, weights) for weights in self.output_weights]
    
    def calculate_output_error(self, target, output):
        return target - output
    
    def backpropagate(self, inputs, hidden_outputs1, hidden_outputs2, output_error):
        # Calculate output layer gradients
        output_gradients = [derivative(output) * error for output, error in zip(self.output_weights, output_error)]

        # Update output layer weights
        for i in range(self.output_size):
            for j in range(self.hidden_size2):
                self.output_weights[i][j] += self.learning_rate * output_gradients[i] * hidden_outputs2[j]

        # Calculate second hidden layer error
        hidden_error2 = [sum([self.output_weights[i][j] * output_gradients[i] for i in range(self.output_size)]) for j in range(self.hidden_size2)]

        # Calculate second hidden layer gradients
        hidden_gradients2 = [derivative(output) * error for output, error in zip(hidden_outputs2, hidden_error2)]

        # Update second hidden layer weights
        for i in range(self.hidden_size2):
            for j in range(self.hidden_size1):
                self.hidden_weights2[i][j] += self.learning_rate * hidden_gradients2[i] * hidden_outputs1[j]

        # ... repeat for first hidden layer ...

    def train(self, inputs, target):
        # Forward pass
        hidden_outputs1, hidden_outputs2, outputs = self.forward(inputs)

        # Calculate output error
        output_error = self.calculate_output_error(target, outputs)

        # Backpropagate error and update weights
        self.backpropagate(inputs, hidden_outputs1, hidden_outputs2, output_error)
