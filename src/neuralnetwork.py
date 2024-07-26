import neuron as nr
import pickle, numpy as np
import matplotlib.pyplot as plt

interpretation = {
    0: "B",
    1: "M"
}

class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.first_hlayer = [nr.Neuron(nr.gelu) for _ in range(hidden_size)]
        self.second_hlayer = [nr.Neuron(nr.gelu) for _ in range(hidden_size)]
        self.output_layer = [nr.Neuron(nr.output) for _ in range(2)]

    def forwardpropagation(self, inputs):
        first_hlayer_output = [neuron.forward(inputs) for neuron in self.first_hlayer]
        second_hlayer_output = [neuron.forward(first_hlayer_output) for neuron in self.second_hlayer]
        raw_output = [neuron.forward(second_hlayer_output) for neuron in self.output_layer]
        return self.find_max(nr.softmax(raw_output))
    
    def __str__(self):
        return f'NeuralNetwork({self.input_size}, {self.hidden_size})'
    
    def __repr__(self):
        return str(self)

    def calculate_weights(self, neuron, error, learning_rate, inputs):
            for j, weight in enumerate(neuron.weights):
                gradient = nr.derivative(neuron.linear_transformation(inputs)) * error
                neuron.weights[j] = weight - learning_rate * gradient

    def backpropagation(self, inputs, expected, learning_rate):
        first_hlayer_output = [neuron.forward(inputs) for neuron in self.first_hlayer]
        second_hlayer_output = [neuron.forward(first_hlayer_output) for neuron in self.second_hlayer]
        raw_output = [neuron.forward(second_hlayer_output) for neuron in self.output_layer]
        output = self.find_max(nr.softmax(raw_output))
        error = expected - output
        for neuron in self.first_hlayer:
            self.calculate_weights(neuron, error, learning_rate, inputs)
        for neuron in self.second_hlayer:
            self.calculate_weights(neuron, error, learning_rate, first_hlayer_output)
        for neuron in self.output_layer:
            self.calculate_weights(neuron, error, learning_rate, second_hlayer_output)

    def train(self, data, learning_rate, epochs):
        losses = []
        for epoch in range(epochs):
            total_error = 0
            for i, row in data.iterrows():
                inputs = row[2:]
                expected = row.iloc[1] == "M"
                self.backpropagation(inputs, expected, learning_rate)
                output = self.forwardpropagation(inputs)
                if (expected != output):
                    print(f'Epoch: {epoch} - Expected: {interpretation[expected]} - Output: {interpretation[output]}')
                total_error += expected - output
            losses.append(total_error/len(data))
            # print(f'Epoch {epoch} - Loss: {losses[-1]}')
        self.plot_losses(epochs, losses)
    
    def plot_losses(self, epochs, losses):
        plt.plot(range(epochs), losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def find_max(self, output: np.array):
        return np.argmax(output)