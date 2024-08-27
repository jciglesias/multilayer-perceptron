import neuron as nr
import pickle, numpy as np
import matplotlib.pyplot as plt

interpretation = {
    0: "M",
    1: "B"
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

    def train(self, data, val_data, learning_rate, epochs):
        losses = []
        accuracy = []
        val_losses = []
        val_accuracies = []
        for epoch in range(epochs):
            total_error = 0
            total_predictions = 0
            correct_predictions = 0
            for i, row in data.iterrows():
                inputs = row[2:]
                expected = row.iloc[1] == "B"
                self.backpropagation(inputs, expected, learning_rate)
                output = self.forwardpropagation(inputs)
                if (expected == output):
                    correct_predictions += 1
                total_predictions += 1
                total_error += expected - output
            losses.append(total_error/len(data))
            accuracy.append(correct_predictions/total_predictions)
            val_loss, val_accuracy = self.validation(val_data)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {losses[-1]}, val_loss: {val_loss}')
        self.plot_metrics(epochs, losses, val_losses, accuracy, val_accuracies)

    def validation(self, data):
        total_error = 0
        total_predictions = 0
        correct_predictions = 0
        for i, row in data.iterrows():
            inputs = row[2:]
            expected = row.iloc[1] == "B"
            output = self.forwardpropagation(inputs)
            if (expected == output):
                correct_predictions += 1
            total_predictions += 1
            total_error += expected - output
        return total_error/len(data), correct_predictions/total_predictions
    
    def plot_metrics(self, epochs, losses, val_losses, accuracies, val_accuracies):
        # Plot Losses
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(epochs), losses, label='Training Loss', color='blue')
        plt.plot(range(epochs), val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss vs Epochs')
        plt.legend()

        # Plot Accuracies
        plt.subplot(1, 2, 2)
        plt.plot(range(epochs), accuracies, label='Training Accuracy', color='blue')
        plt.plot(range(epochs), val_accuracies, label='Validation Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig('metrics.png')
        plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def find_max(self, output: np.array):
        return np.argmax(output)