import src.neuron as nr
import pickle, numpy as np
import matplotlib.pyplot as plt
import concurrent.futures as cf
import time
from pandas import DataFrame
from copy import copy

interpretation = {
    "M": 0,
    "B": 1 
}

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, n_hidden_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers = []
        self.layers.append([nr.Neuron(nr.gelu, input_size) for _ in range(hidden_size)])
        for i in range(1, n_hidden_layers):
            self.layers.append([nr.Neuron(nr.gelu, hidden_size) for _ in range(hidden_size)])
        self.output_layer = [nr.Neuron(nr.output, hidden_size) for _ in range(2)]

    def forwardpropagation(self, inputs):
        output = copy(inputs)
        for layer in self.layers:
            output = [neuron.forward(output) for neuron in layer]
        raw_output = [neuron.forward(output) for neuron in self.output_layer]
        return np.argmax(nr.softmax(raw_output))

    def calculate_weights(self, neuron, error, learning_rate, inputs):
        # if (neuron.diff != 0):
            neuron.diff = 0
            for j, weight in enumerate(neuron.weights):
                gradient = nr.derivative(neuron.linear_transformation(inputs)) * error
                diff = learning_rate * gradient
                neuron.diff += abs(diff)
                neuron.weights[j] = weight - diff
            print(neuron.diff)

    def backpropagation(self, inputs, expected, learning_rate):
        outputs = []
        outputs.append(inputs)
        for layer in self.layers:
            outputs.append([neuron.forward(outputs[-1]) for neuron in layer])
        raw_output = [neuron.forward(outputs[-1]) for neuron in self.output_layer]
        output = np.argmax(nr.softmax(raw_output))
        error = expected - output
        with cf.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(len(self.layers)):
                for neuron in self.layers[i]:
                    futures.append(executor.submit(self.calculate_weights, neuron, error, learning_rate, outputs[i]))
            for neuron in self.output_layer:
                futures.append(executor.submit(self.calculate_weights, neuron, error, learning_rate, outputs[-1]))
            cf.wait(futures)

    def train(self, data: DataFrame, val_data: DataFrame, learning_rate: float, epochs: int):
        losses = []
        accuracy = []
        val_losses = []
        val_accuracies = []
        starting_time = time.time()
        for epoch in range(epochs):
            total_error = 0
            total_predictions = 0
            correct_predictions = 0
            for i, row in data.sample(frac=0.5).iterrows():
                inputs = row[2:]
                expected = interpretation.get(row.iloc[1])
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
            elapsed_time = (time.time() - starting_time) / 60
            yield DataFrame({
                "Epoch": f"{epoch + 1}/{epochs}",
                "Loss": losses[-1],
                "Validation loss": val_loss,
                "Accuracy": accuracy[-1]*100,
                "Validation accuracy": val_accuracy*100,
                "Time": f"{elapsed_time:.2f} minutes",
                "ETA": f"{(elapsed_time * epochs / (epoch + 1)) - elapsed_time:.2f} minutes"
                },index=[epoch])
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
        self.fig = plt.figure(figsize=(12, 5))

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
        # plt.savefig('metrics.png')
        # plt.show()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    def __str__(self):
        return f'NeuralNetwork({self.input_size}, {self.hidden_size}, {len(self.layers)})'
    
    def __repr__(self):
        return str(self)
