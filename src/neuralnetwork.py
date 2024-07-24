import neuron as nr

class NeuralNetwork:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.first_hlayer = [nr.Neuron(nr.gelu) for _ in range(hidden_size)]
        self.second_hlayer = [nr.Neuron(nr.gelu) for _ in range(hidden_size)]
        self.output_layer = nr.Neuron(nr.softmax)

    def forward(self, inputs):
        first_hlayer_output = [neuron.forward(inputs) for neuron in self.first_hlayer]
        second_hlayer_output = [neuron.forward(first_hlayer_output) for neuron in self.second_hlayer]
        return self.output_layer.forward(second_hlayer_output)
    
    def __str__(self):
        return f'NeuralNetwork({self.input_size}, {self.hidden_size})'
    
    def __repr__(self):
        return str(self)

    def calculate_weights(self, neuron, error, learning_rate, inputs):
            for j, weight in enumerate(neuron.weights):
                gradient = nr.derivative(neuron.linear_transformation(inputs)) * error
                neuron.weights[j] = neuron.weights[j] = weight - learning_rate * gradient

    def backpropagation(self, inputs, expected, learning_rate):
        first_hlayer_output = [neuron.forward(inputs) for neuron in self.first_hlayer]
        second_hlayer_output = [neuron.forward(first_hlayer_output) for neuron in self.second_hlayer]
        output = self.output_layer.forward(second_hlayer_output)
        error = expected - output
        for neuron in self.first_hlayer:
            self.calculate_weights(neuron, error, learning_rate, first_hlayer_output)
        for neuron in self.second_hlayer:
            self.calculate_weights(neuron, error, learning_rate, second_hlayer_output)
        self.calculate_weights(self.output_layer, error, learning_rate, second_hlayer_output)

    def train(self, data, learning_rate, epochs):
        for _ in range(epochs):
            for i, row in data.iterrows():
                inputs = row[:-1]
                expected = row[-1]
                self.backpropagation(inputs, expected, learning_rate)