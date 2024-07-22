import pandas as pd

try:
    data = pd.read_csv('data/training.csv')
    
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.columns[0], axis=1)

    # convert data to float
    data = data.apply(pd.to_numeric)
except Exception as e:
    print("Error:", e)
    exit(1)

def forward_propagation(inputs, weights):
    return [sum([i * w for i, w in zip(inputs, weights)])]

def back_propagation(inputs, weights, error, learning_rate):
    return [w - learning_rate * error * i for i, w in zip(inputs, weights)]

def train(data, learning_rate, epochs):
    weights = [0.1, 0.2, 0.3]
    for _ in range(epochs):
        for i, row in data.iterrows():
            inputs = row[:-1]
            expected = row[-1]
            output = forward_propagation(inputs, weights)
            error = expected - output[0]
            weights = back_propagation(inputs, weights, error, learning_rate)
            print(weights)
    return weights

if __name__ == '__main__':
    learning_rate = 0.01
    epochs = 1000
    weights = train(data, learning_rate, epochs)
    print(weights)