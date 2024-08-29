from neuralnetwork import NeuralNetwork
import pandas as pd
import pickle, math

interpretation = {
    0: "M",
    1: "B"
}

def binary_cross_entropy(y, y_hat, epsilon=0.001):
    y = max(epsilon, min(1 - epsilon, y))
    return -y_hat * math.log(y) - (1 - y_hat) * math.log(1 - y)

if __name__=="__main__":
    try:
        data = pd.read_csv('data/test.csv')
        pkl_file = open('model.pkl', 'rb')
    except FileNotFoundError:
        print("File not found")
        exit(1)
    nn:NeuralNetwork = pickle.load(pkl_file)
    e = 0
    for i, row in data.iterrows():
        inputs = row[2:]
        e += binary_cross_entropy(nn.forwardpropagation(inputs), row.iloc[1] == "B")
        print(f"output: {interpretation[nn.forwardpropagation(inputs)]} - expected: {row.iloc[1]}")
    e /= len(data)
    print(f"Binary cross entropy: {e}")