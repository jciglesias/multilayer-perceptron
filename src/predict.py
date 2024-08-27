import math
from neuralnetwork import NeuralNetwork
import pandas as pd
import pickle

interpretation = {
    0: "M",
    1: "B"
}

def binary_cross_entropy(output, expected, epsilon=0.001):
    output = max(min(output, 1 - epsilon), epsilon)
    return -expected * math.log(output) - (1 - expected) * math.log(1 - output)

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
    print(f"Binary Cross Entropy: {e}")