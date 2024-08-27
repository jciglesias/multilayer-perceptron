from neuralnetwork import NeuralNetwork
import pandas as pd
import pickle

interpretation = {
    0: "M",
    1: "B"
}

if __name__=="__main__":
    try:
        data = pd.read_csv('data/test.csv')
        pkl_file = open('model.pkl', 'rb')
    except FileNotFoundError:
        print("File not found")
        exit(1)
    nn:NeuralNetwork = pickle.load(pkl_file)
    for i, row in data.iterrows():
        inputs = row[2:]
        print(f"output: {interpretation[nn.forwardpropagation(inputs)]} - expected: {row.iloc[1]}")