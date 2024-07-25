from neuralnetwork import NeuralNetwork
import pandas as pd
import pickle

interpretation = {
    0: "B",
    1: "M"
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
        print(interpretation[nn.forwardpropagation(inputs)], row.iloc[1])