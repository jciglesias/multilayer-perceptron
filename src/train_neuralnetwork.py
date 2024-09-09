from neuralnetwork import NeuralNetwork
import pandas as pd

if __name__=="__main__":
    try:
        data = pd.read_csv('data/training.csv')
        val_data = pd.read_csv('data/validation.csv')
    except FileNotFoundError:
        print("File not found")
        exit(1)
    nn = NeuralNetwork(30, 40, 4)
    nn.train(data, val_data, 0.001, 100)
    print(nn)
    nn.save('model.pkl')