from neuralnetwork import NeuralNetwork
import pandas as pd

if __name__=="__main__":
    try:
        data = pd.read_csv('data/training.csv')
    except FileNotFoundError:
        print("File not found")
        exit(1)
    nn = NeuralNetwork(30, 30)
    nn.train(data, 0.01, 100)
    print(nn)
    nn.save('model.pkl')