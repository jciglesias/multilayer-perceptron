import pandas as pd
import numpy as np

def split_data(file_path):
    data = pd.read_csv(file_path, header=None)
    
    # Split the data into 40% training 40% validation and 20% testing
    train, validate, test = np.split(data.sample(frac=1), [int(.4*len(data)), int(.6*len(data))])
    
    # Save the data into csv files
    train.to_csv("../data/train.csv", index=False, header=False)
    validate.to_csv("../data/validate.csv", index=False, header=False)
    test.to_csv("../data/test.csv", index=False, header=False)

def read_data_as_numpy(file_path):
    data = pd.read_csv(file_path, header=None)
    return data.to_numpy()

if __name__ == '__main__':
    wrong_path = True
    while wrong_path:
        try:
            file_name = input("Enter the file name from data directory: ")
            split_data(f"../data/{file_name}")
            wrong_path = False
        except Exception as e:
            print(e)
            wrong_path = True