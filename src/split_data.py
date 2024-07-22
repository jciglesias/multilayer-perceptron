import pandas as pd

if __name__=="__main__":
    try:
        data = pd.read_csv("data/data.csv")

        training = data.sample(frac=0.4).reset_index(drop=True)
        validation = data.sample(frac=0.4).reset_index(drop=True)
        test = data.sample(frac=0.2).reset_index(drop=True)

        training.to_csv("data/training.csv", index=False)
        validation.to_csv("data/validation.csv", index=False)
        test.to_csv("data/test.csv", index=False)
    except Exception as e:
        print(e)