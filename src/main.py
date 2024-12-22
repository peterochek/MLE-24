from preprocess import preprocess_data
from train import train_models

if __name__ == "__main__":
    preprocess_data("data/titanic.csv", "data/processed.csv")
    train_models("data/processed.csv", "models/")
