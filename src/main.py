from src.preprocess import preprocess_data
from src.train import train_models

if __name__ == "__main__":
    preprocess_data("data/titanic.csv", "data/processed.csv")
    train_models("data/processed.csv", "models/")
