import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

    X = data.drop(columns=["Survived"])
    y = data["Survived"]

    numeric_features = ["Age", "Fare"]
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = ["Sex", "Embarked"]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    processed = pd.concat([pd.DataFrame(X_processed), y], axis=1)

    processed.to_csv(output_path, index=False)
