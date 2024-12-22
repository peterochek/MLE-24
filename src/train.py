import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from artifacts import log_mlflow_experiment


def train_models(input_path, model_output_path):
    df = pd.read_csv(input_path)
    X = df.drop(["Survived"], axis=1)
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model: {name}")
        print(classification_report(y_test, y_pred))
        with open(
            f"{model_output_path}/{name.replace(' ', '_').lower()}.pkl", "wb"
        ) as f:
            import pickle

            pickle.dump(model, f)

        log_mlflow_experiment(name, accuracy, model_output_path)
