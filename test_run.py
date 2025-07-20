import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import os

def test_model_accuracy():
    # Load trained model
    model_path = "models/model.joblib"
    assert os.path.exists(model_path), "Model file not found!"
    model = joblib.load(model_path)

    # Load dataset
    df = pd.read_csv("data/iris.csv")
    assert not df.empty, "Iris dataset is empty!"

    # Prepare features and target
    X = df.drop(columns=["target", "timestamp"])
    y = df["target"]

    # Encode the categorical 'species' feature
    encoder = OrdinalEncoder()
#    X["species"] = encoder.fit_transform(df[["species"]])

    # Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Assert model is decently accurate
    assert acc >= 0.9, f"Model accuracy too low: {acc:.2f}"
