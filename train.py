import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
mlflow.set_tracking_uri("http://104.154.167.143:8100")
#
mlflow.set_experiment("iris_classification")

df = pd.read_csv("data/iris.csv")
X = df.drop(columns=["target", "timestamp"])
y = df["target"]

encoder = OrdinalEncoder()
#X["species"] = encoder.fit_transform(df[["species"]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    # Save model to DVC-tracked directory
    import joblib
    joblib.dump(model, "models/model.joblib")
