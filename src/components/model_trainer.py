import os
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        print("Starting Model Training")
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        with mlflow.start_run():
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(model, "model")

        os.makedirs("artifacts", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(model, f)
        print("Model training Completed")
        return self.model_path
