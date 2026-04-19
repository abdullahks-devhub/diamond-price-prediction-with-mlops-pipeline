import os
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        print("Starting Model Training")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=0)
        }
        best_model = None
        best_score = -1
        best_model_name = ""

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)

                mlflow.sklearn.log_model(model, name)
                print(f"{name} -> R: {r2}, RMSE: {rmse}")

                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    best_model = model

        os.makedirs("artifacts", exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"Best model: {best_model_name} with R2: {best_score}")
        return self.model_path
