import os
import pickle
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from src.logger import logger
from src.config import MODEL_PATH, MLFLOW_EXPERIMENT_NAME, ARTIFACTS_DIR

class ModelTrainer:
    def __init__(self):
        self.model_path = MODEL_PATH

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return r2, rmse

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        logger.info("Starting model training")
        logger.info(f"X_train shape: {X_train.shape}")

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }

        best_model = None
        best_score = -1
        best_model_name = ""
        results = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                r2, rmse = self.evaluate_model(model, X_test, y_test)

                mlflow.log_param("model_name", name)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(model, name)

                results[name] = {"r2": r2, "rmse": rmse}
                logger.info(f"{name} -> R²: {r2:.4f}, RMSE: {rmse:.2f}")

                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    best_model = model

        logger.info(f"\n{'='*40}")
        logger.info("Model Comparison:")
        for name, scores in results.items():
            logger.info(f"  {name}: R²={scores['r2']:.4f}, RMSE={scores['rmse']:.2f}")
        logger.info(f"Best: {best_model_name} (R²={best_score:.4f})")
        logger.info(f"{'='*40}")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(best_model, f)

        logger.info(f"Best model saved to {self.model_path}")
        return self.model_path