import os
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logger
from src.config import (PREPROCESSOR_PATH, TARGET_COLUMN,
                         NUMERICAL_COLS, CATEGORICAL_COLS, ARTIFACTS_DIR)

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = PREPROCESSOR_PATH

    def get_preprocessor(self):
        num_pipeline = Pipeline([("scaler", StandardScaler())])
        cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, NUMERICAL_COLS),
            ("cat", cat_pipeline, CATEGORICAL_COLS)
        ])
        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        logger.info("Starting data transformation")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]
            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            preprocessor = self.get_preprocessor()
            X_train_trf = preprocessor.fit_transform(X_train)
            X_test_trf = preprocessor.transform(X_test)

            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logger.info(f"Preprocessor saved to {self.preprocessor_path}")
            logger.info("Data transformation completed")
            return X_train_trf, X_test_trf, y_train, y_test

        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise e