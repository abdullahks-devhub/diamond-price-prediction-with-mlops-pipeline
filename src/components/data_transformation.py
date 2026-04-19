import os
import sys
import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def get_preprocessor(self):
        numerical_cols = ["carat", "depth", "table", "x", "y", "z"]
        categorical_cols = ["cut", "color", "clarity"]

        num_pipeline = Pipeline([
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols)
        ])

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        print("Starting data transformation")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        target_column = "price"
        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        preprocessor = self.get_preprocessor()
        X_train_trf = preprocessor.fit_transform(X_train)
        X_test_trf = preprocessor.transform(X_test)

        os.makedirs("artifacts", exist_ok=True)

        with open(self.preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)

        print("Data transformation completed")
        return (X_train_trf, y_train, X_test_trf, y_test)
