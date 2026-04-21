import os

# Paths
ARTIFACTS_DIR = "artifacts"
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(ARTIFACTS_DIR, "raw.csv")
TRAIN_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(ARTIFACTS_DIR, "test.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")

# Data
RAW_DATA_SOURCE = os.path.join(DATA_DIR, "diamonds.csv")
TARGET_COLUMN = "price"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model
NUMERICAL_COLS = ["carat", "depth", "table", "x", "y", "z"]
CATEGORICAL_COLS = ["cut", "color", "clarity"]

# MLflow
MLFLOW_EXPERIMENT_NAME = "diamond-price-prediction"

# Hugging Face
HF_REPO_ID = "AbdullahKS-Devhub/diamond-price-model"