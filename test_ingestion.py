from src.components.data_ingestion import DataIngestion
import pandas as pd

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)