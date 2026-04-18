import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self):
        self.raw_data_path = "artifacts/raw.csv"
        self.train_data_path = "artifacts/train.csv"
        self.test_data_path = "artifacts/test.csv"
    def initiate_data_ingestion(self):
        df = pd.read_csv("data/diamonds.csv")
        os.makedirs("artifacts", exist_ok=True)
        df.to_csv(self.raw_data_path, index=False)

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

        train_set.to_csv(self.train_data_path, index=False)
        test_set.to_csv(self.test_data_path, index=False)

        print("Data Ingestion Completed")
        return self.train_data_path, self.test_data_path
