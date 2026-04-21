import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.config import (RAW_DATA_PATH, TRAIN_DATA_PATH,
                         TEST_DATA_PATH, RAW_DATA_SOURCE,
                         TEST_SIZE, RANDOM_STATE, ARTIFACTS_DIR)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = RAW_DATA_PATH
        self.train_data_path = TRAIN_DATA_PATH
        self.test_data_path = TEST_DATA_PATH

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion")
        try:
            df = pd.read_csv(RAW_DATA_SOURCE)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.raw_data_path}")

            train_set, test_set = train_test_split(
                df, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            train_set.to_csv(self.train_data_path, index=False)
            test_set.to_csv(self.test_data_path, index=False)

            logger.info(f"Train size: {len(train_set)}, Test size: {len(test_set)}")
            logger.info("Data ingestion completed")
            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise e