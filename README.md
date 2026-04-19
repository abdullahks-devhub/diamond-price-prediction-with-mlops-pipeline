## Progress Log

### Day 1 - Setup & EDA
- Created project structure
- Downloaded Kaggle diamonds dataset
- Performed initial EDA

### Day 2 - Data Ingestion
- Built data ingestion module
- Split dataset into train/test sets
- Saved raw, train, and test data in artifacts
- Ensured reproducibility with fixed random seed

### Day 3 - Data Transformation
- Built preprocessing pipeline using ColumnTransformer
- Applied StandardScaler to numerical features
- Applied OneHotEncoder to categorical features
- Prevented data leakage (fit on train only)
- Saved preprocessor object for reuse