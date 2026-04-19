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

### Day 4 - Model Training & Tracking
- Implemented multiple models (Linear Regression, Random Forest)
- Compared model performance
- Selected best model based on R² score
- Integrated MLflow for experiment tracking