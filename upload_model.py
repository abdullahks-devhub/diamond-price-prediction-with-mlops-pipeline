from huggingface_hub import HfApi, upload_file

MODEL_PATH = "artifacts/model.pkl"

REPO_ID = "AbdullahKS-Devhub/diamond-price-prediction-with-mlops-pipeline"

api = HfApi()

# Create repo (run once)
api.create_repo(repo_id=REPO_ID, exist_ok=True)

# Upload model file
upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="model.pkl",
    repo_id=REPO_ID
)

print("Model uploaded successfully!")