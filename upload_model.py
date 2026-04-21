from huggingface_hub import HfApi, upload_file

REPO_ID = "AbdullahKS-Devhub/diamond-price-model"

api = HfApi()
api.create_repo(repo_id=REPO_ID, exist_ok=True)

upload_file(
    path_or_fileobj="artifacts/preprocessor.pkl",
    path_in_repo="preprocessor.pkl",
    repo_id=REPO_ID
)

print("Upload successful")