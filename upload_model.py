import os
from huggingface_hub import HfApi, upload_file
from src.config import HF_REPO_ID, MODEL_PATH, PREPROCESSOR_PATH


def upload_artifacts():
    api = HfApi()
    repo_id = HF_REPO_ID

    print(f"Uploading artifacts to {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True)

    # List of artifacts to upload
    artifacts = {
        "model.pkl": MODEL_PATH,
        "preprocessor.pkl": PREPROCESSOR_PATH
    }

    for file_name, local_path in artifacts.items():
        if os.path.exists(local_path):
            print(f"Uploading {file_name}...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=file_name,
                repo_id=repo_id
            )
        else:
            print(f"⚠️ Warning: {local_path} not found. Skipping.")

    print("🎉 All artifacts uploaded successfully!")


if __name__ == "__main__":
    upload_artifacts()