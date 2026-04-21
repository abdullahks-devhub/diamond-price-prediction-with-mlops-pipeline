import argparse
from src.pipeline.training_pipeline import TrainingPipeline
from upload_model import upload_artifacts

def main():
    parser = argparse.ArgumentParser(description="Diamond Price Prediction Training Pipeline")
    parser.add_argument("--upload", action="store_true", help="Upload artifacts to Hugging Face Hub after training")
    args = parser.parse_args()

    # Run training pipeline
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()

    # Optional upload
    if args.upload:
        print("\n" + "="*40)
        print("Starting Automated Upload to Hugging Face Hub")
        print("="*40)
        upload_artifacts()

if __name__ == "__main__":
    main()