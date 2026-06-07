"""
Fruit Quality Classification - Pipeline Orchestrator
---------------------------------------------------
Runs the entire pipeline sequentially:
1. Preprocessing and data splitting (optional)
2. SVM Classifier Training
3. XGBoost Classifier Training
4. CNN Classifier Training

Usage:
  python run_pipeline.py                  # Runs everything
  python run_pipeline.py --skip_preprocess # Runs only the model training steps
"""

import os
import sys
import subprocess
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate fruit classification pipeline")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip the data preprocessing step and start training directly")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs to train the CNN")
    parser.add_argument("--data_dir", type=str, default="dataset_processed",
                        help="Path to the processed data root directory (containing train, val, test splits)")
    return parser.parse_args()


def run_command(cmd_args):
    print(f"\n[EXEC] Running command: {' '.join(cmd_args)}")
    try:
        result = subprocess.run(cmd_args, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    args = parse_args()

    print("=" * 60)
    print("        FRUIT QUALITY CLASSIFICATION PIPELINE")
    print("=" * 60)

    # 1. Preprocessing
    if not args.skip_preprocess:
        print("\n>>> Phase 1: Preprocessing & Data Preparation")
        run_command([sys.executable, "-m", "src.data.preprocess"])
    else:
        print("\n>>> Phase 1: Preprocessing skipped by user flag.")

    # Validate that the processed data directory exists
    if not os.path.isdir(args.data_dir):
        print(f"\n[ERROR] Processed data directory '{args.data_dir}' not found!")
        print("Please make sure preprocessing has run successfully first or remove the --skip_preprocess flag.")
        sys.exit(1)

    # 2. Train SVM
    print("\n>>> Phase 2: Training SVM Classifier")
    run_command([
        sys.executable, "-m", "src.training.train_svm",
        "--data_dir", args.data_dir
    ])

    # 3. Train XGBoost
    print("\n>>> Phase 3: Training XGBoost Classifier")
    run_command([
        sys.executable, "-m", "src.training.train_xgboost",
        "--data_dir", args.data_dir
    ])

    # 4. Train CNN
    print("\n>>> Phase 4: Training CNN (Deep Learning)")
    run_command([
        sys.executable, "-m", "src.training.train_cnn",
        "--data_dir", args.data_dir,
        "--epochs", str(args.epochs)
    ])

    print("\n" + "=" * 60)
    print("   ALL PHASES COMPLETED SUCCESSFULLY! READY FOR STREAMLIT APP")
    print("=" * 60)
    print("To launch the Streamlit app, run:")
    print("  streamlit run src/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
