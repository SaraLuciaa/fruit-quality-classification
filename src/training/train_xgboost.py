"""
XGBoost Training Script
----------------------
Trains an XGBoost model for fruit quality classification.
Usage: python -m src.training.train_xgboost [--data_dir data] [--output_dir experiments]
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from skimage.feature import hog
import joblib
import xgboost as xgb

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CLASS_MAPPING = {
    "Good Quality_Fruits": "Excellent Quality (Class A)",
    "Regular Quality_Fruits": "Good Quality (Class B)",
    "Bad Quality_Fruits": "Defective Quality (Class C)",
}
CLASS_SHORT = {
    "Excellent Quality (Class A)": "Excellent",
    "Good Quality (Class B)": "Good",
    "Defective Quality (Class C)": "Defective",
}
CLASSES = list(CLASS_SHORT.values())
N_CLASSES = len(CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost fruit quality classifier")
    parser.add_argument("--data_dir", type=str, default="dataset_processed",
                        help="Root data directory containing train and test folders")
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def load_dataset(data_dir):
    images, labels, filenames = [], [], []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for quality_folder, class_name in CLASS_MAPPING.items():
        quality_path = os.path.join(data_dir, quality_folder)
        if not os.path.isdir(quality_path):
            print(f"  [WARN] Folder not found: {quality_path}")
            continue

        fruit_folders = [d for d in os.listdir(quality_path)
                         if os.path.isdir(os.path.join(quality_path, d))]
        for fruit_folder in fruit_folders:
            fruit_path = os.path.join(quality_path, fruit_folder)
            for fname in os.listdir(fruit_path):
                if fname.lower().endswith(valid_ext):
                    path = os.path.join(fruit_path, fname)
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    labels.append(CLASS_SHORT[class_name])
                    filenames.append(path)

    X = images
    y = np.array(labels)
    print(f"\nLoaded {len(X)} images across {len(np.unique(y))} classes")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y == cls)}")
    return X, y, filenames


def extract_features(X_rgb):
    print("\nExtracting features for XGBoost...")
    features = []
    for i, img in enumerate(X_rgb):
        img_np = np.asarray(img, dtype=np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # HOG features
        hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       feature_vector=True)

        # Color histogram features (RGB)
        hist_r = cv2.calcHist([img], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img], [2], None, [32], [0, 256]).flatten()
        hist_feat = np.concatenate([hist_r, hist_g, hist_b])

        # Statistical features (mean, std per channel)
        stats = np.array([img[:,:,c].mean() for c in range(3)] +
                         [img[:,:,c].std() for c in range(3)])

        feat = np.concatenate([hog_feat / (hog_feat.max() + 1e-8),
                               hist_feat / (hist_feat.sum() + 1e-8),
                               stats / 255.0])
        features.append(feat)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(X_rgb)}")

    return np.array(features, dtype=np.float32)


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = plt.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    plt.title("XGBoost Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    tick_marks = np.arange(N_CLASSES)
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)
    thresh = cm_arr.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if cm_arr[i, j] > thresh else "black"
            plt.text(j, i, int(cm_arr[i, j]), ha="center", va="center",
                     color=color, fontsize=10)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results", "xgboost_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    print("="*60)
    print("  TRAINING XGBOOST CLASSIFIER")
    print("="*60)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    # 1. Load datasets (train and test)
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    print(f"Loading training data from: {train_dir}")
    X_train_rgb, y_train_raw, _ = load_dataset(train_dir)
    print(f"Loading test data from: {test_dir}")
    X_test_rgb, y_test_raw, _ = load_dataset(test_dir)

    if len(X_train_rgb) == 0 or len(X_test_rgb) == 0:
        print("\nERROR: No images found. Check data directory.")
        return

    # 2. Extract features
    X_train_feat = extract_features(X_train_rgb)
    X_test_feat = extract_features(X_test_rgb)

    # 3. Scale features and encode labels
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_feat)
    X_test = scaler.transform(X_test_feat)
    joblib.dump(scaler, os.path.join(args.output_dir, "checkpoints", "scaler_xgb.pkl"))

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    joblib.dump(le, os.path.join(args.output_dir, "checkpoints", "label_encoder_xgb.pkl"))

    # 5. Train XGBoost model with GridSearchCV
    print("\nTraining XGBoost with Hyperparameter Tuning...")
    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2]
    }
    
    xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=3, scoring="accuracy", verbose=1)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    # 6. Evaluate model
    y_pred = xgb_best.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nBest Parameters: {xgb_grid.best_params_}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Save model and results
    joblib.dump(xgb_best, os.path.join(args.output_dir, "checkpoints", "xgboost.pkl"))
    print(f"Saved XGBoost model to {args.output_dir}/checkpoints/xgboost.pkl")

    results = {
        "xgboost": {
            "best_params": xgb_grid.best_params_,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
    }

    report_path = os.path.join(args.output_dir, "results", "xgboost_metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(cm, args.output_dir)
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
