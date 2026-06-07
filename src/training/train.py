"""
Model Training Script
---------------------
Trains 2 ML models (Random Forest, SVM) + 1 CNN for fruit quality classification.
Data expected in: data/
  Good Quality_Fruits/   -> Excellent (Class A)
  Regular Quality_Fruits/ -> Good (Class B)
  Bad Quality_Fruits/    -> Defective (Class C)
Usage: python -m src.training.train [--data_dir data] [--epochs 30]
"""

import os, sys, argparse, json, warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from skimage.feature import hog
from skimage import exposure
import joblib

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.preprocess import preprocess_for_cnn

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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
    parser = argparse.ArgumentParser(description="Train fruit quality classifiers")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--epochs", type=int, default=30,
                        help="CNN training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.15)
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

    X = np.array(images, dtype=object)
    y = np.array(labels)
    print(f"\nLoaded {len(X)} images across {len(np.unique(y))} classes")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y == cls)}")
    return X, y, filenames


def extract_features(X_rgb):
    print("\nExtracting features for ML models...")
    features = []
    for i, img in enumerate(X_rgb):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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


def train_ml_models(X_feat, y, output_dir):
    print("\n" + "="*60)
    print("TRAINING TRADITIONAL ML MODELS")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    joblib.dump(scaler, os.path.join(output_dir, "checkpoints", "scaler.pkl"))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, os.path.join(output_dir, "checkpoints", "label_encoder.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc)

    results = {}

    # Random Forest
    print("\n--- Random Forest ---")
    rf_params = {"n_estimators": [100, 200], "max_depth": [10, 20, None],
                 "min_samples_split": [2, 5]}
    rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
                      rf_params, cv=3, scoring="accuracy", verbose=1)
    rf.fit(X_train, y_train)
    rf_best = rf.best_estimator_
    y_pred = rf_best.predict(X_test)
    results["random_forest"] = {
        "best_params": rf.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"  Best params: {rf.best_params_}")
    print(f"  Test Accuracy: {results['random_forest']['accuracy']:.4f}")
    print(f"  F1 Score: {results['random_forest']['f1_score']:.4f}")
    joblib.dump(rf_best, os.path.join(output_dir, "checkpoints", "random_forest.pkl"))

    # SVM
    print("\n--- SVM ---")
    svm_params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]}
    svm = GridSearchCV(SVC(random_state=RANDOM_SEED, probability=True),
                       svm_params, cv=3, scoring="accuracy", verbose=1)
    svm.fit(X_train, y_train)
    svm_best = svm.best_estimator_
    y_pred = svm_best.predict(X_test)
    results["svm"] = {
        "best_params": svm.best_params_,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"  Best params: {svm.best_params_}")
    print(f"  Test Accuracy: {results['svm']['accuracy']:.4f}")
    print(f"  F1 Score: {results['svm']['f1_score']:.4f}")
    joblib.dump(svm_best, os.path.join(output_dir, "checkpoints", "svm.pkl"))

    return results, y_test, rf_best.predict(X_test), svm_best.predict(X_test)


def build_cnn(input_shape=(224, 224, 3)):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
    return model


def train_cnn(X_rgb, y, output_dir, args):
    print("\n" + "="*60)
    print("TRAINING CNN")
    print("="*60)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, os.path.join(output_dir, "checkpoints",
                                 "label_encoder_cnn.pkl"))

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_rgb, y_enc, test_size=args.test_size + args.val_size,
        random_state=RANDOM_SEED, stratify=y_enc)
    val_ratio = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=args.test_size / (args.test_size + args.val_size),
        random_state=RANDOM_SEED, stratify=y_tmp)

    def preprocess_batch(images):
        batch = np.array([cv2.resize(img, (224, 224)) for img in images],
                         dtype=np.float32) / 255.0
        return batch

    X_train_pp = preprocess_batch(X_train)
    X_val_pp = preprocess_batch(X_val)
    X_test_pp = preprocess_batch(X_test)

    print(f"Train: {len(X_train_pp)}, Val: {len(X_val_pp)}, Test: {len(X_test_pp)}")

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        brightness_range=(0.8, 1.2), horizontal_flip=True, fill_mode="nearest")

    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "checkpoints", "cnn_best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        datagen.flow(X_train_pp, y_train, batch_size=args.batch_size),
        validation_data=(X_val_pp, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    y_pred_probs = model.predict(X_test_pp, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_loss, test_acc = model.evaluate(X_test_pp, y_test, verbose=0)

    cnn_results = {
        "accuracy": float(test_acc),
        "loss": float(test_loss),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\nCNN Test Accuracy: {cnn_results['accuracy']:.4f}")
    print(f"CNN F1 Score: {cnn_results['f1_score']:.4f}")

    model.save(os.path.join(output_dir, "checkpoints", "cnn_final.keras"))
    print(f"CNN model saved to {output_dir}/checkpoints/")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"], label="Train")
    ax1.plot(history.history["val_accuracy"], label="Val")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.plot(history.history["loss"], label="Train")
    ax2.plot(history.history["val_loss"], label="Val")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results", "cnn_training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    return cnn_results, y_test, y_pred


def plot_confusion_matrices(results_all, y_test_ml, y_pred_rf, y_pred_svm,
                            y_test_cnn, y_pred_cnn, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Random Forest", "SVM", "CNN"]
    cf_matrixes = [
        results_all["random_forest"]["confusion_matrix"],
        results_all["svm"]["confusion_matrix"],
        results_all["cnn"]["confusion_matrix"],
    ]
    for ax, title, cm in zip(axes, titles, cf_matrixes):
        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        tick_marks = np.arange(N_CLASSES)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        thresh = cm_arr.max() / 2.0
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                color = "white" if cm_arr[i, j] > thresh else "black"
                ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center",
                        color=color, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "results", "confusion_matrices.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    print("="*60)
    print("  FRUIT QUALITY CLASSIFICATION - TRAINING PIPELINE")
    print("="*60)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    X_rgb, y, filenames = load_dataset(args.data_dir)
    if len(X_rgb) == 0:
        print("\nERROR: No images found. Place your data in the following structure:")
        print("  data/")
        for folder in CLASS_MAPPING:
            print(f"    {folder}/")
            print(f"      Apple_*/")
            print(f"      Banana_*/")
            print(f"      ...")
        return

    # 2. Feature extraction for ML models
    print("\n[2/5] Extracting features...")
    X_feat = extract_features(X_rgb)

    # 3. Train ML models
    print("\n[3/5] Training ML models (Random Forest + SVM)...")
    results_ml, y_test_ml, y_pred_rf, y_pred_svm = train_ml_models(
        X_feat, y, args.output_dir)

    # 4. Train CNN
    print("\n[4/5] Training CNN...")
    results_cnn, y_test_cnn, y_pred_cnn = train_cnn(X_rgb, y, args.output_dir, args)

    # 5. Collect results & save
    print("\n[5/5] Saving results...")
    all_results = {**results_ml, "cnn": results_cnn}

    plot_confusion_matrices(all_results, y_test_ml, y_pred_rf, y_pred_svm,
                            y_test_cnn, y_pred_cnn, args.output_dir)

    report_path = os.path.join(args.output_dir, "results", "metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Text summary
    summary = []
    summary.append("="*60)
    summary.append("  FINAL RESULTS - FRUIT QUALITY CLASSIFICATION")
    summary.append("="*60)
    summary.append("")
    summary.append(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    summary.append("-"*68)
    for model_name in ["random_forest", "svm", "cnn"]:
        r = all_results[model_name]
        summary.append(f"{model_name:<20} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                       f"{r['recall']:<12.4f} {r['f1_score']:<12.4f}")
    summary.append("-"*68)
    summary.append("")
    summary.append(f"Models saved in: {args.output_dir}/checkpoints/")
    summary.append(f"Confusion matrices: {args.output_dir}/results/confusion_matrices.png")
    summary.append(f"Full report: {report_path}")
    summary.append("")

    summary_text = "\n".join(summary)
    print(summary_text)

    with open(os.path.join(args.output_dir, "results", "summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
