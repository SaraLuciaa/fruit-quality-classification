"""
CNN Training Script
-------------------
Trains a Keras CNN model for fruit quality classification.
Usage: python -m src.training.train_cnn [--data_dir data] [--epochs 30] [--output_dir experiments]
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import joblib

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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
    parser = argparse.ArgumentParser(description="Train CNN fruit quality classifier")
    parser.add_argument("--data_dir", type=str, default="dataset_processed",
                        help="Root data directory containing train, val, and test folders")
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

    X = images
    y = np.array(labels)
    print(f"\nLoaded {len(X)} images across {len(np.unique(y))} classes")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y == cls)}")
    return X, y, filenames


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


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = plt.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    plt.title("CNN Confusion Matrix", fontsize=12, fontweight="bold")
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
    plt.savefig(os.path.join(output_dir, "results", "cnn_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    print("="*60)
    print("  TRAINING CNN CLASSIFIER")
    print("="*60)

    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)

    # 1. Load datasets (train, val, and test)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")

    print(f"Loading training data from: {train_dir}")
    X_train, y_train_raw, _ = load_dataset(train_dir)
    print(f"Loading validation data from: {val_dir}")
    X_val, y_val_raw, _ = load_dataset(val_dir)
    print(f"Loading test data from: {test_dir}")
    X_test, y_test_raw, _ = load_dataset(test_dir)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("\nERROR: No images found. Check data directory.")
        return

    # 2. Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    joblib.dump(le, os.path.join(args.output_dir, "checkpoints", "label_encoder_cnn.pkl"))

    # 4. Preprocess images
    def preprocess_batch(images):
        batch = np.array([cv2.resize(img, (224, 224)) for img in images],
                         dtype=np.float32) / 255.0
        return batch

    print("\nPreprocessing image batches...")
    X_train_pp = preprocess_batch(X_train)
    X_val_pp = preprocess_batch(X_val)
    X_test_pp = preprocess_batch(X_test)
    print(f"Train samples: {len(X_train_pp)}, Val samples: {len(X_val_pp)}, Test samples: {len(X_test_pp)}")

    # 5. Data Augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        brightness_range=(0.8, 1.2), horizontal_flip=True, fill_mode="nearest")

    # 6. Build CNN Model
    model = build_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "checkpoints", "cnn_best.keras"),
            monitor="val_accuracy", save_best_only=True, mode="max"),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    # 7. Train Model
    print("\nTraining CNN Model...")
    history = model.fit(
        datagen.flow(X_train_pp, y_train, batch_size=args.batch_size),
        validation_data=(X_val_pp, y_val),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 8. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred_probs = model.predict(X_test_pp, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    test_loss, test_acc = model.evaluate(X_test_pp, y_test, verbose=0)

    accuracy = float(test_acc)
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\nCNN Test Accuracy: {accuracy:.4f}")
    print(f"CNN F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 9. Save final model and results
    model.save(os.path.join(args.output_dir, "checkpoints", "cnn_final.keras"))
    print(f"Saved CNN final model to {args.output_dir}/checkpoints/cnn_final.keras")

    results = {
        "cnn": {
            "accuracy": accuracy,
            "loss": float(test_loss),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
    }

    report_path = os.path.join(args.output_dir, "results", "cnn_metrics_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(cm, args.output_dir)

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
    plt.savefig(os.path.join(args.output_dir, "results", "cnn_training_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
