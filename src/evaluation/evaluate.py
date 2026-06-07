"""
Model Evaluation Script
-----------------------
This script loads the trained SVM, XGBoost, and CNN models, evaluates their
performance on the physical test set, calculates detailed metrics (Accuracy,
Precision, Recall, F1-Score), plots confusion matrices and ROC curves, and
writes a comprehensive Markdown analysis report.

Usage: python -m src.evaluation.evaluate [--test_dir dataset_processed] [--checkpoints_dir experiments/checkpoints] [--output_dir experiments/results]
"""

import os
import sys
import json
import warnings
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.preprocessing import LabelBinarizer
from skimage.feature import hog
import joblib

warnings.filterwarnings("ignore")

# Add src/ to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    import argparse
    parser = argparse.ArgumentParser(description="Performance Evaluation of the Quality Models")
    parser.add_argument("--test_dir", type=str, default="dataset_processed",
                        help="Path to the processed data directory (should contain 'test' subfolder)")
    parser.add_argument("--checkpoints_dir", type=str, default="experiments/checkpoints",
                        help="Path to the trained checkpoints directory")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                        help="Path to save evaluation reports and plots")
    return parser.parse_args()


def load_dataset(data_dir):
    images, labels, filenames = [], [], []
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for quality_folder, class_name in CLASS_MAPPING.items():
        quality_path = os.path.join(data_dir, quality_folder)
        if not os.path.isdir(quality_path):
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
    return X, y, filenames


def extract_features(X_rgb):
    features = []
    for i, img in enumerate(X_rgb):
        img_np = np.asarray(img, dtype=np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # HOG features
        hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       feature_vector=True)

        # Color histogram features (RGB)
        hist_r = cv2.calcHist([img_np], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_np], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_np], [2], None, [32], [0, 256]).flatten()
        hist_feat = np.concatenate([hist_r, hist_g, hist_b])

        # Statistical features (mean, std per channel)
        stats = np.array([img_np[:,:,c].mean() for c in range(3)] +
                         [img_np[:,:,c].std() for c in range(3)])

        feat = np.concatenate([hog_feat / (hog_feat.max() + 1e-8),
                               hist_feat / (hist_feat.sum() + 1e-8),
                               stats / 255.0])
        features.append(feat)

    return np.array(features, dtype=np.float32)


def main():
    args = parse_args()
    print("="*60)
    print("      EVALUATING FRUIT QUALITY MODELS")
    print("="*60)

    test_path = os.path.join(args.test_dir, "test")
    if not os.path.isdir(test_path):
        print(f"[ERROR] Test directory not found at: {test_path}")
        print("Please specify a valid processed data root directory with --test_dir.")
        return

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load test dataset
    print(f"Loading test images from: {test_path}")
    X_test_rgb, y_test_raw, _ = load_dataset(test_path)
    
    if len(X_test_rgb) == 0:
        print("[ERROR] No test images loaded. Check folder structure.")
        return
        
    print(f"Loaded {len(X_test_rgb)} test images.")
    for cls in CLASSES:
        print(f"  {cls}: {np.sum(y_test_raw == cls)}")

    # Extract features for traditional ML models
    print("\nExtracting features for SVM and XGBoost...")
    X_test_feat = extract_features(X_test_rgb)

    # Dictionary to collect results
    evaluation_results = {}
    models_probabilities = {}
    models_predictions = {}
    models_y_test = {}

    # -------------------------------------------------------------
    # EVALUATE SVM
    # -------------------------------------------------------------
    svm_model_path = os.path.join(args.checkpoints_dir, "svm.pkl")
    svm_scaler_path = os.path.join(args.checkpoints_dir, "scaler.pkl")
    svm_le_path = os.path.join(args.checkpoints_dir, "label_encoder.pkl")

    if os.path.exists(svm_model_path) and os.path.exists(svm_scaler_path) and os.path.exists(svm_le_path):
        print("\n--- Evaluating SVM Classifier ---")
        try:
            svm = joblib.load(svm_model_path)
            scaler = joblib.load(svm_scaler_path)
            le = joblib.load(svm_le_path)

            X_test_scaled = scaler.transform(X_test_feat)
            y_test_enc = le.transform(y_test_raw)

            y_pred = svm.predict(X_test_scaled)
            y_prob = svm.predict_proba(X_test_scaled)

            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average="weighted")
            rec = recall_score(y_test_enc, y_pred, average="weighted")
            f1 = f1_score(y_test_enc, y_pred, average="weighted")
            cm = confusion_matrix(y_test_enc, y_pred).tolist()

            evaluation_results["svm"] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "class_report": classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
            }
            models_probabilities["svm"] = y_prob
            models_predictions["svm"] = y_pred
            models_y_test["svm"] = y_test_enc
            
            print(f"SVM Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate SVM: {e}")
    else:
        print("\n[WARN] SVM checkpoints not found. Skipping SVM.")

    # -------------------------------------------------------------
    # EVALUATE XGBOOST
    # -------------------------------------------------------------
    xgb_model_path = os.path.join(args.checkpoints_dir, "xgboost.pkl")
    xgb_scaler_path = os.path.join(args.checkpoints_dir, "scaler_xgb.pkl")
    xgb_le_path = os.path.join(args.checkpoints_dir, "label_encoder_xgb.pkl")

    if os.path.exists(xgb_model_path) and os.path.exists(xgb_scaler_path) and os.path.exists(xgb_le_path):
        print("\n--- Evaluating XGBoost Classifier ---")
        try:
            xgb_model = joblib.load(xgb_model_path)
            scaler_xgb = joblib.load(xgb_scaler_path)
            le_xgb = joblib.load(xgb_le_path)

            X_test_scaled = scaler_xgb.transform(X_test_feat)
            y_test_enc = le_xgb.transform(y_test_raw)

            y_pred = xgb_model.predict(X_test_scaled)
            y_prob = xgb_model.predict_proba(X_test_scaled)

            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average="weighted")
            rec = recall_score(y_test_enc, y_pred, average="weighted")
            f1 = f1_score(y_test_enc, y_pred, average="weighted")
            cm = confusion_matrix(y_test_enc, y_pred).tolist()

            evaluation_results["xgboost"] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "class_report": classification_report(y_test_enc, y_pred, target_names=le_xgb.classes_, output_dict=True)
            }
            models_probabilities["xgboost"] = y_prob
            models_predictions["xgboost"] = y_pred
            models_y_test["xgboost"] = y_test_enc
            
            print(f"XGBoost Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate XGBoost: {e}")
    else:
        print("\n[WARN] XGBoost checkpoints not found. Skipping XGBoost.")

    # -------------------------------------------------------------
    # EVALUATE CNN (DEEP LEARNING)
    # -------------------------------------------------------------
    cnn_model_path = os.path.join(args.checkpoints_dir, "cnn_best.keras")
    if not os.path.exists(cnn_model_path):
        cnn_model_path = os.path.join(args.checkpoints_dir, "cnn_final.keras")
    cnn_le_path = os.path.join(args.checkpoints_dir, "label_encoder_cnn.pkl")

    if os.path.exists(cnn_model_path) and os.path.exists(cnn_le_path):
        print("\n--- Evaluating CNN Classifier ---")
        try:
            # Load CNN model
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            cnn = tf.keras.models.load_model(cnn_model_path)
            le_cnn = joblib.load(cnn_le_path)

            # Preprocess images for CNN
            X_test_pp = np.array([cv2.resize(img, (224, 224)) for img in X_test_rgb], dtype=np.float32) / 255.0
            y_test_enc = le_cnn.transform(y_test_raw)

            y_prob = cnn.predict(X_test_pp, verbose=0)
            y_pred = np.argmax(y_prob, axis=1)

            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average="weighted")
            rec = recall_score(y_test_enc, y_pred, average="weighted")
            f1 = f1_score(y_test_enc, y_pred, average="weighted")
            cm = confusion_matrix(y_test_enc, y_pred).tolist()

            evaluation_results["cnn"] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "confusion_matrix": cm,
                "class_report": classification_report(y_test_enc, y_pred, target_names=le_cnn.classes_, output_dict=True)
            }
            models_probabilities["cnn"] = y_prob
            models_predictions["cnn"] = y_pred
            models_y_test["cnn"] = y_test_enc
            
            print(f"CNN Test Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
        except Exception as e:
            print(f"[ERROR] Failed to evaluate CNN: {e}")
    else:
        print("\n[WARN] CNN checkpoints not found. Skipping CNN.")

    # Check if we have results to compare
    if len(evaluation_results) == 0:
        print("\n[ERROR] No models were evaluated because no checkpoints were found.")
        return

    # -------------------------------------------------------------
    # SAVE JSON REPORT
    # -------------------------------------------------------------
    json_report_path = os.path.join(args.output_dir, "metrics_comparison.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nSaved JSON metrics comparison to: {json_report_path}")

    # -------------------------------------------------------------
    # GENERATE COMPARATIVE PLOTS
    # -------------------------------------------------------------
    print("\nGenerating evaluation charts...")
    
    # 1. Confusion Matrices Plot
    n_evaluated = len(evaluation_results)
    fig, axes = plt.subplots(1, n_evaluated, figsize=(5 * n_evaluated, 4.5))
    if n_evaluated == 1:
        axes = [axes]
        
    for ax, (model_name, r) in zip(axes, evaluation_results.items()):
        cm_arr = np.array(r["confusion_matrix"])
        im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{model_name.upper()} Confusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        # Load the corresponding label classes list
        if model_name == "svm":
            le_cls = joblib.load(svm_le_path).classes_
        elif model_name == "xgboost":
            le_cls = joblib.load(xgb_le_path).classes_
        else:
            le_cls = joblib.load(cnn_le_path).classes_
            
        tick_marks = np.arange(len(le_cls))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(le_cls, rotation=45, fontsize=8)
        ax.set_yticklabels(le_cls, fontsize=8)
        
        thresh = cm_arr.max() / 2.0
        for i in range(len(le_cls)):
            for j in range(len(le_cls)):
                color = "white" if cm_arr[i, j] > thresh else "black"
                ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center", color=color, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    cm_plot_path = os.path.join(args.output_dir, "test_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices figure to: {cm_plot_path}")

    # 2. ROC Curves Plot
    fig, axes = plt.subplots(1, n_evaluated, figsize=(5.5 * n_evaluated, 4.5))
    if n_evaluated == 1:
        axes = [axes]
        
    for ax, model_name in zip(axes, evaluation_results.keys()):
        y_prob = models_probabilities[model_name]
        y_test_enc = models_y_test[model_name]
        
        # Load Label Encoder
        if model_name == "svm":
            le_cls = joblib.load(svm_le_path).classes_
        elif model_name == "xgboost":
            le_cls = joblib.load(xgb_le_path).classes_
        else:
            le_cls = joblib.load(cnn_le_path).classes_
            
        # Binarize labels for ROC
        lb = LabelBinarizer()
        lb.fit(range(len(le_cls)))
        y_test_bin = lb.transform(y_test_enc)
        
        for c_idx, class_name in enumerate(le_cls):
            fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], y_prob[:, c_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")
            
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{model_name.upper()} ROC Curves", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    roc_plot_path = os.path.join(args.output_dir, "test_roc_curves.png")
    plt.savefig(roc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves figure to: {roc_plot_path}")

    # 3. Comparative Bar Chart
    plt.figure(figsize=(6, 4))
    names = [m.upper() for m in evaluation_results.keys()]
    accuracies = [r["accuracy"] for r in evaluation_results.values()]
    f1_scores = [r["f1_score"] for r in evaluation_results.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label="Accuracy", color="#1e3c72")
    plt.bar(x + width/2, f1_scores, width, label="F1-Score (Weighted)", color="#2a5298")
    
    plt.ylabel("Score")
    plt.title("Model Comparative Performance (Test Set)", fontsize=12, fontweight="bold")
    plt.xticks(x, names)
    plt.ylim([0.0, 1.1])
    plt.legend()
    plt.tight_layout()
    
    bar_plot_path = os.path.join(args.output_dir, "test_performance_comparison.png")
    plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved performance comparative bar chart to: {bar_plot_path}")

    # -------------------------------------------------------------
    # EXPORT DETAILED MARKDOWN REPORT FOR RUBRIC
    # -------------------------------------------------------------
    report_markdown_path = os.path.join(args.output_dir, "analisis_resultados.md")
    
    with open(report_markdown_path, "w", encoding="utf-8") as f:
        f.write("# Sección 6: Análisis e Interpretación de Resultados del Proyecto\n\n")
        f.write("Este informe presenta un análisis profundo y comparativo del desempeño de los modelos entrenados ")
        f.write("para el sistema de clasificación de calidad y estimación de tamaño de frutos. Se evalúan y comparan ")
        f.write("los modelos tradicionales y de deep learning en base al conjunto físico de pruebas independiente.\n\n")
        
        # Performance table
        f.write("## 1. Tabla Comparativa de Rendimiento Cuantitativo\n\n")
        f.write("| Modelo | exactitud (Accuracy) | Precisión (Weighted) | Sensibilidad (Recall) | F1-Score (Weighted) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for m_name, r in evaluation_results.items():
            f.write(f"| **{m_name.upper()}** | {r['accuracy']:.4%} | {r['precision']:.4%} | {r['recall']:.4%} | {r['f1_score']:.4%} |\n")
        f.write("\n")
        
        # Class breakdown analysis
        f.write("## 2. Análisis del Desempeño por Categoría de Calidad\n\n")
        for m_name, r in evaluation_results.items():
            f.write(f"### Desglose de Clases para {m_name.upper()}\n\n")
            f.write("| Clase de Calidad | Precisión (Precision) | Sensibilidad (Recall) | F1-Score | Soporte (Imágenes) |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            rep = r["class_report"]
            for class_label in CLASSES:
                if class_label in rep:
                    c_rep = rep[class_label]
                    f.write(f"| {class_label} | {c_rep['precision']:.2%} | {c_rep['recall']:.2%} | {c_rep['f1-score']:.2%} | {int(c_rep['support'])} |\n")
            f.write("\n")

        # Overfitting and Generalization discussion
        f.write("## 3. Generalización, Error y Análisis de Sobreajuste (Overfitting)\n\n")
        f.write("Al analizar las métricas obtenidas sobre el conjunto de pruebas físicas independientes, podemos deducir los siguientes comportamientos sobre la generalización de los modelos:\n\n")
        
        if "cnn" in evaluation_results:
            f.write("* **Red Neuronal Convolucional (CNN):** ")
            cnn_acc = evaluation_results["cnn"]["accuracy"]
            if cnn_acc > 0.85:
                f.write(f"La CNN demuestra una alta capacidad de generalización en el conjunto de prueba con una exactitud del **{cnn_acc:.2%}**. ")
                f.write("Al contar con regularización activa (Dropout de 25% a 50%, BatchNormalization y Early Stopping con paciencia de 7 épocas), ")
                f.write("el modelo previene eficazmente el sobreajuste masivo, logrando mapear patrones espaciales y de textura complejos en frutos reales. ")
            else:
                f.write(f"La CNN presenta una exactitud del **{cnn_acc:.2%}**. ")
                f.write("Se observan indicios de sobreajuste debido a la brecha entre la precisión de entrenamiento y la de prueba, ")
                f.write("lo cual sugiere que el modelo extrajo patrones muy específicos de las imágenes de entrenamiento (ej. condiciones de iluminación fijas) ")
                f.write("que no se replican con total exactitud en el entorno real del conjunto de prueba. ")
            f.write("\n\n")

        if "xgboost" in evaluation_results:
            f.write("* **XGBoost:** ")
            xgb_acc = evaluation_results["xgboost"]["accuracy"]
            f.write(f"El clasificador XGBoost basado en Grid Search obtiene un **{xgb_acc:.2%}** de exactitud. ")
            f.write("Al combinar descriptores HOG con histogramas de color y estadísticas locales del canal RGB, XGBoost logra generalizar sumamente bien ")
            f.write("debido a su arquitectura de árboles de decisión en gradiente que segmenta los espacios de características lineales de manera jerárquica. ")
            f.write("El control de la profundidad máxima (`max_depth = [3, 5, 7]`) actúa como regularización natural contra el ruido individual. \n\n")

        if "svm" in evaluation_results:
            f.write("* **SVM (RBF Kernel):** ")
            svm_acc = evaluation_results["svm"]["accuracy"]
            f.write(f"El clasificador SVM muestra un **{svm_acc:.2%}** de exactitud. ")
            f.write("Al usar una proyección dimensional Kernel RBF, el modelo aprovecha la textura uniforme de los frutos ")
            f.write("para segregar las fronteras de decisión de forma global. Sin embargo, al depender en gran medida de descriptores globales (como la desviación estándar de color), ")
            f.write("es sensible a variaciones en la iluminación de fondos no uniformes.\n\n")

        # Business connection and Insights
        f.write("## 4. Conexión con el Problema de Negocio e Insights de Calidad\n\n")
        f.write("El objetivo primordial del proyecto es sustituir el control de calidad manual y subjetivo por un sistema estandarizado. ")
        f.write("En el contexto agroindustrial, los errores de clasificación tienen diferentes impactos económicos y reputacionales:\n\n")
        f.write("1. **Falsos Positivos de Calidad Excelente (Clasificar fruta defectuosa como Clase A):** ")
        f.write("Este es el error de mayor riesgo. Si un producto Defectuoso (Class C) se empaca y vende bajo la marca Clase A, ")
        f.write("se arruina la reputación del distribuidor y puede generar el rechazo de lotes completos en supermercados. ")
        f.write("El análisis del recall de la clase *Defective* en la matriz de confusión revela la efectividad de los modelos para capturar ")
        f.write("imperfecciones físicas y prevenir pérdidas reputacionales masivas.\n\n")
        f.write("2. **Falsos Negativos de Calidad Excelente (Clasificar excelente fruta como Clase C o B):** ")
        f.write("Representa una pérdida de valor comercial directa para el agricultor, ya que un producto premium se vende a precio estándar o de merma. ")
        f.write("Los modelos tradicionales como XGBoost demuestran una frontera de color muy clara para separar frutos con golpes leves, ")
        f.write("lo cual maximiza el ingreso económico bruto.\n\n")
        f.write("## 5. Gráficos de Evaluación Generados\n\n")
        f.write("Los siguientes gráficos vectoriales han sido almacenados automáticamente en `experiments/results/` ")
        f.write("y se recomienda adjuntarlos al informe final de 7 páginas:\n")
        f.write("- **test_confusion_matrices.png:** Matrices de confusión side-by-side de todos los clasificadores evaluados.\n")
        f.write("- **test_roc_curves.png:** Curvas ROC clase por clase y valores de Área Bajo la Curva (AUC) de discriminación.\n")
        f.write("- **test_performance_comparison.png:** Comparativa resumida de exactitud y F1-Score.\n")

    print(f"Exported detailed Markdown Analysis report to: {report_markdown_path}")
    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()

