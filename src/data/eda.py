import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- PLOT CONFIGURATION FOR HIGH-QUALITY GRAPHICS ---
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_theme(style="whitegrid")

def extract_dataset_metrics(dataset_root):
    """
    Traverses the dataset directory structure to count images per fruit and quality class.
    """
    data = []
    if not os.path.exists(dataset_root):
        print(f"[-] Error: Root directory '{dataset_root}' not found.")
        return pd.DataFrame(data)

    for quality_dir in os.listdir(dataset_root):
        quality_path = os.path.join(dataset_root, quality_dir)
        if os.path.isdir(quality_path):
            # Extract quality category (e.g., 'Good', 'Bad', 'Regular')
            quality_label = quality_dir.split()[0] 
            
            for fruit_dir in os.listdir(quality_path):
                fruit_path = os.path.join(quality_path, fruit_dir)
                if os.path.isdir(fruit_path):
                    # Extract fruit name (e.g., 'Apple', 'Banana')
                    fruit_label = fruit_dir.split('_')[0]
                    
                    # Count only valid image files
                    images = [f for f in os.listdir(fruit_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    
                    data.append({
                        "Fruit": fruit_label,
                        "Quality": quality_label,
                        "Count": len(images)
                    })
    return pd.DataFrame(data)

def extract_image_properties(dataset_root, sample_limit=200):
    """
    Analyzes resolution, format, brightness, contrast, and blurriness across the dataset.
    Uses sampling per class to keep execution fast.
    """
    properties = []
    print(f"[*] Analizando propiedades de las imagenes (limite de muestreo: {sample_limit} por subcarpeta)...")
    
    for root, dirs, files in os.walk(dataset_root):
        # Ignorar carpetas del entorno
        if any(ignored in root.split(os.sep) for ignored in ['.venv', '.git', 'notebooks', 'experiments', 'tests', 'dataset_procesados']):
            continue
            
        parts = root.split(os.sep)
        quality_label = "Unknown"
        fruit_label = "Unknown"
        
        # Extraer etiquetas del nombre de las carpetas
        for part in parts:
            if "Quality_Fruits" in part:
                quality_label = part.split()[0]
            if "_" in part and any(f in part for f in ["Apple", "Banana", "Guava", "Lime", "Orange", "Pomegranate"]):
                fruit_label = part.split('_')[0]

        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not image_files:
            continue
            
        # Muestreo aleatorio si la carpeta es muy grande para agilizar el proceso
        if len(image_files) > sample_limit:
            import random
            random.seed(42)
            image_files = random.sample(image_files, sample_limit)
            
        for file in image_files:
            img_path = os.path.join(root, file)
            try:
                # 1. Dimensiones básicas con PIL (rápido)
                with Image.open(img_path) as img:
                    width, height = img.size
                    img_format = img.format
                
                # 2. Métricas de calidad con OpenCV
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    continue
                
                # Convertir a escala de grises
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Brillo (Media)
                brightness = np.mean(gray)
                
                # Contraste (Desviación estándar)
                contrast = np.std(gray)
                
                # Nitidez/Enfoque (Varianza del Laplaciano)
                # Valores más bajos indican imágenes más borrosas/desenfocadas
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                aspect_ratio = width / height
                
                properties.append({
                    "Fruit": fruit_label,
                    "Quality": quality_label,
                    "Format": img_format,
                    "Width": width,
                    "Height": height,
                    "Aspect_Ratio": aspect_ratio,
                    "Total_Pixels": width * height,
                    "Brightness": brightness,
                    "Contrast": contrast,
                    "Sharpness": sharpness
                })
            except Exception:
                continue  # Ignorar imágenes corruptas
                
    return pd.DataFrame(properties)

def plot_class_distribution(df_metrics, output_dir):
    """
    Dibuja y guarda la distribución de clases por fruta y calidad.
    """
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_metrics, x="Fruit", y="Count", hue="Quality", palette="viridis")
    plt.title("Distribucion de Clases y Analisis de Desbalance por Fruta", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Tipo de Fruta", fontsize=12)
    plt.ylabel("Cantidad de Imagenes (Muestras)", fontsize=12)
    plt.legend(title="Clase de Calidad", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "distribucion_clases.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Graficos de distribucion de clases guardados en: {output_dir}")

def plot_quality_metrics(df_props, output_dir):
    """
    Dibuja y guarda las distribuciones de brillo, contraste y nitidez según la calidad.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    metrics = [
        ("Brightness", "Distribucion de Brillo (Media Gris)", "viridis"),
        ("Contrast", "Distribucion de Contraste (Desv. Est. Gris)", "magma"),
        ("Sharpness", "Distribucion de Nitidez (Varianza Laplaciano)", "rocket")
    ]
    
    for idx, (col, title, palette) in enumerate(metrics):
        if col == "Sharpness":
            sns.boxplot(data=df_props, x="Quality", y=col, ax=axes[idx], hue="Quality", palette="Set2", legend=False)
            axes[idx].set_yscale('log') # Escala logarítmica debido a outliers de nitidez
            axes[idx].set_ylabel(f"{col} (Escala Log)", fontsize=11)
        else:
            sns.kdeplot(data=df_props, x=col, hue="Quality", fill=True, ax=axes[idx], common_norm=False, palette="Set1", alpha=0.4)
            axes[idx].set_ylabel("Densidad", fontsize=11)
            
        axes[idx].set_title(title, fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel(col, fontsize=11)
        axes[idx].grid(True, linestyle="--", alpha=0.6)
        
    plt.suptitle("Comparativa Estadistica de Propiedades de Imagen por Calidad", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "calidad_imagenes_estadisticas.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Graficos estadisticos de calidad guardados en: {output_dir}")

def plot_dimensions_distribution(df_props, output_dir):
    """
    Dibuja y guarda la distribución de resoluciones y relaciones de aspecto.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de dispersión de Ancho vs Alto
    sns.scatterplot(data=df_props, x="Width", y="Height", hue="Fruit", style="Quality", ax=axes[0], alpha=0.7, palette="tab10")
    axes[0].set_title("Mapa de Resoluciones (Ancho vs Alto)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Ancho (Pixeles)", fontsize=11)
    axes[0].set_ylabel("Alto (Pixeles)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Distribución de Relaciones de Aspecto
    sns.boxplot(data=df_props, x="Fruit", y="Aspect_Ratio", hue="Quality", ax=axes[1], palette="muted")
    axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label="Cuadrado (1:1)")
    axes[1].set_title("Distribucion de Relacion de Aspecto (Ancho/Alto)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Fruta", fontsize=11)
    axes[1].set_ylabel("Aspect Ratio", fontsize=11)
    axes[1].legend(title="Calidad")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.suptitle("Dimensiones y Relacion de Aspecto en el Dataset", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "dimensiones_variabilidad.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Graficos de dimensiones y proporciones guardados en: {output_dir}")

def find_representative_images(dataset_root):
    """
    Encuentra automáticamente una imagen 'Good' y una 'Bad' para el perfil cromático.
    """
    good_img = None
    bad_img = None
    
    for root, dirs, files in os.walk(dataset_root):
        if any(ignored in root.split(os.sep) for ignored in ['.venv', '.git', 'notebooks', 'experiments', 'tests', 'dataset_procesados']):
            continue
        parts = root.split(os.sep)
        
        quality = None
        for part in parts:
            if "Quality_Fruits" in part:
                quality = part.split()[0]
                
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if image_files:
            img_path = os.path.join(root, image_files[0])
            if quality == "Good" and good_img is None:
                good_img = img_path
            elif quality == "Bad" and bad_img is None:
                bad_img = img_path
                
        if good_img and bad_img:
            break
            
    return good_img, bad_img

def plot_automatic_color_comparison(dataset_root, output_dir):
    """
    Dibuja y guarda el perfil cromático comparativo RGB de dos muestras automáticas.
    """
    good_img_path, bad_img_path = find_representative_images(dataset_root)
    
    if not good_img_path or not bad_img_path:
        print("[-] Advertencia: No se encontraron imagenes representativas de Good y Bad para la comparativa cromatica.")
        return
        
    print(f"[*] Generando analisis cromatico utilizando:\n   - Good: {good_img_path}\n   - Bad: {bad_img_path}")
    
    img_good = cv2.imread(good_img_path)
    img_bad = cv2.imread(bad_img_path)
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    colors = ('b', 'g', 'r') # OpenCV BGR format
    
    # Render Good image (BGR a RGB para matplotlib)
    ax[0, 0].imshow(cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title(f"Muestra: Calidad BUENA ({os.path.basename(good_img_path)})", fontsize=11, fontweight='bold')
    ax[0, 0].axis('off')
    
    # Render Bad image
    ax[1, 0].imshow(cv2.cvtColor(img_bad, cv2.COLOR_BGR2RGB))
    ax[1, 0].set_title(f"Muestra: Calidad MALA ({os.path.basename(bad_img_path)})", fontsize=11, fontweight='bold')
    ax[1, 0].axis('off')
    
    # Plot RGB Histograms
    for i, col in enumerate(colors):
        hist_good = cv2.calcHist([img_good], [i], None, [256], [0, 256])
        hist_bad = cv2.calcHist([img_bad], [i], None, [256], [0, 256])
        
        plot_color = 'blue' if col == 'b' else 'green' if col == 'g' else 'red'
        ax[0, 1].plot(hist_good, color=plot_color, linewidth=2, label=plot_color.capitalize())
        ax[1, 1].plot(hist_bad, color=plot_color, linewidth=2, label=plot_color.capitalize())
        
    ax[0, 1].set_title("Perfil Cromatico (Histograma RGB) - Buena", fontsize=11, fontweight='bold')
    ax[1, 1].set_title("Perfil Cromatico (Histograma RGB) - Mala", fontsize=11, fontweight='bold')
    
    for a in [ax[0, 1], ax[1, 1]]:
        a.set_xlim([0, 256])
        a.set_xlabel("Intensidad de Pixel", fontsize=10)
        a.set_ylabel("Frecuencia", fontsize=10)
        a.grid(True, linestyle="--", alpha=0.5)
        
    plt.suptitle("Analisis Cromatico Comparativo y Variabilidad de Color", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "perfil_variabilidad_color.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Grafico comparativo de perfil cromático guardado en: {output_dir}")

def generate_eda_markdown_report(df_metrics, df_props, output_dir):
    """
    Generates a markdown report containing the raw data metrics, saving it in output_dir.
    """
    report_path = os.path.join(output_dir, "eda_report.md")
    
    # Calculate imbalance
    total_images = df_metrics["Count"].sum()
    df_imbalance = df_metrics.groupby("Quality")["Count"].sum().reset_index()
    df_imbalance["Percentage"] = (df_imbalance["Count"] / total_images) * 100
    
    # Format tables and text
    metrics_str = df_metrics.to_string(index=False)
    desc_stats_str = df_props[['Width', 'Height', 'Brightness', 'Contrast', 'Sharpness']].describe().to_string()
    format_counts_str = df_props['Format'].value_counts().to_string()
    
    imbalance_bullets = ""
    for _, row in df_imbalance.iterrows():
        imbalance_bullets += f"- **{row['Quality']}**: {row['Count']} imagenes ({row['Percentage']:.2f}%)\n"
        
    good_img, bad_img = find_representative_images("dataset")
    
    report_content = f"""# Reporte de Analisis Exploratorio de Datos (EDA) - Datos y Metricas

Este reporte fue generado de manera automatica por el script de analisis exploratorio de datos y contiene los datos puros recopilados.

---

## 1. Estructura del Dataset y Distribucion de Clases

### Distribucion Detallada por Subcarpeta:
```
{metrics_str}
```

### Analisis del Desbalance por Clase de Calidad:
{imbalance_bullets}

---

## 2. Propiedades Fisicas de las Imagenes y Metricas de Calidad
Muestra aleatoria representativa analizada de {len(df_props)} imagenes.

### Estadisticas Descriptivas de las Imagenes:
```
{desc_stats_str}
```

### Formatos de Archivo Encontrados:
```
{format_counts_str}
```

---

## 3. Analisis Cromatico (Perfiles RGB de Referencia)
- **Ruta de imagen de muestra Buena**: `{good_img}`
- **Ruta de imagen de muestra Mala**: `{bad_img}`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Reporte de datos escrito con exito en: {report_path}")

def plot_size_and_outlier_analysis(df_props, output_dir):
    """
    Plots size distribution (Total Pixels) grouped by Fruit type to define thresholds 
    for Small, Medium, and Large classes, and highlights resolution outliers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Size distribution by fruit type (Crucial for size estimation task)
    sns.boxplot(data=df_props, x="Fruit", y="Total_Pixels", hue="Quality", ax=axes[0], palette="Spectral")
    axes[0].set_yscale('log') # Log scale because resolution variance might be massive
    axes[0].set_title("Fruit Size Distribution (Total Pixels) for Threshold Definition", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Fruit Type", fontsize=11)
    axes[0].set_ylabel("Total Pixels (Log Scale)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Aspect Ratio Outlier Detection (Identifying corrupted crops or extreme framing)
    sns.stripplot(data=df_props, x="Fruit", y="Aspect_Ratio", hue="Quality", ax=axes[1], dodge=True, alpha=0.5, palette="coolwarm")
    axes[1].axhline(1.0, color='black', linestyle='--', alpha=0.6, label="Perfect Square (1:1)")
    axes[1].axhline(0.5, color='red', linestyle=':', alpha=0.6, label="Extreme Horizontal/Vertical Bounds")
    axes[1].axhline(2.0, color='red', linestyle=':')
    axes[1].set_title("Aspect Ratio Outlier & Aspect Consistency Detection", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Fruit Type", fontsize=11)
    axes[1].set_ylabel("Aspect Ratio (Width / Height)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.suptitle("Dataset Size Profiling & Anomaly Detection (EDA Extensions)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "size_and_outlier_analysis.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Advanced size and anomaly plots saved as vector PNG in: {output_dir}")


def plot_hsv_segmentation_feasibility(good_img_path, output_dir):
    """
    Analyzes the image in HSV space to evaluate color segmentation feasibility (Hue/Saturation profiles).
    """
    if not good_img_path or not os.path.exists(good_img_path):
        return
        
    img = cv2.imread(good_img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Hue Histogram (Color family definition)
    axes[0].hist(h.ravel(), bins=180, color='purple', alpha=0.7, rwidth=0.9)
    axes[0].set_title("Hue (Color Pureness / Range) Channel Profile", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Hue Value (0-180)", fontsize=10)
    axes[0].set_ylabel("Pixel Frequency", fontsize=10)
    
    # Plot Saturation Histogram (Color intensity)
    axes[1].hist(s.ravel(), bins=256, color='orange', alpha=0.7, rwidth=0.9)
    axes[1].set_title("Saturation (Color Intensity) Profile", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Saturation Value (0-256)", fontsize=10)
    
    # Scatter plot of Hue vs Saturation to check cluster boundaries
    # Sample pixels to avoid plotting millions of points
    h_sample = h.ravel()[::100]
    s_sample = s.ravel()[::100]
    axes[2].scatter(h_sample, s_sample, c=h_sample, cmap='hsv', alpha=0.3, edgecolors='none', s=5)
    axes[2].set_title("Color Segmentation Clusters (Hue vs Saturation)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Hue", fontsize=10)
    axes[2].set_ylabel("Saturation", fontsize=10)
    
    plt.suptitle(f"HSV Color Space Feasibility Analysis ({os.path.basename(good_img_path)})", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    png_path = os.path.join(output_dir, "hsv_segmentation_profile.png")
    plt.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] HSV background segmentability profiling saved as: {output_dir}")


def plot_interclass_discriminant_matrix(df_props, output_dir):
    """
    Plots cross-metric relationships to evaluate if extracted features 
    (Brightness, Contrast, Sharpness) can linearly separate quality categories.
    """
    # Filter out unknown data to make the feature matrix clean
    df_filtered = df_props[df_props["Quality"] != "Unknown"]
    
    plt.figure(figsize=(10, 8))
    # Pairplot to evaluate linear boundaries for Traditional ML models (SVM / Random Forest)
    g = sns.pairplot(data=df_filtered, vars=["Brightness", "Contrast", "Sharpness"], 
                     hue="Quality", palette="Set1", diag_kind="kde", plot_kws={'alpha': 0.6})
    g.fig.suptitle("Feature Discriminant Matrix (Inter-Class Separation Analysis)", fontsize=14, fontweight='bold', y=1.02)
    
    png_path = os.path.join(output_dir, "feature_discriminant_matrix.png")
    g.savefig(png_path, format="png", bbox_inches='tight')
    plt.close()
    print(f"[+] Feature correlation matrix saved as: {output_dir}")

# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Define dataset path and output directories
    # (Adjust 'dataset' to your folder name if it's different)
    DATASET_PATH = "dataset"
    OUTPUT_GRAPHICS_DIR = "reports/figures"
    OUTPUT_REPORT_DIR = "reports"
    
    # Ensure all output directories exist before writing files
    os.makedirs(OUTPUT_GRAPHICS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)

    print("==========================================================")
    print("       STARTING EXPLORATORY DATA ANALYSIS (EDA)           ")
    print("==========================================================")

    # --- STEP 1: CLASS DISTRIBUTION & QUALITY IMBALANCE ---
    print("\n[1/5] Running class distribution analysis...")
    df_metrics = extract_dataset_metrics(DATASET_PATH)
    
    if not df_metrics.empty:
        print("\n--- DATASET SUMMARY TABLE ---")
        print(df_metrics.to_string(index=False))
        
        # Calculate and print precise imbalance percentages for the markdown report
        total_images = df_metrics["Count"].sum()
        df_imbalance = df_metrics.groupby("Quality")["Count"].sum().reset_index()
        df_imbalance["Percentage"] = (df_imbalance["Count"] / total_images) * 100
        
        print("\n--- QUALITY CLASS IMBALANCE ANALYSIS ---")
        for _, row in df_imbalance.iterrows():
            print(f"  - {row['Quality']}: {row['Count']} images ({row['Percentage']:.2f}%)")
            
        # Plot and save class distribution
        plot_class_distribution(df_metrics, OUTPUT_GRAPHICS_DIR)
    else:
        print("[-] Error: No class data found. Check your DATASET_PATH directory structure.")

    print("\n==========================================================")
    # --- STEP 2: IMAGE PHYSICAL PROPERTIES & QUALITY METRICS ---
    print("[2/5] Running physical metadata and quality profiling...")
    # Using sample_limit=45 per subfolder keeps execution fast while remaining statistically robust
    df_props = extract_image_properties(DATASET_PATH, sample_limit=45)
    
    if not df_props.empty:
        print("\n--- DESCRIPTIVE STATISTICS GENERAL SUMMARY ---")
        print(df_props[['Width', 'Height', 'Brightness', 'Contrast', 'Sharpness']].describe().to_string())
        
        print("\n--- FILE FORMAT DISTRIBUTION ---")
        print(df_props['Format'].value_counts().to_string())
        
        # Save baseline dimension and metric plots
        plot_dimensions_distribution(df_props, OUTPUT_GRAPHICS_DIR)
        plot_quality_metrics(df_props, OUTPUT_GRAPHICS_DIR)
    else:
        print("[-] Error: No valid images found to calculate physical properties.")

    print("\n==========================================================")
    # --- STEP 3: ADVANCED EDA EXTENSIONS (SIZE & DISCRIMINANT ANALYSES) ---
    print("[3/5] Running advanced sizing and metric discriminant analyses...")
    if not df_props.empty:
        # Core extension 1: Fruit Size bounding distribution (Crucial for your size estimation task)
        plot_size_and_outlier_analysis(df_props, OUTPUT_GRAPHICS_DIR)
        
        # Core extension 2: Multi-metric correlation (To check if features can linearly separate classes)
        plot_interclass_discriminant_matrix(df_props, OUTPUT_GRAPHICS_DIR)
    else:
        print("[-] Skipping advanced extensions due to missing properties dataframe.")

    print("\n==========================================================")
    # --- STEP 4: CHROMATIC & HSV SEGMENTATION FEASIBILITY PROFILING ---
    print("[4/5] Running chromatic profiles and background segmentability checks...")
    # Automatically locate a representative Good and Bad image for the comparison
    good_img, bad_img = find_representative_images(DATASET_PATH)
    
    if good_img and bad_img:
        # Standard RGB channel analysis
        plot_automatic_color_comparison(DATASET_PATH, OUTPUT_GRAPHICS_DIR)
        
        # HSV space distribution analysis to validate color-threshold segmentation feasibility
        plot_hsv_segmentation_feasibility(good_img, OUTPUT_GRAPHICS_DIR)
    else:
        print("[-] Skipping color analysis: Could not find representative Good/Bad image pairs.")

    print("\n==========================================================")
    # --- STEP 5: AUTOMATED MARKDOWN REPORT GENERATION ---
    print("[5/5] Compiling and writing automated report (eda_report.md)...")
    if not df_metrics.empty and not df_props.empty:
        generate_eda_markdown_report(df_metrics, df_props, OUTPUT_REPORT_DIR)
    else:
        print("[-] Missing data fields. Automated report compilation aborted.")

    print("\n==========================================================")
    print("     Exploratory Data Analysis (EDA) Process Completed!   ")
    print(f"     All figures saved in: '{OUTPUT_GRAPHICS_DIR}'")
    print(f"     Markdown raw metrics saved in: '{OUTPUT_REPORT_DIR}/eda_report.md'")
    print("==========================================================")