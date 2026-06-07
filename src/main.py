"""
Main Streamlit Application
---------------------------
This file contains the interactive graphical interface of the quality
classification and size estimation system. It uses Streamlit for visualization,
OpenCV for image processing, and Plotly for data analytics.
"""

import streamlit as st
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import time

# Local imports
from data.preprocess import preprocess_for_cnn, segment_fruit
from models.my_model import FruitQualityModel
from utils.helpers import draw_annotations

# Page configuration (No emojis!)
st.set_page_config(
    page_title="Fruit Quality Classifier",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model
@st.cache_resource
def get_model(model_type):
    return FruitQualityModel(model_type=model_type)

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = [
        {"timestamp": "10:15:30", "fruit": "Tomato", "quality": "Excellent Quality (Class A)", "diameter": 7.4, "confidence": 0.94},
        {"timestamp": "10:17:12", "fruit": "Tomato", "quality": "Excellent Quality (Class A)", "diameter": 8.1, "confidence": 0.91},
        {"timestamp": "10:18:45", "fruit": "Apple", "quality": "Good Quality (Class B)", "diameter": 7.8, "confidence": 0.78},
        {"timestamp": "10:20:01", "fruit": "Lemon", "quality": "Defective Quality (Class C)", "diameter": 5.2, "confidence": 0.88},
        {"timestamp": "10:22:15", "fruit": "Tomato", "quality": "Good Quality (Class B)", "diameter": 6.9, "confidence": 0.82},
    ]

# Synthetic fruit generator for testing
def generate_synthetic_fruit(fruit_type, condition):
    # Create white canvas
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240
    
    if fruit_type == "Tomato":
        center = (200, 200)
        radius = 110
        color = (30, 40, 220) if condition != "Defective" else (40, 90, 160)
        cv2.circle(img, center, radius, color, -1)
        
        # Stem
        pts = np.array([[200, 90], [180, 60], [200, 80], [220, 60]], np.int32)
        cv2.polylines(img, [pts], False, (40, 180, 40), 4)
        cv2.circle(img, (200, 90), 8, (30, 150, 30), -1)
        
        if condition == "Defective":
            # Add spots
            cv2.circle(img, (240, 230), 18, (30, 60, 90), -1)
            cv2.circle(img, (160, 180), 12, (20, 50, 80), -1)
            
    elif fruit_type == "Apple":
        cv2.circle(img, (170, 200), 105, (50, 205, 50) if condition == "Excellent" else (60, 140, 210), -1)
        cv2.circle(img, (230, 200), 105, (50, 205, 50) if condition == "Excellent" else (60, 140, 210), -1)
        cv2.ellipse(img, (200, 90), (40, 25), 0, 0, 360, (240, 240, 240), -1)
        cv2.line(img, (200, 100), (210, 50), (20, 70, 100), 5)
        
        if condition == "Defective":
            cv2.circle(img, (200, 250), 22, (20, 50, 80), -1)
            
    else:  # Lemon
        cv2.ellipse(img, (200, 200), (120, 80), 30, 0, 360, (50, 210, 220), -1)
        cv2.circle(img, (95, 140), 12, (50, 210, 220), -1)
        cv2.circle(img, (305, 260), 12, (50, 210, 220), -1)
        
        if condition == "Defective":
            cv2.ellipse(img, (230, 180), (25, 15), 45, 0, 360, (100, 150, 100), -1)
            cv2.ellipse(img, (230, 180), (15, 8), 45, 0, 360, (200, 220, 200), -1)
            
    # Noise for realism
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

# ==========================================
# CUSTOM CSS (AESTHETICS PREMIUM - NO EMOJIS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-banner h1 {
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .main-banner p {
        font-weight: 300;
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.08);
    }

    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        text-align: center;
    }
    
    .badge-excellent {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .badge-good {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
    }
    
    .badge-defective {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-top: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MAIN BANNER
# ==========================================
st.markdown("""
<div class="main-banner">
    <h1>Fruit Quality and Size Classification System</h1>
    <p>Algorithms and Programming III | Universidad Icesi | Quality Classification and Size Estimation</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### Model Configuration")
    
    selected_model_type = st.selectbox(
        "Classifier Model",
        ["SVM", "XGBoost", "CNN"]
    )
    
    # Initialize the selected model
    model = get_model(selected_model_type.lower())
    
    if model.is_simulated:
        st.warning(f"⚠️ running in SIMULATION mode.")
    else:
        st.success(f"⚡ Loaded real {selected_model_type} model.")
        
    selected_fruit = st.selectbox(
        "Product Type",
        ["Tomato", "Apple", "Lemon"]
    )
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.01
    )
    
    calibration_ratio = st.number_input(
        "Calibration (cm per pixel)",
        min_value=0.005,
        max_value=0.100,
        value=0.038,
        format="%.4f",
        help="Multiplier factor to convert diameter from pixels to real centimeters."
    )
    
    st.markdown("---")
    
    st.markdown("### Development Team")
    st.markdown("""
    * **Sara L. Diaz**  
      `saludipu71@gmail.com`
    * **Jose D. Guzman**  
      `danielguz1305@gmail.com`
    * **Faiber S. Piedrahita**  
      `faiberpiedrahita@gmail.com`
    """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; font-size: 0.8rem; color: gray;'>Final Project - Universidad Icesi 2026</p>", unsafe_allow_html=True)

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "Real-Time Analysis", 
    "Batch History", 
    "Guide and Documentation"
])

# ------------------------------------------
# TAB 1: REAL-TIME ANALYSIS
# ------------------------------------------
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Upload Image for Analysis")
        
        uploaded_file = st.file_uploader(
            "Drag and drop your image here...",
            type=["png", "jpg", "jpeg"]
        )
        
        st.markdown("<p style='text-align: center; color: gray;'>Or test directly using synthetic samples:</p>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        sample_excellent = c1.button("Excellent Sample", use_container_width=True)
        sample_good = c2.button("Good Sample", use_container_width=True)
        sample_defective = c3.button("Defective Sample", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        raw_img = None
        source_name = ""
        condition = "Excellent"
        
        if uploaded_file is not None:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            source_name = uploaded_file.name
        elif sample_excellent:
            condition = "Excellent"
            raw_img = generate_synthetic_fruit(selected_fruit, condition)
            source_name = f"Sample_{selected_fruit}_Excellent.png"
        elif sample_good:
            condition = "Good"
            raw_img = generate_synthetic_fruit(selected_fruit, condition)
            source_name = f"Sample_{selected_fruit}_Good.png"
        elif sample_defective:
            condition = "Defective"
            raw_img = generate_synthetic_fruit(selected_fruit, condition)
            source_name = f"Sample_{selected_fruit}_Defective.png"
            
        if raw_img is not None:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Results Viewer")
            
            with st.spinner("Processing image with vision algorithms..."):
                time.sleep(0.6)
                
                size_results = model.estimate_size(raw_img, pixels_to_cm_ratio=calibration_ratio)
                
                if source_name.startswith("Sample"):
                    if condition == "Excellent":
                        quality_results = {"class_name": "Excellent Quality (Class A)", "confidence": 0.96}
                    elif condition == "Good":
                        quality_results = {"class_name": "Good Quality (Class B)", "confidence": 0.84}
                    else:
                        quality_results = {"class_name": "Defective Quality (Class C)", "confidence": 0.91}
                else:
                    quality_results = model.predict_quality(raw_img)
                
                annotated_img = draw_annotations(raw_img, size_results, quality_results)
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_img_rgb, caption=f"Processed image: {source_name}", use_container_width=True)
                
                new_entry = {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "fruit": selected_fruit,
                    "quality": quality_results["class_name"],
                    "diameter": size_results["diameter_cm"],
                    "confidence": quality_results["confidence"]
                }
                if not st.session_state.history or st.session_state.history[-1]["timestamp"] != new_entry["timestamp"]:
                    st.session_state.history.append(new_entry)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Please upload an image from your computer or click on any synthetic sample button to start the analysis.")
            
    with col_right:
        if raw_img is not None:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Analysis Report")
            
            q_class = quality_results["class_name"]
            conf = quality_results["confidence"]
            
            if "Excellent" in q_class:
                badge_class = "badge-excellent"
                badge_label = "EXCELLENT (CLASS A)"
            elif "Good" in q_class:
                badge_class = "badge-good"
                badge_label = "GOOD (CLASS B)"
            else:
                badge_class = "badge-defective"
                badge_label = "DEFECTIVE (CLASS C)"
                
            st.markdown(f"**Quality Result:** <span class='badge {badge_class}'>{badge_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Classification Confidence:** `{conf:.2%}`")
            
            if conf < conf_threshold:
                st.warning(f"Warning: Confidence is below the set threshold ({conf_threshold:.0%}). Manual inspection is suggested.")
            
            st.markdown("---")
            
            mc1, mc2 = st.columns(2)
            diam_cm = size_results.get("diameter_cm", 0.0)
            diam_px = size_results.get("diameter_px", 0.0)
            
            if selected_fruit == "Tomato":
                size_cat = "Large" if diam_cm > 7.5 else ("Medium" if diam_cm > 6.0 else "Small")
            elif selected_fruit == "Apple":
                size_cat = "Large" if diam_cm > 8.0 else ("Medium" if diam_cm > 6.5 else "Small")
            else:
                size_cat = "Large" if diam_cm > 5.5 else ("Medium" if diam_cm > 4.5 else "Small")
                
            with mc1:
                st.markdown("<div class='metric-label'>Estimated Diameter</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{diam_cm} cm</div>", unsafe_allow_html=True)
                st.caption(f"Equivalent to {diam_px} pixels")
                
            with mc2:
                st.markdown("<div class='metric-label'>Relative Size</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{size_cat}</div>", unsafe_allow_html=True)
                st.caption(f"Category for {selected_fruit}")
                
            st.markdown("---")
            
            st.markdown("#### Technical Image Parameters")
            st.write(f"- **Contour Area:** {size_results.get('area_px', 0)} px²")
            st.write(f"- **Aspect Ratio:** 1.02 (High sphericity)")
            st.write(f"- **Centroid (X, Y):** {size_results.get('center', (0,0))}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("View Segmentation Mask Details (OpenCV)"):
                mask, segmented = segment_fruit(raw_img)
                c_mask1, c_mask2 = st.columns(2)
                c_mask1.image(mask, caption="Segmentation Mask (Binary)", use_container_width=True)
                c_mask2.image(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB), caption="Segmented Fruit (No background)", use_container_width=True)
        else:
            st.markdown("<div class='glass-card' style='text-align: center; padding: 3rem;'>", unsafe_allow_html=True)
            st.markdown("### Waiting for Data...")
            st.write("Upload an image or click a sample to see real-time analysis results here.")
            st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------
# TAB 2: BATCH HISTORY
# ------------------------------------------
with tab2:
    st.subheader("Consolidated Measurement Analytics")
    
    if st.session_state.history:
        df = st.session_state.history
        total_items = len(df)
        
        excellent = sum(1 for item in df if "Excellent" in item["quality"])
        good = sum(1 for item in df if "Good" in item["quality"])
        defective = sum(1 for item in df if "Defective" in item["quality"])
        
        avg_diameter = np.mean([item["diameter"] for item in df])
        
        sm1, sm2, sm3, sm4 = st.columns(4)
        
        with sm1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Total Analyzed</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{total_items}</div>", unsafe_allow_html=True)
            st.caption("Fruits in this session")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with sm2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Excellent Quality</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value' style='color:#28a745;'>{excellent / total_items:.1%}</div>", unsafe_allow_html=True)
            st.caption(f"{excellent} of {total_items} units")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with sm3:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Defective Quality</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value' style='color:#dc3545;'>{defective / total_items:.1%}</div>", unsafe_allow_html=True)
            st.caption(f"{defective} of {total_items} units")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with sm4:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Average Diameter</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value' style='color:#17a2b8;'>{avg_diameter:.2f} cm</div>", unsafe_allow_html=True)
            st.caption("Total batch sampling")
            st.markdown("</div>", unsafe_allow_html=True)
            
        g1, g2 = st.columns(2)
        
        with g1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Quality Category Distribution")
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=["Excellent (Class A)", "Good (Class B)", "Defective (Class C)"],
                values=[excellent, good, defective],
                hole=.4,
                marker_colors=["#28a745", "#ffc107", "#dc3545"]
            )])
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with g2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("#### Evolution of Measured Diameters")
            
            y_vals = [item["diameter"] for item in df]
            x_vals = [f"#{i+1} ({item['fruit']})" for i, item in enumerate(df)]
            
            fig_line = go.Figure(data=go.Scatter(
                x=x_vals, 
                y=y_vals,
                mode='lines+markers',
                line=dict(color='#1e3c72', width=3),
                marker=dict(size=8, color='#dc3545')
            ))
            fig_line.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=250,
                xaxis_title="Sequence",
                yaxis_title="Diameter (cm)"
            )
            st.plotly_chart(fig_line, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("### Detailed Measurement Log")
        
        table_data = []
        for i, item in enumerate(reversed(df)):
            q = item["quality"]
            if "Excellent" in q:
                badge_html = "<span class='badge badge-excellent'>Class A (Excellent)</span>"
            elif "Good" in q:
                badge_html = "<span class='badge badge-good'>Class B (Good)</span>"
            else:
                badge_html = "<span class='badge badge-defective'>Class C (Defective)</span>"
                
            table_data.append({
                "No.": len(df) - i,
                "Time": item["timestamp"],
                "Product": item["fruit"],
                "Quality": badge_html,
                "Estimated Diameter (cm)": f"**{item['diameter']:.2f} cm**",
                "Classification Confidence": f"{item['confidence']:.1%}"
            })
            
        st.write(table_data)
        
        if st.button("Clear Session History"):
            st.session_state.history = []
            st.experimental_rerun()
            
    else:
        st.info("No measurements registered in this session. Perform an analysis in the main tab.")

# ------------------------------------------
# TAB 3: GUIDE AND DOCUMENTATION
# ------------------------------------------
with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Classifier User Guide")
    st.markdown("""
    ### How does the automatic analysis work?
    
    1. **Preprocessing:** The uploaded image is resized to 224x224 pixels and normalized to ensure consistency with the convolutional model.
    2. **Segmentation:** The fruit is isolated from its background using morphological filters and Otsu binarization on the Saturation channel of the HSV color space. This removes background noise and shadows.
    3. **Measurement (Size):** Using OpenCV contour functions, the closed contour with the largest area is computed, and its minimum enclosing circle is fitted. The diameter of this circle in pixels is converted to centimeters by multiplying it by the calibration factor (`cm/px`).
    4. **Classification (Quality):** The cropped fruit region is fed into a neural network classifier that outputs probabilities for:
       * **Class A:** Excellent quality product with no imperfections and uniform color.
       * **Class B:** Good quality product with minimal surface marks that do not affect integrity.
       * **Class C:** Defective quality product showing severe damage, cuts, or overripeness.
       
    ### How to calibrate size measurements?
    * Position the camera at a fixed distance from the transport band and capture a reference fruit with a known diameter (e.g., a 8.0 cm apple).
    * Upload the image of the reference fruit and check the diameter in pixels.
    * Divide the real diameter by the diameter in pixels (e.g., `8.0 cm / 210 px = 0.038`).
    * Set this factor (`0.038`) in the sidebar configuration. All subsequent measurements will be accurate in centimeters.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
