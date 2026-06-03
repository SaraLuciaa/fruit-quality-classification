# Fruit and Vegetable Classification System Architecture

This document presents the preliminary architecture designed for the fruit and vegetable quality classification and size estimation system. The design prioritizes modularity, ease of integration for machine learning models, and interactivity in the user interface.

---

## Component Details

### 1. User Interface (src/main.py)
Developed in Streamlit, it provides a dynamic interface that allows:
* Uploading images (.jpg, .jpeg, .png).
* Configuration of parameters in the sidebar (detection thresholds, size calibration factors, etc.).
* Real-time visualization of the analyzed fruit with bounding boxes, segmentation contours, and labels.
* Analytical dashboards with interactive charts (Plotly) for batch quality consolidation.

### 2. Data Preprocessing (src/data/preprocess.py)
Module responsible for preparing images before they are sent to the models:
* Image resizing.
* Color space conversion (e.g., BGR to RGB, or BGR to HSV for color-based segmentation).
* Pixel normalization.
* Data augmentation for training.

### 3. Modeling and Inference (src/models/my_model.py)
Contains the classes and logic to load and run the computer vision models:
* **Quality Classifier:** Convolutional Neural Network (e.g., MobileNetV2, ResNet, or a custom CNN) trained to classify products into categories like Excellent (Class A), Good (Class B), or Defective/Overripe (Class C).
* **Size Estimator:** Algorithm that uses OpenCV contour detection to find the main contour of the product, calculate its diameter in pixels, and convert it to real centimeters using a pre-calibrated pixel-to-millimeter ratio (or a known reference object).

### 4. Utilities (src/utils/helpers.py)
Convenience functions supporting system visualization and operations:
* Drawing contours, diameters, and labels on the original image for visual feedback to the user.
* Quick statistical calculations for processed batches.

---

## Data Flow

1. **Upload:** The user uploads a photo of a batch of fruits (e.g., tomatoes) to the Streamlit app.
2. **Segmentation and Detection:** The system isolates the fruit from the background using OpenCV thresholding in HSV space or a deep learning object detection model.
3. **Measurement:** The minor and major axes of the minimum enclosing circle/ellipse are calculated to determine the diameter in pixels, which is converted to centimeters using the calibration factor.
4. **Classification:** The cropped region of the segmented fruit is fed into the CNN, which outputs the probability of belonging to each quality category.
5. **Report:** Bounding boxes and labels are drawn on the image, and session analytical charts are updated dynamically.
