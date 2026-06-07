"""
Modeling and Inference Module
-------------------------------
This file defines the class or functions wrapping the quality
classification models and sizing estimation algorithms.
"""

import cv2
import numpy as np
import os
import joblib

class FruitQualityModel:
    """
    Wrapper class for quality classification and size/quality analysis models.
    """
    
    def __init__(self, model_type: str = 'svm', checkpoints_dir: str = 'experiments/checkpoints'):
        """
        Initializes the model by loading weights from file.
        
        Args:
            model_type (str): The model type to load ('svm', 'xgboost', 'cnn').
            checkpoints_dir (str): Directory where the trained checkpoints are stored.
        """
        self.model_type = model_type.lower()
        self.checkpoints_dir = checkpoints_dir
        self.classes = ['Excellent Quality (Class A)', 'Good Quality (Class B)', 'Defective Quality (Class C)']
        
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        # Load the corresponding model
        if self.model_type == 'svm':
            model_path = os.path.join(checkpoints_dir, 'svm.pkl')
            scaler_path = os.path.join(checkpoints_dir, 'scaler.pkl')
            le_path = os.path.join(checkpoints_dir, 'label_encoder.pkl')
            if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
                raise FileNotFoundError(f"SVM model checkpoints not found in '{checkpoints_dir}'. Please run the training pipeline first.")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            print("Successfully loaded SVM quality model, scaler, and label encoder.")
                    
        elif self.model_type == 'xgboost':
            model_path = os.path.join(checkpoints_dir, 'xgboost.pkl')
            scaler_path = os.path.join(checkpoints_dir, 'scaler_xgb.pkl')
            le_path = os.path.join(checkpoints_dir, 'label_encoder_xgb.pkl')
            if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
                raise FileNotFoundError(f"XGBoost model checkpoints not found in '{checkpoints_dir}'. Please run the training pipeline first.")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
            print("Successfully loaded XGBoost quality model, scaler, and label encoder.")
                    
        elif self.model_type == 'cnn':
            model_path = os.path.join(checkpoints_dir, 'cnn_best.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoints_dir, 'cnn_final.keras')
            le_path = os.path.join(checkpoints_dir, 'label_encoder_cnn.pkl')
            if not (os.path.exists(model_path) and os.path.exists(le_path)):
                raise FileNotFoundError(f"CNN model checkpoints not found in '{checkpoints_dir}'. Please run the training pipeline first.")
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self.label_encoder = joblib.load(le_path)
            print("Successfully loaded CNN quality model and label encoder.")

    def _extract_single_image_features(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts HOG and color histogram features from a BGR image to feed ML models.
        """
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))

        # HOG features
        from skimage.feature import hog
        hog_feat = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                       feature_vector=True)

        # Color histogram features (RGB)
        hist_r = cv2.calcHist([img_rgb], [0], None, [32], [0, 256]).flatten()
        hist_g = cv2.calcHist([img_rgb], [1], None, [32], [0, 256]).flatten()
        hist_b = cv2.calcHist([img_rgb], [2], None, [32], [0, 256]).flatten()
        hist_feat = np.concatenate([hist_r, hist_g, hist_b])

        # Statistical features (mean, std per channel)
        stats = np.array([img_rgb[:,:,c].mean() for c in range(3)] +
                         [img_rgb[:,:,c].std() for c in range(3)])

        feat = np.concatenate([hog_feat / (hog_feat.max() + 1e-8),
                               hist_feat / (hist_feat.sum() + 1e-8),
                               stats / 255.0])
        return feat.astype(np.float32)

    def predict_quality(self, image: np.ndarray) -> dict:
        """
        Predicts quality of a fruit image using the loaded real model.
        
        Args:
            image (np.ndarray): Image in BGR format (OpenCV).
            
        Returns:
            dict: Results with predicted class, confidence, probabilities, and simulation status.
        """
        short_to_long = {
            "Excellent": "Excellent Quality (Class A)",
            "Good": "Good Quality (Class B)",
            "Defective": "Defective Quality (Class C)"
        }

        if self.model_type in ['svm', 'xgboost']:
            # Extract and scale features
            feat = self._extract_single_image_features(image)
            feat_scaled = self.scaler.transform([feat])
            
            # Predict
            pred_idx = self.model.predict(feat_scaled)[0]
            probs = self.model.predict_proba(feat_scaled)[0]
            
            # Decode class
            pred_class_short = self.label_encoder.inverse_transform([pred_idx])[0]
            pred_class_long = short_to_long.get(pred_class_short, pred_class_short)
            confidence = float(probs[pred_idx])
            
            probabilities = {}
            for idx, prob in enumerate(probs):
                cls_short = self.label_encoder.classes_[idx]
                cls_long = short_to_long.get(cls_short, cls_short)
                probabilities[cls_long] = float(prob)
                
        elif self.model_type == 'cnn':
            # Preprocess for CNN (Resize to 224x224 and scale to [0, 1])
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (224, 224))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predict
            probs = self.model.predict(img_batch, verbose=0)[0]
            pred_idx = np.argmax(probs)
            
            # Decode class
            pred_class_short = self.label_encoder.inverse_transform([pred_idx])[0]
            pred_class_long = short_to_long.get(pred_class_short, pred_class_short)
            confidence = float(probs[pred_idx])
            
            probabilities = {}
            for idx, prob in enumerate(probs):
                cls_short = self.label_encoder.classes_[idx]
                cls_long = short_to_long.get(cls_short, cls_short)
                probabilities[cls_long] = float(prob)

        return {
            "class_name": pred_class_long,
            "confidence": confidence,
            "probabilities": probabilities,
            "simulated": False
        }

    def estimate_size(self, image: np.ndarray, pixels_to_cm_ratio: float = 0.026) -> dict:
        """
        Estimates the physical size of the fruit using HSV saturation segmentation
        and an equivalent area-based diameter metric.
        
        Args:
            image (np.ndarray): Image in BGR format.
            pixels_to_cm_ratio (float): Conversion ratio (cm per pixel).
                                        Default is ~0.026 cm/px.
                                        
        Returns:
            dict: Estimated size parameters (diameter in px, diameter in cm, contours).
        """
        # 1. Segment using Saturation channel of HSV space (removes shadows and background)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Otsu thresholding on Saturation channel
        _, thresh = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask (close holes, open borders)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "diameter_px": 0.0,
                "diameter_cm": 0.0,
                "area_px": 0.0,
                "bbox": None,
                "message": "No contours detected."
            }
            
        # Get contour with the largest area (assuming it's the main fruit)
        main_contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(main_contour)
        
        # 2. Calculate equivalent area-based diameter: D = 2 * sqrt(Area / pi)
        # This is extremely stable against shadow noise or small contour protrusions
        import math
        diameter_px = 2 * math.sqrt(area_px / math.pi)
        
        # Convert to real centimeters
        diameter_cm = diameter_px * pixels_to_cm_ratio
        
        # Axis-aligned bounding box
        bx, by, bw, bh = cv2.boundingRect(main_contour)
        
        # Get minimum enclosing circle for centroid and visual tracking
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        
        return {
            "diameter_px": round(diameter_px, 1),
            "diameter_cm": round(diameter_cm, 2),
            "area_px": int(area_px),
            "bbox": (bx, by, bw, bh),
            "center": (int(x), int(y)),
            "radius": int(radius),
            "contour": main_contour,
            "message": "Success"
        }
