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
        Initializes the model by loading weights from file or setting up a fallback simulation.
        
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
        self.is_simulated = True
        
        # Load the corresponding model if checkpoint files exist
        if self.model_type == 'svm':
            model_path = os.path.join(checkpoints_dir, 'svm.pkl')
            scaler_path = os.path.join(checkpoints_dir, 'scaler.pkl')
            le_path = os.path.join(checkpoints_dir, 'label_encoder.pkl')
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
                try:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.label_encoder = joblib.load(le_path)
                    self.is_simulated = False
                    print("Successfully loaded SVM quality model, scaler, and label encoder.")
                except Exception as e:
                    print(f"Error loading SVM model: {e}. Falling back to simulation.")
                    
        elif self.model_type == 'xgboost':
            model_path = os.path.join(checkpoints_dir, 'xgboost.pkl')
            scaler_path = os.path.join(checkpoints_dir, 'scaler_xgb.pkl')
            le_path = os.path.join(checkpoints_dir, 'label_encoder_xgb.pkl')
            if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
                try:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.label_encoder = joblib.load(le_path)
                    self.is_simulated = False
                    print("Successfully loaded XGBoost quality model, scaler, and label encoder.")
                except Exception as e:
                    print(f"Error loading XGBoost model: {e}. Falling back to simulation.")
                    
        elif self.model_type == 'cnn':
            model_path = os.path.join(checkpoints_dir, 'cnn_best.keras')
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoints_dir, 'cnn_final.keras')
            le_path = os.path.join(checkpoints_dir, 'label_encoder_cnn.pkl')
            if os.path.exists(model_path) and os.path.exists(le_path):
                try:
                    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                    self.label_encoder = joblib.load(le_path)
                    self.is_simulated = False
                    print("Successfully loaded CNN quality model and label encoder.")
                except Exception as e:
                    print(f"Error loading CNN model: {e}. Falling back to simulation.")

        if self.is_simulated:
            print(f"Warning: No valid model checkpoint found for {self.model_type.upper()}. Running in simulation mode.")

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
        Predicts quality of a fruit image.
        
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

        if self.is_simulated or self.model is None:
            # Fallback indicating that the model runs in simulation mode
            mean_intensity = np.mean(image)
            if mean_intensity > 150:
                probs = [0.85, 0.10, 0.05]  # Mostly Excellent
            elif mean_intensity > 90:
                probs = [0.15, 0.70, 0.15]  # Mostly Good
            else:
                probs = [0.05, 0.15, 0.80]  # Mostly Defective
                
            pred_idx = int(np.argmax(probs))
            
            return {
                "class_name": self.classes[pred_idx],
                "confidence": probs[pred_idx],
                "probabilities": {self.classes[i]: probs[i] for i in range(len(self.classes))},
                "simulated": True
            }
            
        try:
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
        except Exception as e:
            print(f"Error executing real model inference: {e}. Falling back to simulation.")
            return self.predict_quality(image)

    def estimate_size(self, image: np.ndarray, pixels_to_cm_ratio: float = 0.026) -> dict:
        """
        Estimates the physical size of the fruit using contour detection
        and a pixel-to-centimeter conversion ratio.
        
        Args:
            image (np.ndarray): Image in BGR format.
            pixels_to_cm_ratio (float): Conversion ratio (cm per pixel).
                                        Default is ~0.026 cm/px.
                                        
        Returns:
            dict: Estimated size parameters (diameter in px, diameter in cm, contours).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Filter noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Edge detection or thresholding
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
        
        # Get minimum enclosing circle for the contour
        (x, y), radius = cv2.minEnclosingCircle(main_contour)
        diameter_px = radius * 2
        
        # Convert to real centimeters
        diameter_cm = diameter_px * pixels_to_cm_ratio
        
        # Axis-aligned bounding box
        bx, by, bw, bh = cv2.boundingRect(main_contour)
        
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
