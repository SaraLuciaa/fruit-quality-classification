"""
Project Unit Tests
-------------------
This script contains the basic unit tests to validate the functionality
of preprocessing, segmentation, and size estimation functions.
"""

import sys
import os
import numpy as np
import pytest

# Add src/ directory to path to allow correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.preprocess import preprocess_for_cnn, segment_fruit
from models.my_model import FruitQualityModel

def test_preprocess_for_cnn():
    """
    Verifies that preprocessing resizes correctly and returns the
    expected shape for the convolutional network (1, 224, 224, 3).
    """
    # Create dummy random image of 500x500x3
    dummy_img = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    
    processed = preprocess_for_cnn(dummy_img, target_size=(224, 224))
    
    # Check batch dimensions (batch size, H, W, C)
    assert processed.shape == (1, 224, 224, 3)
    # Check normalization (values between 0.0 and 1.0)
    assert np.min(processed) >= 0.0
    assert np.max(processed) <= 1.0

def test_segment_fruit():
    """
    Verifies that the segmentation function returns a binary mask
    of the same spatial dimensions as the original image.
    """
    # Create mock image (red circle on gray background)
    dummy_img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    import cv2
    cv2.circle(dummy_img, (50, 50), 30, (0, 0, 255), -1)
    
    mask, segmented = segment_fruit(dummy_img)
    
    assert mask.shape == (100, 100)
    assert segmented.shape == (100, 100, 3)
    # Mask values must be binary (0 or 255)
    assert set(np.unique(mask)).issubset({0, 255})

def test_fruit_quality_model_simulation():
    """
    Verifies that the simulated model returns a dictionary with the
    expected keys and datatypes for quality.
    """
    model = FruitQualityModel()
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    results = model.predict_quality(dummy_img)
    
    assert isinstance(results, dict)
    assert "class_name" in results
    assert "confidence" in results
    assert "probabilities" in results
    assert results["simulated"] is True

def test_estimate_size():
    """
    Verifies that size estimation of a known geometric figure returns
    non-null diameters and areas.
    """
    model = FruitQualityModel()
    
    # Create black canvas of 300x300 and draw a white circle (diameter = 100px)
    import cv2
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(canvas, (150, 150), 50, (255, 255, 255), -1)
    
    # Calibration: 0.1 cm per pixel (expected diameter of 100px * 0.1 = 10 cm)
    size_results = model.estimate_size(canvas, pixels_to_cm_ratio=0.1)
    
    assert size_results["message"] == "Success"
    assert size_results["diameter_px"] > 0
    # Allow small tolerance due to GaussianBlur smoothing
    assert abs(size_results["diameter_cm"] - 10.0) < 1.5
    assert size_results["area_px"] > 0
    assert size_results["bbox"] is not None
    assert len(size_results["bbox"]) == 4
