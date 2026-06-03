"""
Data Preprocessing Module
--------------------------
This script contains helper functions to load, resize,
normalize, and transform fruit and vegetable images before being
processed by the analytics models.
"""

import cv2
import numpy as np
from PIL import Image

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from a file path.
    
    Args:
        image_path (str): Path of the image file.
        
    Returns:
        np.ndarray: Loaded image in BGR format (OpenCV).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    return image

def preprocess_for_cnn(image: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocesses an image to be fed into a convolutional neural network (CNN).
    
    Args:
        image (np.ndarray): BGR or RGB image.
        target_size (tuple): Target size (width, height).
        
    Returns:
        np.ndarray: Resized and normalized image (values between 0.0 and 1.0).
    """
    # Resize the image
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # If BGR (OpenCV), convert to RGB (CNNs typically expect RGB)
    if len(resized.shape) == 3 and resized.shape[2] == 3:
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    else:
        resized_rgb = resized
        
    # Normalize values to [0, 1]
    normalized = resized_rgb.astype(np.float32) / 255.0
    
    # Expand dimensions to simulate a batch (batch size of 1)
    batch_image = np.expand_dims(normalized, axis=0)
    
    return batch_image

def segment_fruit(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments the main object (fruit/vegetable) from its background using
    HSV color space thresholding (based on color and contrast).
    
    Args:
        image (np.ndarray): BGR image.
        
    Returns:
        tuple: (binary mask, segmented image)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Smooth to reduce noise
    blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
    
    # Example thresholding for segmentation (should be adjusted based on background)
    # Using Otsu thresholding on saturation/value channel
    s_channel = blurred[:, :, 1]
    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to close holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to the original image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    return mask, segmented
