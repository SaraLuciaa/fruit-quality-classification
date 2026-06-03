"""
Utility and Auxiliary Functions Module
---------------------------------------
This file contains functions for visualization and graphical annotations
on processed images (drawing contours, diameters, and text labels).
"""

import cv2
import numpy as np

def draw_annotations(image: np.ndarray, size_info: dict, quality_info: dict) -> np.ndarray:
    """
    Draws visual annotations on the original fruit image (contour,
    bounding box, diameter labels, and quality classification).
    
    Args:
        image (np.ndarray): Original image in BGR format.
        size_info (dict): Sizing results from FruitQualityModel.estimate_size.
        quality_info (dict): Prediction results from FruitQualityModel.predict_quality.
        
    Returns:
        np.ndarray: Annotated image ready to display.
    """
    annotated = image.copy()
    
    # 1. Draw Bounding Box if it exists
    bbox = size_info.get("bbox")
    if bbox:
        bx, by, bw, bh = bbox
        # Blue color for the bounding box
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        
    # 2. Draw segmented contour if it exists
    contour = size_info.get("contour")
    if contour is not None:
        # Green color for the fruit contour
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        
    # 3. Draw center and minimum enclosing circle
    center = size_info.get("center")
    radius = size_info.get("radius")
    if center and radius:
        cx, cy = center
        # Draw center (small red circle)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        # Draw horizontal diameter line
        cv2.line(annotated, (cx - radius, cy), (cx + radius, cy), (0, 0, 255), 2)
        
    # 4. Add text label for size and quality
    # Determine text color based on quality class
    class_name = quality_info.get("class_name", "")
    if "Excellent" in class_name:
        text_color = (0, 255, 0)  # Green
    elif "Good" in class_name:
        text_color = (0, 255, 255)  # Yellow
    else:
        text_color = (0, 0, 255)  # Red
        
    # Position text at the top-left of the bounding box
    tx = bx if bbox else 20
    ty = by - 10 if (bbox and by - 10 > 20) else 40
    
    # Quality text
    quality_text = f"{class_name} ({quality_info.get('confidence', 0.0):.1%})"
    cv2.putText(annotated, quality_text, (tx, ty), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    
    # Size text
    diameter_cm = size_info.get("diameter_cm", 0.0)
    size_text = f"D: {diameter_cm} cm"
    cv2.putText(annotated, size_text, (tx, ty + 25 if bbox else ty + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
    return annotated
