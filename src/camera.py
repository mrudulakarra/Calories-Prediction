"""
Camera & Capture Management Module
Handles camera capture storage, preview, and mode routing.
"""

import os
from datetime import datetime
from PIL import Image
import numpy as np

def save_uploaded_or_captured_image(image_file, subfolder="images"):
    """
    Saves a captured or uploaded image to the uploads directory.
    Returns relative filepath and PIL Image object.
    """
    target_dir = os.path.join("uploads", subfolder)
    os.makedirs(target_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(target_dir, filename)
    
    # Open with PIL and save standard JPEG
    img = Image.open(image_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(filepath, "JPEG", quality=90)
    
    return filepath, img
