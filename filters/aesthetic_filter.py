# filters/aesthetic_filter.py

import os
from PIL import Image
import cv2
import numpy as np
from filters.aesthetic_score import AestheticScorer

# Initialize the CLIP-based aesthetic scorer once
# We compute the absolute path to the weights file relative to this script
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ava_logos_linearMSE.pth")
scorer = AestheticScorer(weights_path=WEIGHTS_PATH)

def aesthetic_score(image_cv: np.ndarray) -> float:
    """
    Compute a normalized aesthetic score in [0,1] using CLIP + regression head.
    Accepts an OpenCV BGR image, converts to PIL, and returns the model score.
    """
    if image_cv is None or image_cv.size == 0:
        return 0.0
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Create PIL Image
    pil_img = Image.fromarray(img_rgb)
    # Score via the pretrained AestheticScorer
    try:
        score = scorer.score(pil_img)
        return float(score)
    except Exception as e:
        # Fallback in case of model error
        return 0.0

def get_clip_embedding(image_cv: np.ndarray) -> np.ndarray:
    """
    Extract CLIP embedding for the image using the global scorer.
    """
    if image_cv is None or image_cv.size == 0:
        return None
        
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    try:
        # Returns (1, 768) array
        emb = scorer._extract_clip_features_batch([pil_img])[0]
        return emb
    except Exception:
        return None
