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
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # Create PIL Image
    pil_img = Image.fromarray(img_rgb)
    # Score via the pretrained AestheticScorer
    score = scorer.score(pil_img)
    return float(score)
