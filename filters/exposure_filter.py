# filters/exposure_filter.py

import cv2
import numpy as np

def exposure_score(image: np.ndarray) -> float:
    """
    Compute exposure score as the mean pixel intensity of the grayscale image.
    Low values indicate underexposure; high values indicate overexposure.
    """
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))