#filter/blue_filter.py
import cv2
import numpy as np

def blur_score(image: np.ndarray) -> float:
    """
    Compute a robust blur score using variance of the Laplacian normalized by image size.
    Larger values = sharper image; smaller values = blurrier image.
    """
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = float(lap.var())
    # Normalize by image area to reduce scale sensitivity
    h, w = gray.shape
    norm = var / (h * w) * 1e6
    return norm
