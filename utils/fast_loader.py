import os
import logging
import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

try:
    from turbojpeg import TurboJPEG  # type: ignore
    _jpeg = TurboJPEG()
except (ImportError, Exception):
    _jpeg = None
    logger.warning("TurboJPEG not available; falling back to slower PIL/OpenCV loading.")

MAX_EDGE = 640
BLUR_REJECT = 20  # variance of Laplacian

def _blur_score(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()

from typing import Optional

def fast_imread(path: str) -> Optional[np.ndarray]:
    """Return down-scaled BGR or None if unreadable / too blurry."""
    try:
        ext = os.path.splitext(path)[1].lower()
        rgb = None
        
        # Try TurboJPEG if available and file is jpeg
        if _jpeg and ext in {".jpg", ".jpeg"}:
            try:
                with open(path, "rb") as f:
                    rgb = _jpeg.decode(f.read())
            except Exception:
                # Fallback if decode fails
                pass
        
        # Fallback to PIL if rgb is still None
        if rgb is None:
            pil = Image.open(path)
            pil = ImageOps.exif_transpose(pil)
            rgb = np.array(pil)
            
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Down-scale
        h, w = bgr.shape[:2]
        scale = min(MAX_EDGE / max(h, w), 1.0)
        if scale < 1:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        # Cheap blur gate only
        if _blur_score(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)) < BLUR_REJECT:
            logger.warning(f"Image too blurry (score < {BLUR_REJECT}): {path}")
            return None
        return bgr
    except Exception as e:
        logger.error(f"Failed to load image {path}: {str(e)}")
        return None