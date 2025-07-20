# utils/fast_loader.py
"""
TurboJPEG + EXIF-rotate + down-scale only.
Blur rejection kept; exposure is left to the pipeline.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from turbojpeg import TurboJPEG  # type: ignore

_jpeg = TurboJPEG()

MAX_EDGE = 640
BLUR_REJECT = 20        # variance of Laplacian


def _blur_score(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


from typing import Optional

def fast_imread(path: str) -> Optional[np.ndarray]:
    """Return down-scaled BGR or None if unreadable / too blurry."""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg"}:
            with open(path, "rb") as f:
                rgb = _jpeg.decode(f.read())
        else:
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
            return None

        return bgr
    except Exception:
        return None