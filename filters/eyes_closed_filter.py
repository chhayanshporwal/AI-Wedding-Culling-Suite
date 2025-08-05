#filters/eyes_closed_filter.py
import logging
import threading
import cv2
import numpy as np
import mediapipe as mp  # type: ignore

# Thread-local storage for MediaPipe instances (to make it thread-safe)
local = threading.local()
logger = logging.getLogger(__name__)

# Landmark indices for eye aspect ratio (EAR)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def get_face_mesh():
    """
    Get or create a thread-local MediaPipe FaceMesh instance.
    This ensures each thread has its own independent graph, avoiding timestamp conflicts.
    """
    if not hasattr(local, "mp_face_mesh"):
        local.mp_face_mesh = mp.solutions.face_mesh.FaceMesh( # type: ignore
            static_image_mode=True,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return local.mp_face_mesh

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Compute Eye Aspect Ratio (EAR) for one eye given landmarks and their indices.
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||).
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        pts.append((x, y))

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def is_eyes_closed(image: np.ndarray, ear_threshold: float = 0.2) -> bool:
    """
    Returns True if any detected face in the image has average EAR < threshold.
    """
    if image is None or image.size == 0:
        logger.warning("Invalid image provided to eyes_closed filter; skipping.")
        return False

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = get_face_mesh()
    results = mp_face_mesh.process(img_rgb)  # type: ignore

    if not results.multi_face_landmarks:
        return False

    h, w = image.shape[:2]
    for face_landmarks in results.multi_face_landmarks:
        lm = face_landmarks.landmark
        left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)
        if (left_ear + right_ear) / 2.0 < ear_threshold:
            return True
    return False
