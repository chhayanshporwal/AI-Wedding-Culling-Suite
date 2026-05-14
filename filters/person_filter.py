#filters/person_filter.py
import logging
import threading
import numpy as np
from ultralytics import YOLO

# Thread-local storage for YOLO models (to make it thread-safe)
local = threading.local()
logger = logging.getLogger(__name__)

def get_yolo_model():
    """
    Get or create a thread-local YOLO model instance.
    This ensures each thread has its own model, avoiding shared state and fusing conflicts.
    """
    if not hasattr(local, "model"):
        local.model = YOLO('yolov8n.pt')  # Adjust path if needed
    return local.model

def detect_persons(image: np.ndarray, conf_thresh: float = 0.3) -> list[tuple[float, float, float, float]]:
    """
    Detect persons in the image using YOLOv8. Returns a list of bounding boxes
    as tuples (x1, y1, x2, y2) for each detected person.
    """
    if image is None or image.size == 0:
        logger.warning("Invalid image provided to person_filter; skipping.")
        return []

    try:
        # Get thread-local model
        model = get_yolo_model()

        # Run inference on CPU with confidence threshold, disabling fuse to avoid errors
        results = model.predict(source=image, device='cpu', conf=conf_thresh, verbose=False) or []

        if len(results) == 0 or results[0] is None:
            return []

        batch_result = results[0]

        # Ensure boxes attribute exists
        if not hasattr(batch_result, 'boxes') or batch_result.boxes is None:
            return []

        person_boxes: list[tuple[float, float, float, float]] = []
        for box in batch_result.boxes:
            # Extract class index
            cls_val = box.cls
            try:
                cls_idx = int(cls_val[0])
            except Exception:
                cls_idx = int(cls_val)
            if cls_idx != 0:  # Class 0 is 'person' in COCO
                continue

            # Extract bounding box coordinates (numpy.ndarray)
            coords_arr = box.xyxy
            coords = coords_arr.reshape(-1).tolist()
            if len(coords) == 4:
                person_boxes.append((coords[0], coords[1], coords[2], coords[3]))

        return person_boxes

    except Exception as e:
        logger.error(f"Error during person detection: {str(e)}")
        return []