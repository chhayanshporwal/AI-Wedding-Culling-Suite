# filters/person_filter.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (ensure 'yolov8n.pt' is downloaded and placed in project root)
model = YOLO('yolov8n.pt')

def detect_persons(image: np.ndarray, conf_thresh: float = 0.3) -> list[tuple[float, float, float, float]]:
    """
    Detect persons in the image using YOLOv8. Returns a list of bounding boxes
    as tuples (x1, y1, x2, y2) for each detected person.
    """
    if image is None:
        return []

    # Run inference on CPU with confidence threshold
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
        if cls_idx != 0:
            continue

        # Extract bounding box coordinates (numpy.ndarray)
        coords_arr = box.xyxy
        coords = coords_arr.reshape(-1).tolist()
        if len(coords) == 4:
            person_boxes.append((coords[0], coords[1], coords[2], coords[3]))
    return person_boxes
