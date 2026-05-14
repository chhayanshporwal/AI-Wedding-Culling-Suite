# filters/cropped_filter.py

from typing import List, Tuple

def is_cropped(
    face_boxes:  List[Tuple[float, float, float, float]],
    img_w:       float,
    img_h:       float,
    edge_margin: float = 0.05
) -> bool:
    """
    Returns True if any face box comes within `edge_margin` fraction
    of the image border, indicating a cropped face.
    """
    if not face_boxes:
        return False

    w_m = img_w * edge_margin
    h_m = img_h * edge_margin

    for x1, y1, x2, y2 in face_boxes:
        if (x1 < w_m) or (y1 < h_m) or ((img_w - x2) < w_m) or ((img_h - y2) < h_m):
            return True

    return False
