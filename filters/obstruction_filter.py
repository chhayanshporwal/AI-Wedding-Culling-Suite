# filters/obstruction_filter.py

from typing import List, Tuple

def iou(
    boxA: Tuple[float, float, float, float],
    boxB: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection-over-Union (IoU) between two boxes (x1,y1,x2,y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    if interArea == 0.0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def is_obstructed(
    person_boxes: List[Tuple[float, float, float, float]],
    face_boxes:   List[Tuple[float, float, float, float]],
    min_iou:      float = 0.5
) -> bool:
    """
    Returns True if any person box has no corresponding face box
    whose center lies inside it AND whose IoU ≥ min_iou.
    This flags truly obstructed person detections.
    """
    for p in person_boxes:
        px1, py1, px2, py2 = p
        matched = False
        for f in face_boxes:
            fx1, fy1, fx2, fy2 = f
            # face center
            cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if iou(p, f) >= min_iou:
                    matched = True
                    break
        if not matched:
            # this person bbox has no valid face → likely obstructed
            return True
    return False
