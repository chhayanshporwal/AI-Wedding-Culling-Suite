# filters/face_id_filter.py

import os
import json
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from typing import Dict, List, Tuple
import logging

import config
from logging_config import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Directory and index paths from config
EMB_DIR    = config.EMBEDDINGS_DIR
# INDEX_FILE isn't in config; define here
INDEX_FILE = os.path.join(EMB_DIR, "index.json")

# In-memory map: VIP name → averaged, normalized embedding
VIP_EMBS: Dict[str, np.ndarray] = {}

# Unified face detector: MTCNN for both enrollment and matching
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# FaceNet embedding model
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def load_vip_embeddings() -> None:
    """
    Load per-VIP embeddings from INDEX_FILE, average them, normalize,
    and store in VIP_EMBS for fast lookup.
    """
    global VIP_EMBS
    if VIP_EMBS:
        return
    if not os.path.exists(INDEX_FILE):
        # Silently skip if index is missing (no enrolled VIPs yet)
        logger.debug("VIP index file not found: %s", INDEX_FILE)
        return
    try:
        with open(INDEX_FILE, "r") as f:
            idx = json.load(f)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse VIP index: %s", e)
        return

    for person, fnames in idx.items():
        mats: List[np.ndarray] = []
        for fn in fnames:
            path = os.path.join(EMB_DIR, fn)
            if os.path.exists(path):
                mats.append(np.load(path))
            else:
                logger.warning("Embedding file missing: %s", path)
        if mats:
            mean_emb = np.mean(np.vstack(mats), axis=0)
            VIP_EMBS[person] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        else:
            logger.warning("No embeddings loaded for VIP: %s", person)


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces using MTCNN, returning pixel-space boxes (x1, y1, x2, y2).
    """
    try:
        pil = Image.fromarray(image[..., ::-1])
        found = mtcnn.detect(pil)
        boxes = found[0] if found and found[0] is not None else None
        if boxes is None:
            return []
        result: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.tolist())
            result.append((x1, y1, x2, y2))
        return result
    except Exception as e:
        logger.error("Error during face detection: %s", e)
        return []

def match_vips(image: np.ndarray, thresh: float = config.VIP_COSINE_THRESH) -> List[Tuple[str, float]]:
    """
    Detect faces, compute embeddings, and return list of
    (VIP_name, cosine_similarity) for all VIPs with similarity ≥ thresh.
    """
    load_vip_embeddings()
    if not VIP_EMBS:
        return []

    face_boxes = detect_faces(image)
    if not face_boxes:
        return []

    pil_img = Image.fromarray(image[..., ::-1])
    matches: List[Tuple[str, float]] = []

    for (x1, y1, x2, y2) in face_boxes:
        try:
            crop = pil_img.crop((x1, y1, x2, y2)).resize((160, 160))
            arr = np.array(crop)
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = (t - 0.5) / 0.5
            t = t.to(device)
            with torch.no_grad():
                emb = resnet(t).cpu().numpy()[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
        except Exception as e:
            logger.warning("Failed to compute embedding for face: %s", e)
            continue

        for person, vip_emb in VIP_EMBS.items():
            cos = float(np.dot(emb, vip_emb))
            if cos >= thresh:
                matches.append((person, cos))

    return matches
