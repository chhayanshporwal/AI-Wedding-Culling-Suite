# scripts/enroll_vips.py
# Industry-grade VIP enrolment – case-insensitive, instant cache refresh.

import os
import json
import glob
import tempfile
import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import config
from logging_config import setup_logging
from filters.face_id_filter import load_vip_embeddings   # instant refresh

setup_logging()
logger = logging.getLogger(__name__)

os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


def enroll(person: str, folder: str) -> None:
    """
    Enrol all face images in `folder` under VIP `person`.

    - Detect the largest/only face per image.
    - Compute FaceNet embedding.
    - Persist to disk.
    - Atomically update index.json.
    - Force in-memory cache refresh for immediate GUI/API visibility.
    """
    person_dir = os.path.join(config.EMBEDDINGS_DIR, person)
    os.makedirs(person_dir, exist_ok=True)

    # Case-insensitive glob
    patterns = [
        "**/*.[jJ][pP][gG]",
        "**/*.[jJ][pP][eE][gG]",
        "**/*.[pP][nN][gG]",
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, pat), recursive=True))
    files = sorted(files)

    if not files:
        logger.warning("No VIP images in %s", folder)
        return

    saved = []
    for idx, fpath in enumerate(files):
        try:
            img = Image.open(fpath).convert("RGB")
        except Exception as e:
            logger.warning("Unreadable %s (%s)", fpath, e)
            continue

        boxes, *_ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            logger.info("No face in %s", fpath)
            continue

        x1, y1, x2, y2 = map(int, boxes[0])
        face = img.crop((x1, y1, x2, y2)).resize((160, 160))

        arr = np.array(face)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(device)

        with torch.no_grad():
            emb = resnet(tensor).cpu().numpy()[0]

        fn = f"{person}_{idx:03d}.npy"
        out_path = os.path.join(person_dir, fn)
        np.save(out_path, emb)
        saved.append(os.path.join(person, fn))
        logger.debug("Saved %s", out_path)

    idx_path = os.path.join(config.EMBEDDINGS_DIR, "index.json")
    os.makedirs(os.path.dirname(idx_path), exist_ok=True)

    try:
        with open(idx_path, "r") as jf:
            idx_map = json.load(jf)
    except (FileNotFoundError, json.JSONDecodeError):
        idx_map = {}

    idx_map[person] = saved
    with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(idx_path), delete=False) as tf:
        json.dump(idx_map, tf, indent=2)
        tmp = tf.name
    os.replace(tmp, idx_path)

    logger.info("Enrolled %d embeddings for '%s'", len(saved), person)

    # Ensure any running GUI/API sees the new VIP immediately
    load_vip_embeddings()
    logger.debug("VIP cache flushed & reloaded.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enrol VIP embeddings")
    parser.add_argument("person", help="VIP name")
    parser.add_argument("folder", help="Folder with VIP portraits")
    args = parser.parse_args()
    setup_logging()
    enroll(args.person, args.folder)